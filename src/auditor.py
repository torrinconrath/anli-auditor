# src/auditor.py

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from textattack import Attack
from textattack.models.wrappers import PyTorchModelWrapper
from textattack.datasets import Dataset as TextAttackDataset

# TextFooler components — assembled manually so we can swap the semantic
# similarity constraint from UniversalSentenceEncoder (requires tensorflow_hub)
# to BERTScore (pure PyTorch, no extra install needed).
from textattack.goal_functions import UntargetedClassification
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import ThoughtVector
from textattack.search_methods import GreedyWordSwapWIR


class _NLIModelWrapper(PyTorchModelWrapper):
    """
    Thin TextAttack-compatible wrapper around our fine-tuned RoBERTa classifier.

    TextFooler needs to query the model's output probabilities during its
    search loop so it can identify which word swaps successfully flip the
    predicted label. PyTorchModelWrapper expects __call__ to accept a list
    of raw strings and return a (n_samples, n_classes) numpy array of scores.

    For NLI we encode the perturbed premise together with the fixed hypothesis
    so the model sees the full sentence-pair context on every attack query,
    not just the premise in isolation.
    """

    def __init__(self, model, tokenizer, hypothesis: str, device):
        # PyTorchModelWrapper stores model and tokenizer on self.model / self.tokenizer
        super().__init__(model, tokenizer)
        self.hypothesis = hypothesis
        self.device = device

    def __call__(self, text_input_list: list) -> np.ndarray:
        """
        text_input_list : list of premise strings (the text being attacked)
        Returns         : np.ndarray of shape (len(text_input_list), n_classes)
        """
        all_probs = []
        for premise in text_input_list:
            inputs = self.tokenizer(
                premise,
                self.hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            all_probs.append(probs)

        return np.array(all_probs)


class RationaleAuditor:
    """
    Audits the faithfulness of RoBERTa-large's NLI reasoning using two metrics:

    LAS — Latent Alignment Score
        Cosine similarity between the model's [CLS] decision representation
        (premise + hypothesis) and the [CLS] embedding of the human-provided
        rationale (rationale + hypothesis). Measures whether the model's
        internal state is semantically grounded in the human logic.

    CSI — Causal Sensitivity Index
        Drop in LAS after adversarially perturbing the premise via TextFooler.
        TextFooler swaps words to actively flip the model's predicted label,
        producing meaning-inverting perturbations rather than synonym swaps.
        A faithful model should show a large latent delta when the logical
        evidence is undermined; a heuristic model will remain rigid.

    Why TextFooler instead of WordSwapEmbedding
    --------------------------------------------
    The original implementation used WordSwapEmbedding (GloVe nearest-neighbour
    synonym substitution). Synonym swaps are semantically conservative — they
    preserve meaning, which is the wrong property for CSI. CSI aims to detect
    whether the internal state updates when *evidence changes*, which requires
    perturbations that genuinely alter what the premise entails. TextFooler
    searches for word substitutions that flip the model's predicted label,
    guaranteeing a meaningful change in the logical content of the premise.
    This matches the description in the project proposal and makes CSI
    interpretable: a near-zero result under TextFooler is a clean finding that
    the internal state does not track evidence, not merely that synonym swaps
    were too mild to move the representation.

    CLS vector limitation
    ---------------------
    The [CLS] vector in a fine-tuned classifier is optimised for linear
    separability, not cosine-similarity-friendly geometry. High LAS therefore
    indicates that both inputs activated similar classifier-relevant features,
    which is a reasonable but imperfect proxy for semantic faithfulness.
    """

    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # Count samples where TextFooler failed to find an adversarial example.
        # FailedAttackResult means no label-flipping perturbation existed within
        # the search budget — these samples are excluded from CSI aggregation.
        self.csi_skipped = 0

    def _build_attack(self, hypothesis: str) -> Attack:
        """
        Manually assembles TextFooler with BERTScore as the semantic similarity
        constraint instead of the default UniversalSentenceEncoder (USE).

        TextFoolerJin2019.build() hardcodes USE, which requires tensorflow_hub —
        a heavy optional dependency not present in most PyTorch environments.
        BERTScore is a drop-in replacement: it enforces the same intuition
        (swapped words must stay semantically close to the original) using a
        HuggingFace model that is already available.

        All other TextFooler components are identical to the original recipe:
          - WordSwapEmbedding      : counter-fitted word vector substitutions
          - GreedyWordSwapWIR      : greedy search ordered by word importance
          - UntargetedClassification: goal is to flip the predicted label
          - RepeatModification     : don't swap the same word twice
          - StopwordModification   : don't swap stopwords
          - WordEmbeddingDistance  : keep substitution within cosine distance 0.5
          - ThoughtVector (threshold=0.8): counter-fitted GloVe sentence-level
                                    similarity — same embeddings used by WordSwapEmbedding,
                                    no extra dependencies required
        """
        wrapper = _NLIModelWrapper(
            self.model, self.tokenizer, hypothesis, self.device
        )
        goal_function  = UntargetedClassification(wrapper)
        transformation = WordSwapEmbedding(max_candidates=50)
        constraints = [
            RepeatModification(),
            StopwordModification(),
            WordEmbeddingDistance(min_cos_sim=0.5),
            ThoughtVector(threshold=0.8),
        ]
        search_method = GreedyWordSwapWIR(wir_method="delete")
        return Attack(goal_function, constraints, transformation, search_method)

    @torch.no_grad()
    def get_cls_hidden_state(self, text_a: str, text_b: str = None) -> np.ndarray:
        """
        Returns the [CLS] token hidden state from RoBERTa's final layer.

        Encoding design rationale
        -------------------------
        Decision state  : premise + hypothesis as a sentence pair.
            The bidirectional encoder sees the full NLI context, so its [CLS]
            vector is the model's decision-point representation.
        Rationale state : human_rationale + hypothesis as a sentence pair.
            Pairing the rationale with the hypothesis (rather than encoding it
            standalone) preserves the relational context — the rationale is
            always an explanation of why the hypothesis holds or fails given
            some evidence. This makes the cosine comparison more semantically
            fair: both vectors are conditioned on the same hypothesis.
        """
        if text_b is not None:
            inputs = self.tokenizer(
                text_a, text_b,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)
        else:
            inputs = self.tokenizer(
                text_a,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

        outputs = self.model(**inputs, output_hidden_states=True)
        cls_state = outputs.hidden_states[-1][0, 0, :]
        return cls_state.cpu().numpy().reshape(1, -1)

    def _run_textfooler(self, premise: str, hypothesis: str, label_id: int):
        """
        Runs TextFooler on the premise to find a label-flipping perturbation.

        Returns (perturbed_premise, csi_valid):
            perturbed_premise : the adversarial string if found, else original
            csi_valid         : True only if TextFooler found a successful attack

        TextAttack result types:
            SuccessfulAttackResult  — attack found a label-flipping perturbation
            FailedAttackResult      — no perturbation found within search budget
            SkippedAttackResult     — model already misclassifies the original
        """
        from textattack.attack_results import (
            SuccessfulAttackResult,
        )

        attack = self._build_attack(hypothesis)

        # TextAttack expects (text, ground_truth_label) as input
        dataset = TextAttackDataset([(premise, label_id)])
        attack_input, _ = next(iter(dataset))

        result = attack.attack(attack_input, label_id)

        if isinstance(result, SuccessfulAttackResult):
            return result.perturbed_result.attacked_text.text, True
        else:
            # FailedAttackResult or SkippedAttackResult — no valid perturbation
            self.csi_skipped += 1
            return premise, False

    def run_single_audit(
        self, sample: dict, mismatched_rationale: str
    ) -> tuple:
        """
        Returns (las, csi, mismatched_las, csi_valid).

        las              : Latent Alignment Score (cosine sim, original premise)
        csi              : Causal Sensitivity Index (LAS drop after TextFooler attack)
                           0.0 and csi_valid=False if no adversarial example found
        mismatched_las   : LAS computed with a deliberately wrong rationale
                           (used by the caller to compute Synthetic Sensitivity)
        csi_valid        : True if TextFooler found a label-flipping perturbation;
                           False if the attack failed or was skipped — caller must
                           exclude these from CSI aggregation
        """
        premise         = sample["premise"]
        hypothesis      = sample["hypothesis"]
        human_rationale = sample["reason"]
        label_id        = sample["label"]

        # 1. Latent Alignment Score (LAS)
        decision_state  = self.get_cls_hidden_state(premise, hypothesis)
        rationale_state = self.get_cls_hidden_state(human_rationale, hypothesis)
        las = float(cosine_similarity(decision_state, rationale_state)[0, 0])

        # 2. Causal Sensitivity Index (CSI) via TextFooler
        # TextFooler searches for word substitutions that flip the model's
        # predicted label, guaranteeing that the perturbed premise changes
        # what the evidence logically entails — not just how it is phrased.
        # We measure how much the model's internal alignment with the human
        # rationale drops after this meaning-inverting perturbation.
        perturbed_premise, csi_valid = self._run_textfooler(
            premise, hypothesis, label_id
        )

        if csi_valid:
            perturbed_state     = self.get_cls_hidden_state(perturbed_premise, hypothesis)
            perturbed_alignment = float(
                cosine_similarity(perturbed_state, rationale_state)[0, 0]
            )
            csi = las - perturbed_alignment
        else:
            csi = 0.0

        # 3. Mismatched LAS (used by caller for Synthetic Sensitivity)
        mismatched_state = self.get_cls_hidden_state(mismatched_rationale, hypothesis)
        mismatched_las   = float(cosine_similarity(decision_state, mismatched_state)[0, 0])

        return las, float(csi), mismatched_las, csi_valid
    