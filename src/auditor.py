# src/auditor.py

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from textattack import Attack
from textattack.models.wrappers import PyTorchModelWrapper
from textattack.datasets import Dataset as TextAttackDataset
from textattack.goal_functions import UntargetedClassification
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import ThoughtVector
from textattack.search_methods import GreedyWordSwapWIR
from textattack.attack_results import SuccessfulAttackResult


class _NLIModelWrapper(PyTorchModelWrapper):
    """
    TextAttack-compatible wrapper for our fine-tuned RoBERTa NLI classifier.

    TextFooler queries the model repeatedly during its greedy search to
    identify which word swaps flip the predicted label. The wrapper encodes
    each candidate perturbed premise together with the fixed hypothesis so
    every query sees the full NLI sentence-pair context.
    """

    def __init__(self, model, tokenizer, hypothesis: str, device):
        super().__init__(model, tokenizer)
        self.hypothesis = hypothesis
        self.device = device

    def __call__(self, text_input_list: list) -> np.ndarray:
        all_probs = []
        for premise in text_input_list:
            inputs = self.tokenizer(
                premise, self.hypothesis,
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
    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.csi_skipped = 0

    def _build_attack(self, hypothesis: str) -> Attack:
        """
        Assembles TextFooler using ThoughtVector in place of the default
        UniversalSentenceEncoder constraint, avoiding the tensorflow_hub
        dependency. All other components match the original Jin et al. recipe:
        counter-fitted embedding swaps, greedy word importance ranking, and
        an untargeted label-flip objective.
        """
        wrapper = _NLIModelWrapper(self.model, self.tokenizer, hypothesis, self.device)
        return Attack(
            UntargetedClassification(wrapper),
            [
                RepeatModification(),
                StopwordModification(),
                WordEmbeddingDistance(min_cos_sim=0.5),
                ThoughtVector(threshold=0.8),
            ],
            WordSwapEmbedding(max_candidates=50),
            GreedyWordSwapWIR(wir_method="delete"),
        )

    @torch.no_grad()
    def get_cls_hidden_state(self, text_a: str, text_b: str = None) -> np.ndarray:
        """
        Returns the [CLS] token hidden state from RoBERTa's final layer.
        Pass text_b to encode a sentence pair (premise+hypothesis or
        rationale+hypothesis); omit it for standalone text.
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

    def _run_textfooler(self, premise: str, hypothesis: str, label_id: int) -> tuple:
        """
        Runs TextFooler on the premise to find a label-flipping perturbation.
        Returns (perturbed_premise, csi_valid). csi_valid is False when
        TextFooler fails (FailedAttackResult) or the sample is already
        misclassified (SkippedAttackResult); those samples are excluded from
        CSI aggregation by the caller.
        """
        attack = self._build_attack(hypothesis)
        dataset = TextAttackDataset([(premise, label_id)])
        attack_input, _ = next(iter(dataset))
        result = attack.attack(attack_input, label_id)

        if isinstance(result, SuccessfulAttackResult):
            return result.perturbed_result.attacked_text.text, True

        self.csi_skipped += 1
        return premise, False

    def run_single_audit(self, sample: dict, mismatched_rationale: str) -> tuple:
        """
        Returns (las, csi, mismatched_las, csi_valid).

        LAS           — cosine similarity between the model's [CLS] decision
                        representation (premise+hypothesis) and the [CLS] of the
                        human rationale (rationale+hypothesis).
        CSI           — drop in LAS after TextFooler finds a label-flipping
                        perturbation of the premise. 0.0 when csi_valid=False.
        mismatched_las — LAS with a deliberately wrong rationale, used by the
                        caller to compute Synthetic Sensitivity.
        csi_valid     — True only if TextFooler found a successful attack.
        """
        premise         = sample["premise"]
        hypothesis      = sample["hypothesis"]
        human_rationale = sample["reason"]
        label_id        = sample["label"]

        # 1. Latent Alignment Score (LAS)
        # Both states are conditioned on the hypothesis so the cosine comparison
        # is symmetric — the decision vector and rationale vector are both shaped
        # by the same relational context.
        decision_state  = self.get_cls_hidden_state(premise, hypothesis)
        rationale_state = self.get_cls_hidden_state(human_rationale, hypothesis)
        las = float(cosine_similarity(decision_state, rationale_state)[0, 0])

        # 2. Causal Sensitivity Index (CSI) via TextFooler
        # TextFooler's label-flip objective guarantees the perturbed premise
        # changes what the evidence logically entails, not merely how it is
        # phrased. A faithful model should show a measurable latent delta;
        # a heuristic model's internal state will remain rigid.
        perturbed_premise, csi_valid = self._run_textfooler(premise, hypothesis, label_id)

        if csi_valid:
            perturbed_state     = self.get_cls_hidden_state(perturbed_premise, hypothesis)
            perturbed_alignment = float(cosine_similarity(perturbed_state, rationale_state)[0, 0])
            csi = las - perturbed_alignment
        else:
            csi = 0.0

        # 3. Mismatched LAS (for Synthetic Sensitivity)
        mismatched_state = self.get_cls_hidden_state(mismatched_rationale, hypothesis)
        mismatched_las   = float(cosine_similarity(decision_state, mismatched_state)[0, 0])

        return las, float(csi), mismatched_las, csi_valid
    