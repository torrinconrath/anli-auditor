# src/auditor.py

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.augmentation import Augmenter


class RationaleAuditor:
    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        transformation = WordSwapEmbedding(max_candidates=10)
        constraints = [RepeatModification(), StopwordModification()]
        self.augmenter = Augmenter(
            transformation=transformation,
            constraints=constraints,
            pct_words_to_swap=0.25, # 25% CSI shifts
            transformations_per_example=1,
        )

    @torch.no_grad()
    def get_cls_hidden_state(self, text_a: str, text_b: str = None) -> np.ndarray:
        """
        Returns the [CLS] token hidden state from RoBERTa's final layer.
        For a premise/hypothesis pair, pass both as text_a and text_b so
        the model encodes their relationship bidirectionally.
        For a standalone rationale string, pass only text_a.
        [CLS] in RoBERTa is mean-pooled during pre-training to represent
        the full sequence, making it the natural decision-point representation.
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
        # [CLS] is always at position 0 in RoBERTa
        cls_state = outputs.hidden_states[-1][0, 0, :]
        return cls_state.cpu().numpy().reshape(1, -1)

    def run_single_audit(self, sample: dict, mismatched_rationale: str) -> tuple:
        premise = sample["premise"]
        hypothesis = sample["hypothesis"]
        human_rationale = sample["reason"]

        # 1. Latent Alignment Score (LAS)
        # [CLS] state of the premise+hypothesis pair encodes the model's
        # decision representation. Compare to [CLS] of the human rationale
        # to measure semantic alignment between internal state and human reasoning.
        decision_state   = self.get_cls_hidden_state(premise, hypothesis)
        rationale_state  = self.get_cls_hidden_state(human_rationale, hypothesis)
        las = cosine_similarity(decision_state, rationale_state)[0, 0]

        # 2. Causal Sensitivity Index (CSI)
        # Perturb the premise adversarially and measure how much the alignment
        # with the human rationale drops. High CSI = model is sensitive to
        # evidence changes, consistent with faithful reasoning.
        perturbed_premise = self.augmenter.augment(premise)[0]
        perturbed_state   = self.get_cls_hidden_state(perturbed_premise, hypothesis)
        perturbed_alignment = cosine_similarity(perturbed_state, rationale_state)[0, 0]
        csi = float(las - perturbed_alignment)

        # 3. Mismatched LAS (Used for calculating Synthetic Sensitivity later)
        mismatched_state = self.get_cls_hidden_state(mismatched_rationale, hypothesis)
        mismatched_las = cosine_similarity(decision_state, mismatched_state)[0, 0]

        return float(las), float(csi), float(mismatched_las)