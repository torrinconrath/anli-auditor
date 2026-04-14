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
        self.device = model.device

        transformation = WordSwapEmbedding(max_candidates=10)
        constraints = [RepeatModification(), StopwordModification()]
        self.augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0.1, transformations_per_example=1)

    @torch.no_grad()
    def get_hidden_state_at_decision_point(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1][0, -1, :].cpu().numpy().reshape(1, -1)

    @torch.no_grad()
    def get_text_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        embedding_layer = self.model.get_input_embeddings()
        embeddings = embedding_layer(inputs.input_ids)
        return torch.mean(embeddings, dim=1).cpu().numpy()

    def run_single_audit(self, sample):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        human_rationale = sample['reason']
        
        inference_prompt = f"""Analyze the following premise and hypothesis to determine the relationship. First, provide a step-by-step rationale, and then conclude with the final label (Entailment, Neutral, or Contradiction).

### Premise:
{premise}

### Hypothesis:
{hypothesis}

### Rationale:"""
        
        # --- Latent Alignment Score (LAS) ---
        original_hidden_state = self.get_hidden_state_at_decision_point(inference_prompt)
        human_rationale_embedding = self.get_text_embedding(human_rationale)
        las = cosine_similarity(original_hidden_state, human_rationale_embedding)[0, 0]

        # --- Causal Sensitivity Index (CSI) ---
        perturbed_premise = self.augmenter.augment(premise)[0]
        perturbed_prompt = inference_prompt.replace(premise, perturbed_premise)
        
        perturbed_hidden_state = self.get_hidden_state_at_decision_point(perturbed_prompt)
        perturbed_alignment = cosine_similarity(perturbed_hidden_state, human_rationale_embedding)[0, 0]
        csi = las - perturbed_alignment
        
        return las, csi
    