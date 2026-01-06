from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# -----------------------
# Device helper
# -----------------------
def device_for_torch() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------
# Tokenizer helper
# -----------------------
def build_tokenizer(pretrained_name: str):
    tok = AutoTokenizer.from_pretrained(pretrained_name, use_fast=True)

    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.pad_token = tok.sep_token

    return tok


# -----------------------
# Outputs container
# -----------------------
@dataclass
class ModelOutputs:
    logits: torch.Tensor
    probs: torch.Tensor
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


# -----------------------
# Attention rollout helper 
# -----------------------
def attention_rollout(
    attentions: Tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    start_layer: int = 0,
) -> torch.Tensor:
    
    # Stack -> (L, B, H, S, S)
    att = torch.stack(attentions, dim=0)

    # Average heads -> (L, B, S, S)
    att = att.mean(dim=2)

    # Apply attention mask 
    # mask: (B, 1, 1, S)
    B, S = attention_mask.shape
    key_mask = attention_mask.view(B, 1, 1, S).float()
    att = att * key_mask  # broadcast to (L, B, S, S)

    # Add residual connection & normalize
    eye = torch.eye(S, device=att.device).view(1, 1, S, S)
    att = att + eye
    att = att / (att.sum(dim=-1, keepdim=True) + 1e-12)

    # Rollout multiplication
    joint = att[start_layer]
    for layer in range(start_layer + 1, att.size(0)):
        joint = att[layer].bmm(joint)

    return joint  # (B, S, S)


# -----------------------
# Model
# -----------------------
class SentimentClassifier(nn.Module):
    

    def __init__(
        self,
        pretrained_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
        pooling: str = "cls",  
    ):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.num_labels = num_labels
        self.pooling = pooling

        try:
            self.encoder = AutoModel.from_pretrained(pretrained_name, attn_implementation="eager")
        except TypeError:
            self.encoder = AutoModel.from_pretrained(pretrained_name)

        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.encoder.config, "dim")

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        
        if self.pooling == "cls":
            return last_hidden[:, 0, :]  
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()  
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return summed / denom
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}. Use 'cls' or 'mean'.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attentions: bool = False,
        return_hidden_states: bool = False,
        return_rollout: bool = False,
        rollout_start_layer: int = 0,
    ) -> ModelOutputs:
        
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attentions or return_rollout,
            output_hidden_states=return_hidden_states,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state  # (B, S, H)
        pooled = self._pool(last_hidden, attention_mask)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)

        attentions = outputs.attentions if (return_attentions or return_rollout) else None
        hidden_states = outputs.hidden_states if return_hidden_states else None

        
        if return_rollout and attentions is not None:
            rollout = attention_rollout(attentions, attention_mask, start_layer=rollout_start_layer) 
        
            rollout = rollout.unsqueeze(1)
            attentions = tuple(attentions) + (rollout,)

        return ModelOutputs(logits=logits, probs=probs, attentions=attentions, hidden_states=hidden_states)
