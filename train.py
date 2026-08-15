#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train RISE.

RISE performs reliability-aware sequence-structure learning for antimicrobial
peptide prediction. The training workflow contains:
  1) sequence/structure evidential prediction;
  2) neighborhood-based candidate identification;
  3) pLDDT-based structural reliability conditioning;
  4) multi-source soft pseudo-label construction and verification;
  5) recursive effective-supervision refinement.

Feature preparation is implemented in data/ and features/.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from data.csv_dataset import load_csv_split
from data.fasta import (
    build_clean_labels_from_reference,
    load_binary_split,
    norm_seq,
    resolve_noise_source,
)
from features.sequence_features import extract_token_embeddings
from features.structure_features import extract_graphs, get_mean_graph_features


# =============================================================================
# Basic utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_auc(y_true, prob) -> float:
    try:
        return float(roc_auc_score(y_true, prob))
    except Exception:
        return float("nan")


def binary_metrics(y_true, prob_amp, ids=None, seqs=None):
    y_true = np.asarray(y_true).astype(np.int64)
    prob_amp = np.asarray(prob_amp).astype(np.float64)
    pred = (prob_amp >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    metrics = {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "Sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp + 1e-12)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "AUC": safe_auc(y_true, prob_amp),
        "MCC": float(matthews_corrcoef(y_true, pred)),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }

    rows = []
    for i in range(len(y_true)):
        rows.append(
            {
                "id": ids[i] if ids is not None else str(i),
                "seq": seqs[i] if seqs is not None else "",
                "true_label": int(y_true[i]),
                "prob_amp": float(prob_amp[i]),
                "pred_label": int(pred[i]),
            }
        )
    return metrics, rows


def to_onehot(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    return F.one_hot(labels.long().view(-1), num_classes=num_classes).float()


def inject_symmetric_noise(
    labels: np.ndarray,
    train_idx: np.ndarray,
    noise_rate: float,
    seed: int,
):
    """Fallback noise injection for FASTA inputs only."""
    noisy = labels.copy().astype(np.int64)
    mask = np.zeros_like(noisy, dtype=bool)
    if noise_rate <= 0:
        return noisy, mask

    rng = np.random.default_rng(seed)
    n_flip = int(round(len(train_idx) * noise_rate))
    if n_flip <= 0:
        return noisy, mask

    flip = rng.choice(train_idx, size=n_flip, replace=False)
    noisy[flip] = 1 - noisy[flip]
    mask[flip] = True
    return noisy, mask


# =============================================================================
# Evidence and evidential losses
# =============================================================================

def evidence_to_probs(evidence: torch.Tensor) -> torch.Tensor:
    """Dirichlet predictive probability p=(e+1)/sum(e+1)."""
    alpha = torch.clamp(evidence, min=0.0) + 1.0
    return alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)


def evidence_to_opinion(evidence: torch.Tensor):
    evidence = torch.clamp(evidence, min=0.0)
    k = evidence.size(1)
    strength = evidence.sum(dim=1, keepdim=True) + float(k)
    belief = evidence / strength
    uncertainty = float(k) / strength
    return belief, uncertainty


def ds_evidence_fusion(
    e1: torch.Tensor,
    e2: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Dempster-Shafer fusion in the evidential-opinion space."""
    k = e1.size(1)
    e1 = torch.clamp(e1, min=0.0)
    e2 = torch.clamp(e2, min=0.0)

    s1 = e1.sum(dim=1, keepdim=True) + float(k)
    s2 = e2.sum(dim=1, keepdim=True) + float(k)
    b1, b2 = e1 / s1, e2 / s2
    u1, u2 = float(k) / s1, float(k) / s2

    pairwise = b1.unsqueeze(2) * b2.unsqueeze(1)
    diagonal = torch.diagonal(pairwise, dim1=1, dim2=2).sum(dim=1, keepdim=True)
    conflict = (
        pairwise.sum(dim=(1, 2), keepdim=False).view(-1, 1) - diagonal
    ).clamp(min=0.0, max=1.0 - eps)

    denominator = (1.0 - conflict).clamp_min(eps)
    belief = (b1 * b2 + b1 * u2 + b2 * u1) / denominator
    uncertainty = (u1 * u2) / denominator
    fused = belief * (float(k) / uncertainty.clamp_min(eps))
    return torch.clamp(fused, min=0.0)


def discount_structural_prediction(
    structural_evidence: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """Transfer unsupported structural belief to uncertainty."""
    belief, uncertainty = evidence_to_opinion(structural_evidence)
    k = structural_evidence.size(1)
    rho = rho.view(-1, 1).clamp(0.0, 1.0)
    adjusted_belief = rho * belief
    adjusted_uncertainty = 1.0 - rho + rho * uncertainty
    prediction = adjusted_belief + adjusted_uncertainty / float(k)
    return prediction / prediction.sum(dim=1, keepdim=True).clamp_min(1e-12)


def dirichlet_kl(alpha, num_classes, device):
    beta = torch.ones((1, num_classes), device=device)
    s_alpha = torch.sum(alpha, dim=1, keepdim=True)
    s_beta = torch.sum(beta, dim=1, keepdim=True)
    ln_b = torch.lgamma(s_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    ln_b_uniform = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(s_beta)
    dg0 = torch.digamma(s_alpha)
    dg1 = torch.digamma(alpha)
    return (
        torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True)
        + ln_b
        + ln_b_uniform
    )


def edl_ce_loss(y_onehot, alpha, num_classes, step, annealing_step, device):
    strength = torch.sum(alpha, dim=1, keepdim=True)
    evidence = alpha - 1.0
    ce = torch.sum(
        y_onehot * (torch.digamma(strength) - torch.digamma(alpha)),
        dim=1,
        keepdim=True,
    )
    adjusted_alpha = evidence * (1.0 - y_onehot) + 1.0
    annealing = min(1.0, float(step) / float(max(1, annealing_step)))
    return ce + annealing * dirichlet_kl(
        adjusted_alpha, num_classes, device
    )


def confidence_transition_loss(conf, transition, y, num_classes, device):
    """Confidence regularization for a view-specific transition matrix."""
    conf = conf.view(-1)
    y = y.long().view(-1)
    diagonal = torch.diagonal(transition, offset=0, dim1=-2, dim2=-1)

    class_sum = torch.zeros(num_classes, device=device)
    class_count = torch.bincount(y, minlength=num_classes).float().to(device)
    class_sum.scatter_add_(0, y, conf)
    class_mean = class_sum / (class_count + 1e-5)

    y_onehot = torch.zeros(len(y), num_classes, device=device)
    y_onehot.scatter_(1, y.view(-1, 1), 1.0)

    observed_loss = (
        (conf.view(-1, 1) - diagonal.view(1, -1)) ** 2 * y_onehot
    ).sum(dim=1)
    other_loss = (
        (class_mean.view(1, -1) - diagonal.view(1, -1)) ** 2
        * (1.0 - y_onehot)
    ).sum(dim=1)
    return observed_loss + other_loss


def refined_mse_loss(effective_y, fused_evidence, refined_mask):
    if refined_mask is None or not bool(refined_mask.any()):
        return None
    prediction = evidence_to_probs(fused_evidence)
    return torch.sum(
        (effective_y[refined_mask] - prediction[refined_mask]) ** 2,
        dim=1,
    ).mean()


# =============================================================================
# Dataset and batching
# =============================================================================

class SequenceStructureDataset(Dataset):
    def __init__(
        self,
        seq_data,
        graph_data,
        indices=None,
        labels_override=None,
        effective_supervision=None,
    ):
        if len(seq_data["labels"]) != len(graph_data["labels"]):
            raise ValueError("Sequence/structure sample-count mismatch.")
        self.seq_data = seq_data
        self.graph_data = graph_data
        self.indices = (
            list(range(len(seq_data["labels"])))
            if indices is None
            else [int(i) for i in indices]
        )
        self.labels_override = labels_override
        self.effective_supervision = effective_supervision

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real = int(self.indices[idx])
        y = int(self.seq_data["labels"][real])
        if self.labels_override is not None:
            y = int(self.labels_override[real])

        if self.effective_supervision is None:
            effective = np.zeros(2, dtype=np.float32)
            effective[y] = 1.0
        else:
            effective = self.effective_supervision[real].astype(np.float32)

        return {
            "id": self.seq_data["ids"][real],
            "seq": self.seq_data["seqs"][real],
            "seq_x": self.seq_data["embeddings"][real],
            "graph": self.graph_data["graphs"][real],
            "y": y,
            "idx": real,
            "effective_y": effective,
        }


def collate_sequence_structure(batch):
    seq_lengths = [int(item["seq_x"].shape[0]) for item in batch]
    max_len = max(seq_lengths)
    dim = int(batch[0]["seq_x"].shape[1])
    batch_size = len(batch)

    seq_x = torch.zeros((batch_size, max_len, dim), dtype=torch.float32)
    seq_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, item in enumerate(batch):
        x = item["seq_x"].float()
        length = x.size(0)
        seq_x[i, :length] = x
        seq_mask[i, :length] = True

    graph_x_parts, edge_parts, graph_batch_parts = [], [], []
    offset = 0
    for graph_index, item in enumerate(batch):
        graph = item["graph"]
        graph_x = graph["x"].float()
        edge_index = graph["edge_index"].long()
        node_count = graph_x.size(0)
        graph_x_parts.append(graph_x)
        if edge_index.numel() > 0:
            edge_parts.append(edge_index + offset)
        graph_batch_parts.append(
            torch.full((node_count,), graph_index, dtype=torch.long)
        )
        offset += node_count

    return {
        "seq_x": seq_x,
        "seq_mask": seq_mask,
        "seq_lengths": torch.tensor(seq_lengths, dtype=torch.long),
        "graph_x": torch.cat(graph_x_parts, dim=0),
        "edge_index": (
            torch.cat(edge_parts, dim=1)
            if edge_parts
            else torch.zeros((2, 0), dtype=torch.long)
        ),
        "graph_batch": torch.cat(graph_batch_parts, dim=0),
        "y": torch.tensor([item["y"] for item in batch], dtype=torch.long),
        "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
        "effective_y": torch.tensor(
            np.stack([item["effective_y"] for item in batch], axis=0),
            dtype=torch.float32,
        ),
        "id": [item["id"] for item in batch],
        "seq": [item["seq"] for item in batch],
    }


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


# =============================================================================
# RISE evidential backbone
# =============================================================================

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers, dropout, classifier_hidden):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        feature_dim = hidden_dim * 2
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, classifier_hidden),
            nn.Tanh(),
            nn.Linear(classifier_hidden, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = feature_dim

    def forward(self, x, mask, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.detach().cpu().long(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=x.size(1),
        )
        scores = self.attn(output).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=1)
        feature = torch.sum(output * weights.unsqueeze(-1), dim=1)
        return self.dropout(feature)


class GraphSAGELayer(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.neigh_lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            output = self.self_lin(x)
        else:
            source, target = edge_index[0], edge_index[1]
            aggregate = torch.zeros_like(x)
            aggregate.index_add_(0, target, x[source])
            degree = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
            aggregate = aggregate / degree.clamp(min=1.0).unsqueeze(-1)
            output = self.self_lin(x) + self.neigh_lin(aggregate)
        output = self.dropout(F.relu(output))
        return self.norm(x + output)


def attention_pool(h, batch, attention):
    graph_count = int(batch.max().item()) + 1 if batch.numel() else 0
    scores = attention(h).squeeze(-1)
    pooled = []
    for graph_index in range(graph_count):
        mask = batch == graph_index
        local_h = h[mask]
        local_scores = scores[mask]
        weights = torch.softmax(local_scores, dim=0).view(-1, 1)
        pooled.append(torch.sum(weights * local_h, dim=0))
    return torch.stack(pooled, dim=0)


class StructureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, classifier_hidden, dropout):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [GraphSAGELayer(hidden_dim, dropout) for _ in range(layers)]
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden),
            nn.Tanh(),
            nn.Linear(classifier_hidden, 1),
        )
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, batch):
        h = self.node_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        return attention_pool(h, batch, self.attn)


class EvidenceHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


class RISEModel(nn.Module):
    def __init__(self, seq_input_dim, graph_input_dim, args):
        super().__init__()
        self.fusion_type = args.fusion_type
        self.num_classes = 2

        self.seq_encoder = SequenceEncoder(
            seq_input_dim,
            args.hidden_dim,
            args.lstm_layers,
            args.dropout,
            args.classifier_hidden,
        )
        self.struct_encoder = StructureEncoder(
            graph_input_dim,
            args.gnn_hidden_dim,
            args.gnn_layers,
            args.classifier_hidden,
            args.dropout,
        )
        self.seq_head = EvidenceHead(
            self.seq_encoder.out_dim,
            args.classifier_hidden,
            args.dropout,
            2,
        )
        self.struct_head = EvidenceHead(
            self.struct_encoder.out_dim,
            args.classifier_hidden,
            args.dropout,
            2,
        )

        self.T_seq = nn.Parameter(torch.eye(self.num_classes))
        self.T_struct = nn.Parameter(torch.eye(self.num_classes))

    def encode(self, batch):
        h_seq = self.seq_encoder(
            batch["seq_x"], batch["seq_mask"], batch["seq_lengths"]
        )
        h_struct = self.struct_encoder(
            batch["graph_x"], batch["edge_index"], batch["graph_batch"]
        )
        return h_seq, h_struct

    def fuse_clean_evidence(self, e_seq, e_struct):
        if self.fusion_type == "ds":
            return ds_evidence_fusion(e_seq, e_struct)
        if self.fusion_type == "sum":
            return e_seq + e_struct
        if self.fusion_type == "mean":
            return 0.5 * (e_seq + e_struct)
        raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

    def evidence_from_features(self, h_seq, h_struct):
        e_seq = self.seq_head(h_seq)
        e_struct = self.struct_head(h_struct)
        e_fused = self.fuse_clean_evidence(e_seq, e_struct)
        return e_seq, e_struct, e_fused

    @staticmethod
    def transfer_evidence(evidence, transition):
        return evidence @ transition

    def forward(self, batch, apply_noise_correction=False, return_features=False):
        h_seq, h_struct = self.encode(batch)
        e_seq, e_struct, e_clean_fused = self.evidence_from_features(h_seq, h_struct)

        if apply_noise_correction:
            e_seq_transformed = self.transfer_evidence(e_seq, self.T_seq)
            e_struct_transformed = self.transfer_evidence(e_struct, self.T_struct)
            e_transformed_fused = ds_evidence_fusion(
                e_seq_transformed, e_struct_transformed
            )
        else:
            e_seq_transformed = e_seq
            e_struct_transformed = e_struct
            e_transformed_fused = e_clean_fused

        if return_features:
            return (
                e_seq,
                e_struct,
                e_clean_fused,
                e_seq_transformed,
                e_struct_transformed,
                e_transformed_fused,
                h_seq,
                h_struct,
            )
        return e_clean_fused, e_transformed_fused

    def transition_consistency_loss(self):
        # Manuscript: ||T^s - T^g||_F^2.
        return torch.sum((self.T_seq - self.T_struct) ** 2)

    @torch.no_grad()
    def normalize_transition_matrices_(self):
        for transition in (self.T_seq, self.T_struct):
            transition.data.clamp_(min=0.0)
            transition.data = transition.data / transition.data.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)


# =============================================================================
# Neighborhood representation and candidate identification
# =============================================================================

def get_mean_sequence_embeddings(seq_data):
    return np.stack(
        [embedding.mean(dim=0).numpy() for embedding in seq_data["embeddings"]],
        axis=0,
    ).astype(np.float32)


def build_joint_knn_features(seq_data, graph_data, args):
    seq_mean = get_mean_sequence_embeddings(seq_data)
    graph_mean = get_mean_graph_features(graph_data)

    # Normalize each view before concatenation so dimensional scale does not
    # make one representation dominate the neighborhood geometry.
    seq_mean = seq_mean / (
        np.linalg.norm(seq_mean, axis=1, keepdims=True) + 1e-8
    )
    graph_mean = graph_mean / (
        np.linalg.norm(graph_mean, axis=1, keepdims=True) + 1e-8
    )

    if args.knn_feature == "seq":
        return seq_mean.astype(np.float32)
    if args.knn_feature == "struct":
        return graph_mean.astype(np.float32)
    return np.concatenate([seq_mean, graph_mean], axis=1).astype(np.float32)


def build_similarity_and_neighbors(mean_features, train_idx, k, device):
    x = torch.tensor(mean_features, dtype=torch.float32, device=device)
    sample_count = x.size(0)
    distance_squared = torch.cdist(x, x, p=2) ** 2

    nearest = torch.zeros((sample_count, k), dtype=torch.long, device=device)
    sigma = torch.zeros((sample_count, 1), dtype=torch.float32, device=device)
    train_set = set(int(i) for i in np.asarray(train_idx).tolist())

    for i in range(sample_count):
        candidates = [j for j in train_set if j != i]
        if not candidates:
            candidates = [i]

        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
        order = torch.argsort(distance_squared[i, candidate_tensor])
        selected = candidate_tensor[order[: min(k, len(candidates))]]

        if selected.numel() < k:
            if selected.numel() > 0:
                padding = selected[-1:].repeat(k - selected.numel())
            else:
                padding = torch.tensor([i] * k, dtype=torch.long, device=device)
            selected = torch.cat([selected, padding], dim=0)

        nearest[i] = selected[:k]
        sigma[i] = torch.mean(distance_squared[i, nearest[i]]).clamp(min=1e-5)

    similarity = torch.exp(-distance_squared / (sigma ** 2))
    return similarity, nearest


def js_divergence(p, q, eps=1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    midpoint = 0.5 * (p + q)
    return 0.5 * torch.sum(p * (p.log() - midpoint.log()), dim=-1) + 0.5 * torch.sum(
        q * (q.log() - midpoint.log()), dim=-1
    )


def minmax_normalize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.min()) / (values.max() - values.min() + 1e-12)


@torch.no_grad()
def candidate_scores(
    fused_evidence,
    nearest_indices,
    similarity_matrix,
    train_idx,
    observed_labels,
    device,
):
    """Equation (9): neighborhood-weighted label/evidence inconsistency."""
    observed = torch.as_tensor(observed_labels, dtype=torch.long, device=device)
    observed_onehot = F.one_hot(observed, num_classes=2).float()
    train_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)

    # p^f is the predictive distribution induced by the Dirichlet opinion.
    fused_probs = evidence_to_probs(fused_evidence)

    self_and_neighbors = torch.cat(
        [train_tensor.view(-1, 1), nearest_indices[train_tensor]], dim=1
    )
    local_probs = fused_probs[self_and_neighbors]
    local_labels = observed_onehot[train_tensor].unsqueeze(1).expand_as(local_probs)
    divergence = js_divergence(local_labels, local_probs)

    rows = train_tensor.unsqueeze(1).expand(-1, nearest_indices.size(1))
    neighbor_similarity = similarity_matrix[rows, nearest_indices[train_tensor]]
    local_similarity = torch.cat(
        [torch.ones_like(neighbor_similarity[:, :1]), neighbor_similarity], dim=1
    )
    weights = torch.softmax(local_similarity, dim=1)
    score = torch.sum(weights * divergence, dim=1)

    normalized = minmax_normalize(score)
    all_scores = torch.zeros(
        fused_evidence.size(0), dtype=torch.float32, device=device
    )
    all_scores[train_tensor] = normalized
    return all_scores


@torch.no_grad()
def neighborhood_prediction(
    indices,
    fused_evidence,
    nearest_indices,
    similarity_matrix,
):
    """Equation (15): Softmax of weighted untransformed fused evidence."""
    if indices.numel() == 0:
        return torch.empty((0, fused_evidence.size(1)), device=fused_evidence.device)

    self_and_neighbors = torch.cat(
        [indices.view(-1, 1), nearest_indices[indices]], dim=1
    )
    local_evidence = fused_evidence[self_and_neighbors]

    rows = indices.unsqueeze(1).expand(-1, nearest_indices.size(1))
    neighbor_similarity = similarity_matrix[rows, nearest_indices[indices]]
    local_similarity = torch.cat(
        [torch.ones_like(neighbor_similarity[:, :1]), neighbor_similarity], dim=1
    )
    weights = torch.softmax(local_similarity, dim=1).unsqueeze(-1)
    aggregated_evidence = torch.sum(weights * local_evidence, dim=1)
    return torch.softmax(aggregated_evidence, dim=1)


# =============================================================================
# Structural reliability
# =============================================================================

def structural_reliability_from_graphs(
    graphs,
    train_idx,
    low_quantile=0.15,
    high_quantile=0.70,
):
    """Compute r_i and rho_i from residue-level normalized pLDDT.

    r_i = 0.50*mean + 0.30*q20 + 0.20*(1-low_confidence_ratio)
    with low-confidence threshold 0.50.
    """
    reliability_values = []

    for graph in graphs:
        x = graph.get("x")
        if (
            not isinstance(x, torch.Tensor)
            or x.ndim != 2
            or x.numel() == 0
            or x.size(1) == 0
        ):
            reliability_values.append(0.0)
            continue

        confidence = x[:, 0].detach().cpu().numpy().astype(np.float32)
        confidence = np.clip(confidence, 0.0, 1.0)
        if confidence.size == 0:
            reliability_values.append(0.0)
            continue

        mean_confidence = float(np.mean(confidence))
        q20 = float(np.quantile(confidence, 0.20))
        low_ratio = float(np.mean(confidence < 0.50))
        reliability = (
            0.50 * mean_confidence
            + 0.30 * q20
            + 0.20 * (1.0 - low_ratio)
        )
        reliability_values.append(float(np.clip(reliability, 0.0, 1.0)))

    reliability = np.asarray(reliability_values, dtype=np.float32)
    train_values = reliability[np.asarray(train_idx, dtype=np.int64)]
    r_low = float(np.quantile(train_values, low_quantile))
    r_high = float(np.quantile(train_values, high_quantile))
    if r_high <= r_low + 1e-8:
        r_high = min(1.0, r_low + 1e-6)

    rho = np.clip(
        (reliability - r_low) / (r_high - r_low + 1e-12),
        0.0,
        1.0,
    ).astype(np.float32)
    return reliability, rho, r_low, r_high


# =============================================================================
# Multi-source supervision refinement
# =============================================================================

@torch.no_grad()
def refine_supervision(
    *,
    threshold,
    fused_evidence,
    sequence_evidence,
    structural_evidence,
    nearest_indices,
    similarity_matrix,
    train_idx,
    observed_labels,
    previous_effective_supervision,
    structural_rho,
    historical_probs,
    history_initialized,
    history_beta,
    weight_sequence,
    weight_neighbor,
    weight_history,
    weight_structure,
    pseudo_confidence_threshold,
    margin_threshold,
    device,
    keep_soft_label=True,
):
    """One RISE supervision-assessment step.

    Samples that fail the current verification keep y^(t-1). This is deliberate:
    the effective supervision is recursive and is never reset to the initial
    observed label merely because a later proposal fails verification.
    """
    observed = torch.as_tensor(observed_labels, dtype=torch.long, device=device)
    train_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    previous_effective = torch.as_tensor(
        previous_effective_supervision,
        dtype=torch.float32,
        device=device,
    )
    corrected = previous_effective.clone()

    fused_probs = evidence_to_probs(fused_evidence)

    # Historical prediction p_h^(t).
    updated_history = historical_probs.clone()
    if not history_initialized:
        updated_history[train_tensor] = fused_probs[train_tensor].detach()
    else:
        updated_history[train_tensor] = (
            history_beta * historical_probs[train_tensor]
            + (1.0 - history_beta) * fused_probs[train_tensor].detach()
        )
        updated_history[train_tensor] = updated_history[train_tensor] / updated_history[
            train_tensor
        ].sum(dim=1, keepdim=True).clamp_min(1e-12)

    scores = candidate_scores(
        fused_evidence,
        nearest_indices,
        similarity_matrix,
        train_idx,
        observed_labels,
        device,
    )
    candidates = train_tensor[scores[train_tensor] >= threshold]

    candidate_mask = torch.zeros(
        len(observed_labels), dtype=torch.bool, device=device
    )
    accepted_mask = torch.zeros(
        len(observed_labels), dtype=torch.bool, device=device
    )
    candidate_mask[candidates] = True

    diagnostics = {
        "threshold": float(threshold),
        "candidate_count": int(candidates.numel()),
        "accepted_count": 0,
        "score_mean_train": float(scores[train_tensor].mean().item()),
        "score_std_train": float(scores[train_tensor].std().item()),
    }

    if candidates.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            corrected.cpu().numpy(),
            updated_history,
            True,
            diagnostics,
            scores,
            candidate_mask,
            accepted_mask,
        )

    p_sequence = evidence_to_probs(sequence_evidence[candidates])
    p_neighbor = neighborhood_prediction(
        candidates,
        fused_evidence,
        nearest_indices,
        similarity_matrix,
    )
    p_history = updated_history[candidates]
    p_structure = discount_structural_prediction(
        structural_evidence[candidates], structural_rho[candidates]
    )

    weights = torch.tensor(
        [
            weight_sequence,
            weight_neighbor,
            weight_history,
            weight_structure,
        ],
        dtype=torch.float32,
        device=device,
    )
    weights = weights / weights.sum().clamp_min(1e-12)

    pseudo = (
        weights[0] * p_sequence
        + weights[1] * p_neighbor
        + weights[2] * p_history
        + weights[3] * p_structure
    )
    pseudo = pseudo / pseudo.sum(dim=1, keepdim=True).clamp_min(1e-12)

    neighbor_class = torch.argmax(p_neighbor, dim=1)
    pseudo_class = torch.argmax(pseudo, dim=1)
    observed_candidate = observed[candidates]
    confidence = torch.max(pseudo, dim=1).values
    sorted_probs = torch.sort(pseudo, dim=1, descending=True).values
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]

    accepted = (
        (neighbor_class != observed_candidate)
        & (pseudo_class != observed_candidate)
        & (confidence >= pseudo_confidence_threshold)
        & (margin >= margin_threshold)
    )
    accepted_indices = candidates[accepted]
    accepted_pseudo = pseudo[accepted]

    if accepted_indices.numel() > 0:
        if keep_soft_label:
            corrected[accepted_indices] = accepted_pseudo
        else:
            corrected[accepted_indices] = F.one_hot(
                torch.argmax(accepted_pseudo, dim=1), num_classes=2
            ).float()
        accepted_mask[accepted_indices] = True

    diagnostics.update(
        {
            "accepted_count": int(accepted_indices.numel()),
            "candidate_confidence_mean": float(confidence.mean().item()),
            "candidate_margin_mean": float(margin.mean().item()),
            "candidate_rho_mean": float(structural_rho[candidates].mean().item()),
            "accepted_rho_mean": (
                float(structural_rho[accepted_indices].mean().item())
                if accepted_indices.numel() > 0
                else float("nan")
            ),
        }
    )

    return (
        accepted_indices,
        corrected.cpu().numpy(),
        updated_history,
        True,
        diagnostics,
        scores,
        candidate_mask,
        accepted_mask,
    )


def noise_detection_metrics(refined_indices, injected_noise_mask):
    predicted = np.zeros_like(injected_noise_mask, dtype=bool)
    if refined_indices is not None and refined_indices.numel() > 0:
        predicted[
            refined_indices.detach().cpu().numpy().astype(np.int64)
        ] = True

    truth = injected_noise_mask.astype(bool)
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return {
        "noise_det_precision": float(precision),
        "noise_det_recall": float(recall),
        "noise_det_f1": float(f1),
        "noise_det_tp": tp,
        "noise_det_fp": fp,
        "noise_det_fn": fn,
    }


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, seq_data, graph_data, args, labels_override=None):
    device = torch.device(args.device)
    dataset = SequenceStructureDataset(
        seq_data,
        graph_data,
        labels_override=labels_override,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sequence_structure,
        pin_memory=True,
    )

    model.eval()
    labels, probabilities, ids, sequences = [], [], [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        fused_evidence, _ = model(batch, apply_noise_correction=False)
        prediction = evidence_to_probs(fused_evidence)
        probabilities.extend(prediction[:, 1].detach().cpu().numpy().tolist())
        labels.extend(batch["y"].detach().cpu().numpy().astype(np.int64).tolist())
        ids.extend(batch["id"])
        sequences.extend(batch["seq"])

    return binary_metrics(
        np.asarray(labels),
        np.asarray(probabilities),
        ids=ids,
        seqs=sequences,
    )


# =============================================================================
# Training
# =============================================================================

def train_rise(
    args,
    train_seq,
    train_graph,
    eval_seq,
    eval_graph,
    train_idx,
    val_idx,
    seq_input_dim,
    graph_input_dim,
    clean_labels_override=None,
    file_label_noise_mask_override=None,
):
    device = torch.device(args.device)
    observed_input_labels = train_seq["labels"].numpy().astype(np.int64)
    sample_count = len(observed_input_labels)

    if clean_labels_override is not None:
        clean_labels = np.asarray(clean_labels_override, dtype=np.int64)
        if clean_labels.shape != observed_input_labels.shape:
            raise ValueError(
                "clean_labels_override shape mismatch: "
                f"{clean_labels.shape} vs {observed_input_labels.shape}"
            )
        if not np.isin(clean_labels, [0, 1]).all():
            raise ValueError("clean_labels_override must contain only 0/1 values.")
        clean_reference_matched = np.ones(sample_count, dtype=bool)
        clean_reference_conflicts = 0
    else:
        (
            clean_labels,
            clean_reference_matched,
            clean_reference_conflicts,
        ) = build_clean_labels_from_reference(
            train_seq["seqs"], args, observed_input_labels
        )

    derived_file_noise_mask = observed_input_labels != clean_labels
    if file_label_noise_mask_override is not None:
        file_label_noise_mask = np.asarray(
            file_label_noise_mask_override, dtype=bool
        )
        if file_label_noise_mask.shape != observed_input_labels.shape:
            raise ValueError("file_label_noise_mask_override shape mismatch.")
        if not np.array_equal(file_label_noise_mask, derived_file_noise_mask):
            raise ValueError(
                "CSV is_noisy values disagree with target and clean_target."
            )
    else:
        file_label_noise_mask = derived_file_noise_mask

    # Pre-generated CSV noise is used directly. FASTA inputs retain an optional
    # internal symmetric-noise path for compatibility with local experiments.
    if clean_labels_override is not None:
        resolved_noise_source = "file"
    else:
        resolved_noise_source = resolve_noise_source(args)

    if resolved_noise_source == "file":
        noisy_labels = observed_input_labels.copy().astype(np.int64)
        injected_noise_mask = file_label_noise_mask.copy()
        internal_noise_rate_used = 0.0
    elif resolved_noise_source == "internal":
        noisy_labels, injected_noise_mask = inject_symmetric_noise(
            clean_labels,
            train_idx,
            args.noise_rate,
            args.seed,
        )
        file_label_noise_mask = np.zeros_like(clean_labels, dtype=bool)
        internal_noise_rate_used = float(args.noise_rate)
    else:
        raise ValueError(f"Unsupported noise source: {resolved_noise_source}")

    args._resolved_noise_source = resolved_noise_source
    args._internal_noise_rate_used = internal_noise_rate_used
    args._input_file_noise_count_all = int(file_label_noise_mask.sum())
    args._input_file_noise_count_train = int(file_label_noise_mask[train_idx].sum())
    args._input_file_noise_count_val = int(file_label_noise_mask[val_idx].sum())
    args._input_file_noise_rate_all = float(file_label_noise_mask.mean())
    args._input_file_noise_rate_train = float(file_label_noise_mask[train_idx].mean())
    args._input_file_noise_rate_val = float(file_label_noise_mask[val_idx].mean())
    args._effective_train_noise_rate = float(injected_noise_mask[train_idx].mean())
    args._clean_ref_matched_count = int(clean_reference_matched.sum())
    args._clean_ref_missing_count = int((~clean_reference_matched).sum())
    args._clean_ref_conflict_count = int(clean_reference_conflicts)

    print(
        "Noise setting | "
        f"source={resolved_noise_source} | "
        f"train_noise={int(injected_noise_mask[train_idx].sum())}/{len(train_idx)} "
        f"({args._effective_train_noise_rate:.4f})"
    )

    # Effective supervision is persistent across refinement steps:
    # y^(0)=observed label; failed future proposals retain y^(t-1).
    effective_supervision = (
        F.one_hot(torch.tensor(noisy_labels, dtype=torch.long), num_classes=2)
        .float()
        .numpy()
    )

    knn_features = build_joint_knn_features(train_seq, train_graph, args)
    similarity_matrix, nearest_indices = build_similarity_and_neighbors(
        knn_features,
        train_idx,
        args.knn_k,
        device,
    )

    (
        structural_reliability,
        structural_rho_np,
        reliability_low_value,
        reliability_high_value,
    ) = structural_reliability_from_graphs(
        train_graph["graphs"],
        train_idx,
        low_quantile=args.reliability_low_quantile,
        high_quantile=args.reliability_high_quantile,
    )
    structural_rho = torch.tensor(
        structural_rho_np, dtype=torch.float32, device=device
    )

    model = RISEModel(seq_input_dim, graph_input_dim, args).to(device)

    backbone_parameters = []
    transition_parameters = []
    for name, parameter in model.named_parameters():
        if name in {"T_seq", "T_struct"}:
            transition_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    optimizer_model = torch.optim.AdamW(
        backbone_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer_transition = torch.optim.AdamW(
        transition_parameters,
        lr=args.t_lr,
        weight_decay=args.weight_decay,
    )

    best_score = -float("inf")
    best_state = None
    best_epoch = -1
    bad_epochs = 0

    ever_refined_indices = torch.empty(0, dtype=torch.long, device=device)
    historical_probs = torch.zeros(
        (sample_count, 2), dtype=torch.float32, device=device
    )
    history_initialized = False

    history_rows: List[Dict[str, Any]] = []
    last_refinement: Dict[str, Any] = {}
    last_candidate_scores = torch.full(
        (sample_count,), float("nan"), dtype=torch.float32, device=device
    )
    last_candidate_mask = torch.zeros(
        sample_count, dtype=torch.bool, device=device
    )
    last_accepted_mask = torch.zeros(
        sample_count, dtype=torch.bool, device=device
    )

    for epoch in range(1, args.epochs + 1):
        train_dataset = SequenceStructureDataset(
            train_seq,
            train_graph,
            indices=train_idx.tolist(),
            labels_override=noisy_labels,
            effective_supervision=effective_supervision,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_sequence_structure,
            pin_memory=True,
        )

        validation_dataset = SequenceStructureDataset(
            train_seq,
            train_graph,
            indices=val_idx.tolist(),
            labels_override=clean_labels,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_sequence_structure,
            pin_memory=True,
        )

        model.train()
        sequence_evidence_all = torch.zeros(
            (sample_count, 2), dtype=torch.float32, device=device
        )
        structural_evidence_all = torch.zeros(
            (sample_count, 2), dtype=torch.float32, device=device
        )
        fused_evidence_all = torch.zeros(
            (sample_count, 2), dtype=torch.float32, device=device
        )

        losses = []
        base_losses = []
        auxiliary_losses = []
        transition_losses = []
        confidence_losses = []
        refinement_losses = []
        mixing_losses = []

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            y = batch["y"].long()
            indices = batch["idx"].long()
            effective_y = batch["effective_y"].float()
            y_onehot = to_onehot(y, 2).to(device)

            (
                e_seq,
                e_struct,
                e_fused_clean,
                e_seq_transformed,
                e_struct_transformed,
                e_fused_transformed,
                h_seq,
                h_struct,
            ) = model(
                batch,
                apply_noise_correction=True,
                return_features=True,
            )

            # Base transformed/fused evidential learning.
            loss_base = edl_ce_loss(
                y_onehot,
                e_fused_transformed + 1.0,
                2,
                epoch,
                args.annealing_epoch,
                device,
            ).mean()

            # Branch-level auxiliary evidential learning.
            loss_aux = torch.tensor(0.0, device=device)
            if args.lambda_aux > 0:
                loss_seq_aux = edl_ce_loss(
                    y_onehot,
                    e_seq_transformed + 1.0,
                    2,
                    epoch,
                    args.annealing_epoch,
                    device,
                ).mean()
                loss_struct_aux = edl_ce_loss(
                    y_onehot,
                    e_struct_transformed + 1.0,
                    2,
                    epoch,
                    args.annealing_epoch,
                    device,
                ).mean()
                loss_aux = 0.5 * (loss_seq_aux + loss_struct_aux)

            # View-transition consistency.
            loss_transition = model.transition_consistency_loss()

            # Prediction-confidence regularization.
            alpha_transformed = e_fused_transformed + 1.0
            uncertainty = 2.0 / torch.sum(alpha_transformed, dim=1)
            confidence = 1.0 - uncertainty
            loss_confidence = 0.5 * (
                confidence_transition_loss(
                    confidence, model.T_seq, y, 2, device
                ).mean()
                + confidence_transition_loss(
                    confidence, model.T_struct, y, 2, device
                ).mean()
            )

            # Refined-supervision loss is applied to samples that have ever
            # passed verification, using their current persistent target.
            if ever_refined_indices.numel() > 0:
                refined_mask = torch.isin(indices, ever_refined_indices)
                refined_value = refined_mse_loss(
                    effective_y,
                    e_fused_clean,
                    refined_mask,
                )
                loss_refinement = (
                    refined_value
                    if refined_value is not None
                    else torch.tensor(0.0, device=device)
                )
            else:
                refined_mask = torch.zeros_like(y, dtype=torch.bool, device=device)
                loss_refinement = torch.tensor(0.0, device=device)

            # Representation interpolation objective.
            loss_mix = torch.tensor(0.0, device=device)
            if (
                args.use_mixup
                and bool(refined_mask.any())
                and bool((~refined_mask).any())
            ):
                refined_positions = torch.nonzero(
                    refined_mask, as_tuple=False
                ).view(-1)
                clean_pool = torch.nonzero(~refined_mask, as_tuple=False).view(-1)
                random_choice = torch.randint(
                    0,
                    clean_pool.numel(),
                    (refined_positions.numel(),),
                    device=device,
                )
                clean_positions = clean_pool[random_choice]
                mix_lambda = torch.distributions.Beta(
                    args.mixup_alpha, args.mixup_alpha
                ).sample((refined_positions.numel(),)).to(device)
                mix_lambda = torch.max(mix_lambda, 1.0 - mix_lambda).view(-1, 1)

                h_seq_mix = (
                    mix_lambda * h_seq[refined_positions]
                    + (1.0 - mix_lambda) * h_seq[clean_positions]
                )
                h_struct_mix = (
                    mix_lambda * h_struct[refined_positions]
                    + (1.0 - mix_lambda) * h_struct[clean_positions]
                )
                y_mix = (
                    mix_lambda * effective_y[refined_positions]
                    + (1.0 - mix_lambda) * y_onehot[clean_positions]
                )
                _, _, e_mix = model.evidence_from_features(h_seq_mix, h_struct_mix)
                p_mix = evidence_to_probs(e_mix)
                loss_mix = torch.sum((y_mix - p_mix) ** 2, dim=1).mean()

            loss = (
                loss_base
                + args.lambda_aux * loss_aux
                + args.lambda_t_consistency * loss_transition
                + args.lambda_conf * loss_confidence
                + args.lambda_mse * loss_refinement
                + args.lambda_mix * loss_mix
            )

            optimizer_model.zero_grad()
            optimizer_transition.zero_grad()
            loss.backward()

            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad_norm
                )

            optimizer_model.step()
            if epoch > args.warmup_epochs:
                optimizer_transition.step()
                model.normalize_transition_matrices_()

            sequence_evidence_all[indices] = e_seq.detach()
            structural_evidence_all[indices] = e_struct.detach()
            fused_evidence_all[indices] = e_fused_clean.detach()

            losses.append(float(loss.item()))
            base_losses.append(float(loss_base.item()))
            auxiliary_losses.append(float(loss_aux.item()))
            transition_losses.append(float(loss_transition.item()))
            confidence_losses.append(float(loss_confidence.item()))
            refinement_losses.append(float(loss_refinement.item()))
            mixing_losses.append(float(loss_mix.item()))

        # Validation uses the untransformed fused evidence.
        model.eval()
        validation_probs, validation_labels = [], []
        with torch.no_grad():
            for batch in validation_loader:
                batch = move_batch_to_device(batch, device)
                fused_evidence, _ = model(batch, apply_noise_correction=False)
                prediction = evidence_to_probs(fused_evidence)
                validation_probs.extend(
                    prediction[:, 1].detach().cpu().numpy().tolist()
                )
                validation_labels.extend(
                    batch["y"].detach().cpu().numpy().astype(np.int64).tolist()
                )

        validation_metrics, _ = binary_metrics(
            np.asarray(validation_labels),
            np.asarray(validation_probs),
        )
        if args.monitor == "val_auc":
            score = validation_metrics["AUC"]
        elif args.monitor == "val_f1":
            score = validation_metrics["F1"]
        else:
            score = validation_metrics["Accuracy"]

        min_best_epoch = int(args.min_best_epoch)
        if args.best_after_start_correct:
            min_best_epoch = max(min_best_epoch, int(args.start_correct) + 1)
        can_save_best = epoch >= min_best_epoch

        if can_save_best and score > best_score + args.min_delta:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        elif can_save_best:
            bad_epochs += 1

        # Periodic candidate identification and verified supervision refinement.
        if (
            epoch >= args.start_correct
            and (epoch - args.start_correct) % args.correct_every == 0
        ):
            (
                accepted_indices,
                effective_supervision,
                historical_probs,
                history_initialized,
                last_refinement,
                last_candidate_scores,
                last_candidate_mask,
                last_accepted_mask,
            ) = refine_supervision(
                threshold=args.threshold,
                fused_evidence=fused_evidence_all.detach(),
                sequence_evidence=sequence_evidence_all.detach(),
                structural_evidence=structural_evidence_all.detach(),
                nearest_indices=nearest_indices,
                similarity_matrix=similarity_matrix,
                train_idx=train_idx,
                observed_labels=noisy_labels,
                previous_effective_supervision=effective_supervision,
                structural_rho=structural_rho,
                historical_probs=historical_probs,
                history_initialized=history_initialized,
                history_beta=args.history_beta,
                weight_sequence=args.weight_sequence,
                weight_neighbor=args.weight_neighbor,
                weight_history=args.weight_history,
                weight_structure=args.weight_structure,
                pseudo_confidence_threshold=args.pseudo_confidence_threshold,
                margin_threshold=args.margin_threshold,
                device=device,
                keep_soft_label=args.keep_soft_label,
            )

            if accepted_indices.numel() > 0:
                ever_refined_indices = torch.unique(
                    torch.cat([ever_refined_indices, accepted_indices])
                )

            if args.threshold_decay != 1.0:
                args.threshold = max(
                    args.min_threshold,
                    args.threshold * args.threshold_decay,
                )

        history_row = {
            "epoch": int(epoch),
            "loss": float(np.mean(losses)),
            "loss_base": float(np.mean(base_losses)),
            "loss_aux": float(np.mean(auxiliary_losses)),
            "loss_transition": float(np.mean(transition_losses)),
            "loss_confidence": float(np.mean(confidence_losses)),
            "loss_refinement": float(np.mean(refinement_losses)),
            "loss_mix": float(np.mean(mixing_losses)),
            "val_acc": float(validation_metrics["Accuracy"]),
            "val_f1": float(validation_metrics["F1"]),
            "val_auc": float(validation_metrics["AUC"]),
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "refined_total": int(ever_refined_indices.numel()),
        }
        history_rows.append(history_row)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == best_epoch:
            print(
                f"[RISE {epoch:03d}/{args.epochs}] "
                f"loss={history_row['loss']:.5f} "
                f"base={history_row['loss_base']:.5f} "
                f"aux={history_row['loss_aux']:.5f} "
                f"T={history_row['loss_transition']:.5f} "
                f"conf={history_row['loss_confidence']:.5f} "
                f"ref={history_row['loss_refinement']:.5f} "
                f"mix={history_row['loss_mix']:.5f} | "
                f"val_acc={history_row['val_acc']:.4f} "
                f"val_f1={history_row['val_f1']:.4f} "
                f"val_auc={history_row['val_auc']:.4f} | "
                f"refined={history_row['refined_total']} | "
                f"best={best_score:.4f}@{best_epoch}"
            )

        if bad_epochs >= args.patience:
            print(
                f"Early stopping at epoch={epoch}; best_epoch={best_epoch}."
            )
            break

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_epoch = int(history_rows[-1]["epoch"])
        if args.monitor == "val_auc":
            best_score = float(history_rows[-1]["val_auc"])
        elif args.monitor == "val_f1":
            best_score = float(history_rows[-1]["val_f1"])
        else:
            best_score = float(history_rows[-1]["val_acc"])

    model.load_state_dict(best_state)
    eval_metrics, prediction_rows = evaluate(
        model,
        eval_seq,
        eval_graph,
        args,
    )

    eval_metrics.update(
        {
            "best_epoch": int(best_epoch),
            "best_val_score": float(best_score),
            "noise_rate": float(args.noise_rate),
            "noise_source": args._resolved_noise_source,
            "effective_train_noise_rate": float(args._effective_train_noise_rate),
            "warmup_epochs": int(args.warmup_epochs),
            "start_correct": int(args.start_correct),
            "correct_every": int(args.correct_every),
            "knn_k": int(args.knn_k),
            "fusion_type": args.fusion_type,
            "lambda_aux": float(args.lambda_aux),
            "lambda_transition": float(args.lambda_t_consistency),
            "lambda_confidence": float(args.lambda_conf),
            "lambda_refinement": float(args.lambda_mse),
            "lambda_mix": float(args.lambda_mix),
            "candidate_threshold_final": float(args.threshold),
            "refined_total": int(ever_refined_indices.numel()),
            "reliability_low_value": float(reliability_low_value),
            "reliability_high_value": float(reliability_high_value),
            "last_refinement": last_refinement,
            **noise_detection_metrics(ever_refined_indices, injected_noise_mask),
        }
    )

    audit = {
        "structural_reliability": structural_reliability,
        "structural_rho": structural_rho_np,
        "last_candidate_scores": last_candidate_scores.detach().cpu().numpy(),
        "last_candidate_mask": last_candidate_mask.detach().cpu().numpy(),
        "last_accepted_mask": last_accepted_mask.detach().cpu().numpy(),
        "ever_refined_mask": np.isin(
            np.arange(sample_count),
            ever_refined_indices.detach().cpu().numpy().astype(np.int64),
        ),
    }

    return (
        model,
        eval_metrics,
        prediction_rows,
        history_rows,
        effective_supervision,
        noisy_labels,
        injected_noise_mask,
        observed_input_labels,
        file_label_noise_mask,
        clean_labels,
        audit,
    )


# =============================================================================
# Output helpers
# =============================================================================

def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return str(value)


def save_run_config(
    args,
    output_path,
    sequence_input_dim,
    graph_input_dim,
    train_size,
    validation_size,
):
    public_args = {
        key: _json_ready(value)
        for key, value in vars(args).items()
        if not key.startswith("_")
    }

    config = {
        "format_version": 3,
        "model": "RISE",
        "algorithm": "RISE",
        "sequence_input_dim": int(sequence_input_dim),
        "graph_input_dim": int(graph_input_dim),
        "train_size": int(train_size),
        "validation_size": int(validation_size),
        "arguments": public_args,
        "refinement_configuration": {
            "candidate_threshold": float(args.initial_threshold),
            "reliability_weights": [0.50, 0.30, 0.20],
            "reliability_quantile": 0.20,
            "low_residue_confidence_threshold": 0.50,
            "reliability_low_quantile": float(args.reliability_low_quantile),
            "reliability_high_quantile": float(args.reliability_high_quantile),
            "history_beta": float(args.history_beta),
            "source_weights": {
                "sequence": float(args.weight_sequence),
                "neighborhood": float(args.weight_neighbor),
                "history": float(args.weight_history),
                "structure": float(args.weight_structure),
            },
            "pseudo_confidence_threshold": float(
                args.pseudo_confidence_threshold
            ),
            "margin_threshold": float(args.margin_threshold),
        },
    }

    Path(output_path).write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_outputs(
    *,
    args,
    model,
    metrics,
    prediction_rows,
    history,
    effective_supervision,
    noisy_labels,
    injected_noise_mask,
    sequence_data,
    graph_data,
    train_idx,
    val_idx,
    audit,
    observed_input_labels,
    file_label_noise_mask,
    clean_labels,
):
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = result_dir / "val_metrics.json"
    predictions_path = result_dir / "val_predictions.csv"
    checkpoint_path = result_dir / "best_model.pt"
    history_path = result_dir / "train_history.csv"
    info_path = result_dir / "train_info.csv"

    metrics_path.write_text(
        json.dumps(_json_ready(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "seq", "true_label", "prob_amp", "pred_label"],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(history[0].keys()) if history else ["epoch"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    train_set = set(int(i) for i in np.asarray(train_idx).tolist())
    val_set = set(int(i) for i in np.asarray(val_idx).tolist())
    info_rows = []

    for index in range(len(clean_labels)):
        if index in train_set:
            split = "train"
        elif index in val_set:
            split = "val"
        else:
            split = "unknown"

        graph = graph_data["graphs"][index]
        candidate_score = audit["last_candidate_scores"][index]
        info_rows.append(
            {
                "id": sequence_data["ids"][index],
                "seq": sequence_data["seqs"][index],
                "split": split,
                "observed_input_label": int(observed_input_labels[index]),
                "clean_label": int(clean_labels[index]),
                "file_label_is_noisy": int(bool(file_label_noise_mask[index])),
                "training_label": int(noisy_labels[index]),
                "injected_noise": int(bool(injected_noise_mask[index])),
                "effective_nonamp": float(effective_supervision[index, 0]),
                "effective_amp": float(effective_supervision[index, 1]),
                "structural_reliability": float(
                    audit["structural_reliability"][index]
                ),
                "structural_rho": float(audit["structural_rho"][index]),
                "candidate_score_last": (
                    "" if not np.isfinite(candidate_score) else float(candidate_score)
                ),
                "candidate_last": int(bool(audit["last_candidate_mask"][index])),
                "accepted_last": int(bool(audit["last_accepted_mask"][index])),
                "ever_refined": int(bool(audit["ever_refined_mask"][index])),
                "pdb_path": graph.get("pdb_path", ""),
                "plddt_mean": float(graph.get("plddt_mean", 0.0)),
                "plddt_min": float(graph.get("plddt_min", 0.0)),
                "length_struct": int(graph.get("length_struct", 0)),
            }
        )

    with info_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(info_rows[0].keys()) if info_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(info_rows)

    torch.save(model.state_dict(), checkpoint_path)

    print("\nValidation metrics")
    print(json.dumps(_json_ready(metrics), ensure_ascii=False, indent=2))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved training audit: {info_path}")


# =============================================================================
# Configuration
# =============================================================================

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML configuration not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {yaml_path}")
    return data


def _nested_get(mapping: Mapping[str, Any], path: Sequence[str], default=None):
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _dataset_key(name: str) -> str:
    normalized = str(name).strip().lower().replace("_", "-")
    if "xuamp" in normalized:
        return "xuamp"
    if "genpept" in normalized:
        return "genpept"
    raise ValueError(
        f"Unsupported dataset {name!r}; expected XUAMP or GenPept-Curated-2025."
    )


def _format_noise(noise: float) -> str:
    return f"{float(noise):.3f}"


def _expand_template(value, context):
    if value is None:
        return None
    text = str(value)
    try:
        return text.format(**context)
    except KeyError as exc:
        raise ValueError(
            f"Unknown placeholder {exc} in path template: {text}"
        ) from exc


def resolve_experiment_paths(
    experiment_config,
    paths_config,
    dataset_override,
    noise,
    rep,
):
    dataset_name = dataset_override or _nested_get(
        experiment_config, ("experiment", "dataset")
    )
    if not dataset_name:
        raise ValueError("Dataset is missing from CLI and experiment config.")

    key = _dataset_key(dataset_name)
    dataset_paths = _nested_get(paths_config, ("datasets", key))
    if not isinstance(dataset_paths, Mapping):
        raise ValueError(f"Missing datasets.{key} in paths YAML.")

    root = str(dataset_paths.get("root", ""))
    noise_text = _format_noise(noise)
    context = {"root": root, "noise": noise_text, "rep": int(rep)}
    run_dir = _expand_template(dataset_paths.get("run_dir_template"), context)
    if not run_dir:
        raise ValueError(f"datasets.{key}.run_dir_template is required.")
    context["run_dir"] = run_dir

    resolved = {
        "dataset_key": key,
        "dataset_name": str(dataset_name),
        "noise_text": noise_text,
        "run_dir": run_dir,
    }

    for field in (
        "train_csv",
        "val_csv",
        "test_csv",
        "train_amp",
        "train_nonamp",
        "val_amp",
        "val_nonamp",
        "test_amp",
        "test_nonamp",
    ):
        value = _expand_template(dataset_paths.get(f"{field}_template"), context)
        if value:
            resolved[field] = value

    for field in ("pdb_dir", "clean_ref_amp", "clean_ref_nonamp"):
        value = _expand_template(dataset_paths.get(field), context)
        if value:
            resolved[field] = value

    runtime = paths_config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime in paths YAML must be a mapping.")
    for field in ("device", "esm_model_path", "output_root", "cache_root"):
        if runtime.get(field) is not None:
            resolved[field] = str(runtime[field])

    output_root = Path(resolved.get("output_root", "./outputs")).expanduser()
    cache_root = Path(resolved.get("cache_root", "./cache")).expanduser()
    resolved["result_dir"] = str(
        output_root / key / f"noise_{noise_text}" / f"rep{int(rep)}"
    )
    resolved["cache_dir"] = str(
        cache_root / key / f"noise_{noise_text}" / f"rep{int(rep)}"
    )
    return resolved


_CONFIG_ARGUMENT_MAP = {
    "noise_source": ("data", "noise_source"),
    "esm_batch_size": ("features", "esm_batch_size"),
    "contact_threshold": ("features", "contact_threshold"),
    "graph_knn_k": ("features", "graph_knn_k"),
    "plddt_scale": ("features", "plddt_scale"),
    "index_pdb_by_seq": ("features", "index_pdb_by_seq"),
    "allow_missing_pdb": ("features", "allow_missing_pdb"),
    "use_aa_onehot": ("features", "use_aa_onehot"),
    "disable_plddt_feature": ("features", "disable_plddt_feature"),
    "delete_cache_after_run": ("cache", "delete_after_run"),
    "hidden_dim": ("model", "hidden_dim"),
    "lstm_layers": ("model", "lstm_layers"),
    "gnn_hidden_dim": ("model", "gnn_hidden_dim"),
    "gnn_layers": ("model", "gnn_layers"),
    "classifier_hidden": ("model", "classifier_hidden"),
    "dropout": ("model", "dropout"),
    "fusion_type": ("model", "fusion_type"),
    "knn_feature": ("neighborhood", "feature"),
    "knn_k": ("neighborhood", "knn_k"),
    "lambda_aux": ("loss", "lambda_aux"),
    "lambda_t_consistency": ("loss", "lambda_t_consistency"),
    "lambda_conf": ("loss", "lambda_conf"),
    "lambda_mse": ("loss", "lambda_mse"),
    "lambda_mix": ("loss", "lambda_mix"),
    "batch_size": ("training", "batch_size"),
    "eval_batch_size": ("training", "eval_batch_size"),
    "num_workers": ("training", "num_workers"),
    "epochs": ("training", "epochs"),
    "start_correct": ("training", "start_correct"),
    "correct_every": ("training", "correct_every"),
    "patience": ("training", "patience"),
    "monitor": ("training", "monitor"),
    "min_delta": ("training", "min_delta"),
    "min_best_epoch": ("training", "min_best_epoch"),
    "best_after_start_correct": ("training", "best_after_start_correct"),
    "lr": ("training", "lr"),
    "t_lr": ("training", "transition_lr"),
    "weight_decay": ("training", "weight_decay"),
    "clip_grad_norm": ("training", "clip_grad_norm"),
    "annealing_epoch": ("training", "annealing_epoch"),
    "seed": ("training", "seed"),
    "log_every": ("training", "log_every"),
    "log_every_extract": ("training", "log_every_extract"),
    "threshold": ("calibration", "threshold"),
    "threshold_decay": ("calibration", "threshold_decay"),
    "min_threshold": ("calibration", "min_threshold"),
    "keep_soft_label": ("calibration", "keep_soft_label"),
    "use_mixup": ("calibration", "use_mixup"),
    "mixup_alpha": ("calibration", "mixup_alpha"),
}


def build_parser():
    parser = argparse.ArgumentParser(description="Train RISE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--noise", required=True, type=float)
    parser.add_argument("--rep", required=True, type=int)

    for name in (
        "train_csv",
        "val_csv",
        "train_amp",
        "train_nonamp",
        "val_amp",
        "val_nonamp",
        "result_dir",
        "cache_dir",
        "device",
        "esm_model_path",
        "pdb_dir",
        "pdb_map_csv",
        "clean_ref_amp",
        "clean_ref_nonamp",
    ):
        parser.add_argument(f"--{name}", default=None)

    for name in (
        "esm_batch_size",
        "graph_knn_k",
        "batch_size",
        "eval_batch_size",
        "num_workers",
        "hidden_dim",
        "lstm_layers",
        "gnn_hidden_dim",
        "gnn_layers",
        "classifier_hidden",
        "epochs",
        "start_correct",
        "correct_every",
        "patience",
        "min_best_epoch",
        "annealing_epoch",
        "knn_k",
        "seed",
        "log_every",
        "log_every_extract",
    ):
        parser.add_argument(f"--{name}", type=int, default=None)

    for name in (
        "contact_threshold",
        "dropout",
        "lambda_aux",
        "lambda_t_consistency",
        "lambda_conf",
        "lambda_mse",
        "lambda_mix",
        "lr",
        "t_lr",
        "weight_decay",
        "clip_grad_norm",
        "min_delta",
        "threshold",
        "threshold_decay",
        "min_threshold",
        "mixup_alpha",
    ):
        parser.add_argument(f"--{name}", type=float, default=None)

    parser.add_argument("--plddt_scale", choices=["auto", "100", "1"], default=None)
    parser.add_argument("--fusion_type", choices=["ds", "sum", "mean"], default=None)
    parser.add_argument("--knn_feature", choices=["dual", "seq", "struct"], default=None)
    parser.add_argument(
        "--monitor", choices=["val_auc", "val_f1", "val_acc"], default=None
    )
    parser.add_argument(
        "--noise_source", choices=["auto", "file", "internal"], default=None
    )

    for name in (
        "index_pdb_by_seq",
        "allow_missing_pdb",
        "disable_plddt_feature",
        "use_aa_onehot",
        "delete_cache_after_run",
        "best_after_start_correct",
        "keep_soft_label",
        "use_mixup",
    ):
        parser.add_argument(
            f"--{name}", action=argparse.BooleanOptionalAction, default=None
        )
    return parser


def _apply_config(cli, experiment_config, resolved_paths):
    values = vars(cli).copy()
    for attribute, config_path in _CONFIG_ARGUMENT_MAP.items():
        if values.get(attribute) is None:
            configured = _nested_get(experiment_config, config_path)
            if configured is not None:
                values[attribute] = configured

    for attribute in (
        "device",
        "esm_model_path",
        "pdb_dir",
        "train_csv",
        "val_csv",
        "train_amp",
        "train_nonamp",
        "val_amp",
        "val_nonamp",
        "clean_ref_amp",
        "clean_ref_nonamp",
        "result_dir",
        "cache_dir",
    ):
        if values.get(attribute) is None and resolved_paths.get(attribute) is not None:
            values[attribute] = resolved_paths[attribute]

    values["dataset"] = resolved_paths["dataset_key"]
    values["dataset_name"] = resolved_paths["dataset_name"]
    values["noise_rate"] = float(cli.noise)
    values["noise"] = float(cli.noise)
    values["rep"] = int(cli.rep)
    values["validation_mode"] = "predefined"
    return argparse.Namespace(**values)


def resolve_training_args(cli):
    experiment_config = _load_yaml(cli.config)
    paths_config = _load_yaml(cli.paths)
    resolved_paths = resolve_experiment_paths(
        experiment_config,
        paths_config,
        cli.dataset,
        cli.noise,
        cli.rep,
    )
    args = _apply_config(cli, experiment_config, resolved_paths)

    defaults = {
        "device": "cuda:0",
        "esm_batch_size": 8,
        "pdb_map_csv": None,
        "index_pdb_by_seq": False,
        "allow_missing_pdb": False,
        "contact_threshold": 8.0,
        "graph_knn_k": 8,
        "plddt_scale": "auto",
        "disable_plddt_feature": False,
        "use_aa_onehot": True,
        "delete_cache_after_run": False,
        "batch_size": 16,
        "eval_batch_size": 64,
        "num_workers": 0,
        "hidden_dim": 128,
        "lstm_layers": 1,
        "gnn_hidden_dim": 128,
        "gnn_layers": 3,
        "classifier_hidden": 128,
        "dropout": 0.30,
        "fusion_type": "ds",
        "knn_feature": "dual",
        "lambda_aux": 0.05,
        "lambda_t_consistency": 0.05,
        "epochs": 100,
        "start_correct": 5,
        "correct_every": 5,
        "patience": 40,
        "monitor": "val_f1",
        "min_delta": 1e-4,
        "min_best_epoch": 1,
        "best_after_start_correct": True,
        "lr": 3e-4,
        "t_lr": 1e-3,
        "weight_decay": 1e-4,
        "clip_grad_norm": 1.0,
        "annealing_epoch": 80,
        "knn_k": 20,
        "lambda_conf": 0.01,
        "lambda_mse": 1.0,
        "lambda_mix": 1.0,
        "threshold": 0.80,
        "threshold_decay": 1.0,
        "min_threshold": 0.60,
        "keep_soft_label": True,
        "use_mixup": True,
        "mixup_alpha": 0.30,
        "noise_source": "file",
        "seed": 42,
        "log_every": 5,
        "log_every_extract": 500,
    }
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    # Paper-defined schedule and refinement parameters.
    args.warmup_epochs = 5
    args.reliability_low_quantile = 0.15
    args.reliability_high_quantile = 0.70
    args.history_beta = 0.70
    args.weight_sequence = 0.35
    args.weight_neighbor = 0.35
    args.weight_history = 0.15
    args.weight_structure = 0.15
    args.pseudo_confidence_threshold = 0.55
    args.margin_threshold = 0.10

    source_weight_sum = (
        args.weight_sequence
        + args.weight_neighbor
        + args.weight_history
        + args.weight_structure
    )
    if source_weight_sum <= 0:
        raise ValueError("Pseudo-label source weights must sum to a positive value.")

    args.config = str(Path(cli.config).expanduser().resolve())
    args.paths = str(Path(cli.paths).expanduser().resolve())
    _require_paths(args)
    return args


def _require_paths(args):
    csv_fields = ("train_csv", "val_csv")
    fasta_fields = ("train_amp", "train_nonamp", "val_amp", "val_nonamp")
    has_all_csv = all(getattr(args, field, None) for field in csv_fields)
    has_any_csv = any(getattr(args, field, None) for field in csv_fields)
    has_all_fasta = all(getattr(args, field, None) for field in fasta_fields)
    has_any_fasta = any(getattr(args, field, None) for field in fasta_fields)

    if has_any_csv and not has_all_csv:
        raise ValueError("CSV mode requires train_csv and val_csv.")
    if not has_all_csv and has_any_fasta and not has_all_fasta:
        raise ValueError(
            "FASTA mode requires train_amp, train_nonamp, val_amp and val_nonamp."
        )

    if has_all_csv:
        args.input_format = "csv"
        required_data = csv_fields
    elif has_all_fasta:
        args.input_format = "fasta"
        required_data = fasta_fields
    else:
        raise ValueError("Configure either CSV or FASTA train/validation inputs.")

    required = list(required_data) + [
        "esm_model_path",
        "pdb_dir",
        "result_dir",
        "cache_dir",
    ]
    missing = [field for field in required if not getattr(args, field, None)]
    if missing:
        raise ValueError("Missing required path(s): " + ", ".join(missing))

    for field in required_data:
        path = Path(getattr(args, field)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{field} not found: {path}")
        setattr(args, field, str(path.resolve()))

    esm_path = Path(args.esm_model_path).expanduser()
    if not esm_path.exists():
        raise FileNotFoundError(f"esm_model_path not found: {esm_path}")
    args.esm_model_path = str(esm_path.resolve())

    pdb_path = Path(args.pdb_dir).expanduser()
    if not pdb_path.is_dir():
        raise FileNotFoundError(f"pdb_dir not found: {pdb_path}")
    args.pdb_dir = str(pdb_path.resolve())

    if args.pdb_map_csv:
        mapping_path = Path(args.pdb_map_csv).expanduser()
        if not mapping_path.is_file():
            raise FileNotFoundError(f"pdb_map_csv not found: {mapping_path}")
        args.pdb_map_csv = str(mapping_path.resolve())

    for field in ("clean_ref_amp", "clean_ref_nonamp"):
        value = getattr(args, field, None)
        if value:
            path = Path(value).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"{field} not found: {path}")
            setattr(args, field, str(path.resolve()))


def combine_feature_data(
    train_sequence,
    train_graph,
    validation_sequence,
    validation_graph,
):
    train_count = len(train_sequence["labels"])
    validation_count = len(validation_sequence["labels"])

    sequence_data = {
        "ids": list(train_sequence["ids"]) + list(validation_sequence["ids"]),
        "seqs": list(train_sequence["seqs"]) + list(validation_sequence["seqs"]),
        "labels": torch.cat(
            [train_sequence["labels"].long(), validation_sequence["labels"].long()],
            dim=0,
        ),
        "embeddings": list(train_sequence["embeddings"])
        + list(validation_sequence["embeddings"]),
    }
    graph_data = {
        "ids": list(train_graph["ids"]) + list(validation_graph["ids"]),
        "seqs": list(train_graph["seqs"]) + list(validation_graph["seqs"]),
        "labels": torch.cat(
            [train_graph["labels"].long(), validation_graph["labels"].long()],
            dim=0,
        ),
        "graphs": list(train_graph["graphs"]) + list(validation_graph["graphs"]),
    }
    train_idx = np.arange(train_count, dtype=np.int64)
    val_idx = np.arange(
        train_count,
        train_count + validation_count,
        dtype=np.int64,
    )
    return sequence_data, graph_data, train_idx, val_idx


def assert_alignment(sequence_data, graph_data, name):
    sequence_count = len(sequence_data["labels"])
    graph_count = len(graph_data["labels"])
    if sequence_count != graph_count:
        raise RuntimeError(
            f"{name}: sequence/structure length mismatch: "
            f"{sequence_count} vs {graph_count}"
        )

    for index in range(sequence_count):
        if norm_seq(sequence_data["seqs"][index]) != norm_seq(
            graph_data["seqs"][index]
        ):
            raise RuntimeError(f"{name}: sequence mismatch at index={index}")
        if int(sequence_data["labels"][index]) != int(graph_data["labels"][index]):
            raise RuntimeError(f"{name}: label mismatch at index={index}")

    print(
        f"{name} sequence/structure alignment checked: {sequence_count} samples"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    cli = build_parser().parse_args()
    args = resolve_training_args(cli)

    result_dir = Path(args.result_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir = str(result_dir)
    args.cache_dir = str(cache_dir)
    args.initial_threshold = float(args.threshold)

    set_seed(args.seed)

    clean_labels_override = None
    file_label_noise_mask_override = None

    if args.input_format == "csv":
        train_records = load_csv_split(
            args.train_csv,
            label_column="target",
            expected_split="train",
        )
        validation_records = load_csv_split(
            args.val_csv,
            label_column="clean_target",
            expected_split="val",
        )
        clean_labels_override = np.asarray(
            [record["clean_target"] for record in train_records]
            + [record["clean_target"] for record in validation_records],
            dtype=np.int64,
        )
        file_label_noise_mask_override = np.asarray(
            [record["is_noisy"] for record in train_records]
            + [record["is_noisy"] for record in validation_records],
            dtype=bool,
        )
    else:
        train_records = load_binary_split(args.train_amp, args.train_nonamp)
        validation_records = load_binary_split(args.val_amp, args.val_nonamp)

    print(
        f"Predefined {args.input_format} split | "
        f"train={len(train_records)} | validation={len(validation_records)}"
    )

    train_sequence = extract_token_embeddings(
        records=train_records,
        esm_model_path=args.esm_model_path,
        cache_fp=str(cache_dir / "train_esm2_token.pt"),
        device=args.device,
        esm_batch_size=args.esm_batch_size,
    )
    validation_sequence = extract_token_embeddings(
        records=validation_records,
        esm_model_path=args.esm_model_path,
        cache_fp=str(cache_dir / "val_esm2_token.pt"),
        device=args.device,
        esm_batch_size=args.esm_batch_size,
    )
    train_graph = extract_graphs(
        train_records,
        str(cache_dir / "train_struct_graph.pt"),
        args,
        "train",
    )
    validation_graph = extract_graphs(
        validation_records,
        str(cache_dir / "val_struct_graph.pt"),
        args,
        "validation",
    )

    assert_alignment(train_sequence, train_graph, "train")
    assert_alignment(validation_sequence, validation_graph, "validation")

    sequence_data, graph_data, train_idx, val_idx = combine_feature_data(
        train_sequence,
        train_graph,
        validation_sequence,
        validation_graph,
    )
    assert_alignment(sequence_data, graph_data, "combined train/validation")

    sequence_input_dim = int(sequence_data["embeddings"][0].shape[1])
    graph_input_dim = int(graph_data["graphs"][0]["x"].shape[1])

    (
        model,
        validation_metrics,
        validation_prediction_rows,
        history,
        effective_supervision,
        noisy_labels,
        injected_noise_mask,
        observed_input_labels,
        file_label_noise_mask,
        clean_labels,
        audit,
    ) = train_rise(
        args=args,
        train_seq=sequence_data,
        train_graph=graph_data,
        eval_seq=validation_sequence,
        eval_graph=validation_graph,
        train_idx=train_idx,
        val_idx=val_idx,
        seq_input_dim=sequence_input_dim,
        graph_input_dim=graph_input_dim,
        clean_labels_override=clean_labels_override,
        file_label_noise_mask_override=file_label_noise_mask_override,
    )

    validation_metrics.update(
        {
            "evaluation_split": "validation",
            "validation_mode": "predefined",
            "model": "RISE",
            "algorithm": "RISE",
            "dataset": args.dataset_name,
            "noise": float(args.noise),
            "rep": int(args.rep),
            "candidate_threshold_initial": float(args.initial_threshold),
        }
    )

    save_outputs(
        args=args,
        model=model,
        metrics=validation_metrics,
        prediction_rows=validation_prediction_rows,
        history=history,
        effective_supervision=effective_supervision,
        noisy_labels=noisy_labels,
        injected_noise_mask=injected_noise_mask,
        sequence_data=sequence_data,
        graph_data=graph_data,
        train_idx=train_idx,
        val_idx=val_idx,
        audit=audit,
        observed_input_labels=observed_input_labels,
        file_label_noise_mask=file_label_noise_mask,
        clean_labels=clean_labels,
    )

    np.savez_compressed(
        result_dir / "split_indices.npz",
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
    )
    save_run_config(
        args=args,
        output_path=result_dir / "run_config.json",
        sequence_input_dim=sequence_input_dim,
        graph_input_dim=graph_input_dim,
        train_size=len(train_idx),
        validation_size=len(val_idx),
    )

    print("\nRISE training completed.")
    print(f"run_config.json: {result_dir / 'run_config.json'}")

    if args.delete_cache_after_run and cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Deleted cache: {cache_dir}")


if __name__ == "__main__":
    main()
