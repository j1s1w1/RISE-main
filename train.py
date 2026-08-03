#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train RISE with EC-RML-Safe label refinement.

This is the single training implementation for the public repository.  It
contains the dual-view sequence/structure model, evidential fusion, SharedT
training backbone, and EC-RML-Safe reliability-aware label refinement.

Feature preparation remains separated into:
  - data/fasta.py
  - data/csv_dataset.py
  - features/sequence_features.py
  - features/structure_features.py

No external SharedT or Safe implementation file is required at runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from data.fasta import (
    build_clean_labels_from_reference,
    load_binary_split,
    norm_seq,
    resolve_noise_source,
)
from data.csv_dataset import load_csv_split
from features.sequence_features import extract_token_embeddings
from features.structure_features import extract_graphs, get_mean_graph_features

# =============================================================================
# Reproducibility and metrics
# =============================================================================

def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_auc(y_true, prob):
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
        rows.append({
            "id": ids[i] if ids is not None else str(i),
            "seq": seqs[i] if seqs is not None else "",
            "true_label": int(y_true[i]),
            "prob_amp": float(prob_amp[i]),
            "pred_label": int(pred[i]),
        })
    return metrics, rows

def to_onehot(labels: torch.Tensor, num_classes=2):
    return F.one_hot(labels.long().view(-1), num_classes=num_classes).float()

def inject_symmetric_noise(labels: np.ndarray, train_idx: np.ndarray, noise_rate: float, seed: int):
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
# Cross-view neighborhood representation
# =============================================================================

def get_mean_seq_embeddings(seq_data):
    return np.stack([x.mean(dim=0).numpy() for x in seq_data["embeddings"]], axis=0).astype(np.float32)

def build_dual_knn_features(seq_data, graph_data, args):
    seq_mean = get_mean_seq_embeddings(seq_data)
    graph_mean = get_mean_graph_features(graph_data)

    # 避免 ESM2 1280 维完全压过结构均值特征，先做 L2 normalize 后拼接
    seq_mean = seq_mean / (np.linalg.norm(seq_mean, axis=1, keepdims=True) + 1e-8)
    graph_mean = graph_mean / (np.linalg.norm(graph_mean, axis=1, keepdims=True) + 1e-8)

    if args.knn_feature == "seq":
        return seq_mean.astype(np.float32)
    if args.knn_feature == "struct":
        return graph_mean.astype(np.float32)
    return np.concatenate([seq_mean, graph_mean], axis=1).astype(np.float32)

# =============================================================================
# Dataset and batching
# =============================================================================

class DualModalDataset(Dataset):
    def __init__(self, seq_data, graph_data, indices=None, labels_override=None, corrected_probs=None):
        assert len(seq_data["labels"]) == len(graph_data["labels"]), "seq/struct sample num mismatch"
        self.seq_data = seq_data
        self.graph_data = graph_data
        self.indices = list(range(len(seq_data["labels"]))) if indices is None else list(indices)
        self.labels_override = labels_override
        self.corrected_probs = corrected_probs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real = int(self.indices[idx])

        y = int(self.seq_data["labels"][real])
        if self.labels_override is not None:
            y = int(self.labels_override[real])

        if self.corrected_probs is not None:
            cy = self.corrected_probs[real].astype(np.float32)
        else:
            cy = np.zeros(2, dtype=np.float32)
            cy[y] = 1.0

        return {
            "id": self.seq_data["ids"][real],
            "seq": self.seq_data["seqs"][real],
            "seq_x": self.seq_data["embeddings"][real],
            "graph": self.graph_data["graphs"][real],
            "y": y,
            "idx": real,
            "corrected_y": cy,
        }

def collate_dual(batch):
    # sequence padding
    seq_lens = [int(item["seq_x"].shape[0]) for item in batch]
    max_len = max(seq_lens)
    dim = int(batch[0]["seq_x"].shape[1])
    bsz = len(batch)

    seq_x = torch.zeros((bsz, max_len, dim), dtype=torch.float32)
    seq_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    for i, item in enumerate(batch):
        x = item["seq_x"].float()
        l = x.size(0)
        seq_x[i, :l] = x
        seq_mask[i, :l] = True

    # graph batching
    xs, eis, bvec = [], [], []
    offset = 0
    for bi, item in enumerate(batch):
        g = item["graph"]
        gx = g["x"].float()
        ei = g["edge_index"].long()
        n = gx.size(0)

        xs.append(gx)
        if ei.numel() > 0:
            eis.append(ei + offset)
        bvec.append(torch.full((n,), bi, dtype=torch.long))
        offset += n

    graph_x = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long)
    graph_batch = torch.cat(bvec, dim=0)

    return {
        "seq_x": seq_x,
        "seq_mask": seq_mask,
        "seq_lengths": torch.tensor(seq_lens, dtype=torch.long),
        "graph_x": graph_x,
        "edge_index": edge_index,
        "graph_batch": graph_batch,
        "y": torch.tensor([item["y"] for item in batch], dtype=torch.long),
        "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
        "corrected_y": torch.tensor(np.stack([item["corrected_y"] for item in batch], axis=0), dtype=torch.float32),
        "id": [item["id"] for item in batch],
        "seq": [item["seq"] for item in batch],
    }

# =============================================================================
# Evidential losses
# =============================================================================

def dirichlet_kl(alpha, num_classes, device):
    beta = torch.ones((1, num_classes), device=device)
    s_alpha = torch.sum(alpha, dim=1, keepdim=True)
    s_beta = torch.sum(beta, dim=1, keepdim=True)
    ln_b = torch.lgamma(s_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    ln_b_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(s_beta)
    dg0 = torch.digamma(s_alpha)
    dg1 = torch.digamma(alpha)
    return torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + ln_b + ln_b_uni

def edl_ce_loss(y_onehot, alpha, num_classes, step, annealing_step, device):
    s = torch.sum(alpha, dim=1, keepdim=True)
    e = alpha - 1.0
    ce = torch.sum(y_onehot * (torch.digamma(s) - torch.digamma(alpha)), dim=1, keepdim=True)
    alp = e * (1.0 - y_onehot) + 1.0
    anneal = min(1.0, float(step) / float(max(1, annealing_step)))
    return ce + anneal * dirichlet_kl(alp, num_classes, device)

def conf_loss(conf, T, y, num_classes, device):
    """
    Uncertainty-guided loss for a *global view-shared* transition matrix T.

    T has shape [C, C] and is shared by every training sample in this view.
    For a sample with observed label y_i, its diagonal retention probability
    T[y_i, y_i] is aligned with the sample confidence. For the other classes,
    the batch-wise class confidence mean is used, matching the original
    TMNR-style class-level regularization while avoiding sample-index lookup.
    """
    conf = conf.view(-1)
    y = y.long().view(-1)
    diag = torch.diagonal(T, offset=0, dim1=-2, dim2=-1)  # [C]

    class_sum = torch.zeros(num_classes, device=device)
    class_cnt = torch.bincount(y, minlength=num_classes).float().to(device)
    class_sum.scatter_add_(0, y, conf)
    class_mean = class_sum / (class_cnt + 1e-5)

    y_oh = torch.zeros(len(y), num_classes, device=device)
    y_oh.scatter_(1, y.view(-1, 1), 1.0)

    observed_diag_loss = ((conf.view(-1, 1) - diag.view(1, -1)) ** 2 * y_oh).sum(dim=1)
    other_diag_loss = ((class_mean.view(1, -1) - diag.view(1, -1)) ** 2 * (1.0 - y_oh)).sum(dim=1)
    return observed_diag_loss + other_diag_loss

def corrected_mse_loss(corrected_y, alpha_clean, noisy_mask):
    if noisy_mask is None or noisy_mask.sum() == 0:
        return None
    p = alpha_clean / alpha_clean.sum(dim=1, keepdim=True)
    return torch.sum((corrected_y[noisy_mask] - p[noisy_mask]) ** 2, dim=1).mean()

# =============================================================================
# Dual-view model
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
        feat_dim = hidden_dim * 2
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, classifier_hidden),
            nn.Tanh(),
            nn.Linear(classifier_hidden, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = feat_dim

    def forward(self, x, mask, lengths):
        lengths_cpu = lengths.detach().cpu().long()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))

        valid_mask = mask.bool()
        attn_scores = self.attn(out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~valid_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        feat = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)
        return self.dropout(feat)

class GraphSAGELayer(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.neigh_lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            out = self.self_lin(x)
        else:
            src, dst = edge_index[0], edge_index[1]
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, x[src])
            deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            agg = agg / deg.clamp(min=1.0).unsqueeze(-1)
            out = self.self_lin(x) + self.neigh_lin(agg)

        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(x + out)

def attention_pool(h, batch, attn):
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    scores = attn(h).squeeze(-1)
    pooled = []
    for g in range(num_graphs):
        mask = batch == g
        hg = h[mask]
        sg = scores[mask]
        wg = torch.softmax(sg, dim=0).view(-1, 1)
        pooled.append(torch.sum(wg * hg, dim=0))
    return torch.stack(pooled, dim=0)

class StructureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, classifier_hidden, dropout):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([GraphSAGELayer(hidden_dim, dropout) for _ in range(layers)])
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

def ds_evidence_fusion(e1: torch.Tensor, e2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Dempster-Shafer / Subjective Logic evidence fusion.

    输入:
        e1, e2: [B, K]，非负 evidence
    过程:
        alpha = e + 1
        S = sum(alpha) = sum(e) + K
        belief b_k = e_k / S
        uncertainty u = K / S

        C = sum_{i != j} b1_i * b2_j
        b_k = (b1_k*b2_k + b1_k*u2 + b2_k*u1) / (1-C)
        u   = (u1*u2) / (1-C)

        再由 b/u 反推 evidence:
        e_k = b_k * K / u

    这个融合发生在 evidence 层：
        - clean evidence 融合用于最终预测和 KNN-JS；
        - noisy evidence 融合用于 EDL 拟合 noisy label。
    """
    k = e1.size(1)

    e1 = torch.clamp(e1, min=0.0)
    e2 = torch.clamp(e2, min=0.0)

    s1 = e1.sum(dim=1, keepdim=True) + float(k)
    s2 = e2.sum(dim=1, keepdim=True) + float(k)

    b1 = e1 / s1
    b2 = e2 / s2
    u1 = float(k) / s1
    u2 = float(k) / s2

    bb = b1.unsqueeze(2) * b2.unsqueeze(1)  # [B, K, K]
    diag = torch.diagonal(bb, dim1=1, dim2=2).sum(dim=1, keepdim=True)
    conflict = (bb.sum(dim=(1, 2), keepdim=False).view(-1, 1) - diag).clamp(min=0.0, max=1.0 - eps)

    denom = (1.0 - conflict).clamp_min(eps)
    b = (b1 * b2 + b1 * u2 + b2 * u1) / denom
    u = (u1 * u2) / denom

    fused_e = b * (float(k) / u.clamp_min(eps))
    return torch.clamp(fused_e, min=0.0)

class DualModalTMNR2(nn.Module):
    """
    双模态 TMNR²：每个视图一个全局共享噪声转移矩阵 + noisy-evidence DS fusion。

    严格对应 TMNR² 的 view-specific T 设定：

        T_seq    ∈ R^{C×C}: 所有训练样本共享的序列视图噪声转移矩阵
        T_struct ∈ R^{C×C}: 所有训练样本共享的结构视图噪声转移矩阵

    对任意训练样本 i：
        e_seq_i clean evidence    -> T_seq    -> e_seq_i noisy evidence
        e_struct_i clean evidence -> T_struct -> e_struct_i noisy evidence

    两个 noisy evidence 再进行 Dempster-Shafer 融合并用 noisy label 监督。
    测试/验证时不使用 T，而是直接用 clean fused evidence 预测。
    """
    def __init__(self, seq_input_dim, graph_input_dim, args):
        super().__init__()
        self.fusion_type = args.fusion_type
        self.num_classes = 2

        self.seq_encoder = SequenceEncoder(
            input_dim=seq_input_dim,
            hidden_dim=args.hidden_dim,
            lstm_layers=args.lstm_layers,
            dropout=args.dropout,
            classifier_hidden=args.classifier_hidden,
        )
        self.struct_encoder = StructureEncoder(
            input_dim=graph_input_dim,
            hidden_dim=args.gnn_hidden_dim,
            layers=args.gnn_layers,
            classifier_hidden=args.classifier_hidden,
            dropout=args.dropout,
        )

        self.seq_head = EvidenceHead(
            input_dim=self.seq_encoder.out_dim,
            hidden_dim=args.classifier_hidden,
            dropout=args.dropout,
            num_classes=2,
        )
        self.struct_head = EvidenceHead(
            input_dim=self.struct_encoder.out_dim,
            hidden_dim=args.classifier_hidden,
            dropout=args.dropout,
            num_classes=2,
        )

        # 一个视图一个共享 T；二分类时每个矩阵为 [2, 2]。
        self.T_seq = nn.Parameter(torch.eye(self.num_classes))
        self.T_struct = nn.Parameter(torch.eye(self.num_classes))

    def encode(self, batch):
        h_seq = self.seq_encoder(batch["seq_x"], batch["seq_mask"], batch["seq_lengths"])
        h_struct = self.struct_encoder(batch["graph_x"], batch["edge_index"], batch["graph_batch"])
        return h_seq, h_struct

    def fuse_clean_evidence(self, e_seq, e_struct):
        if self.fusion_type == "ds":
            return ds_evidence_fusion(e_seq, e_struct)
        elif self.fusion_type == "sum":
            return e_seq + e_struct
        elif self.fusion_type == "mean":
            return 0.5 * (e_seq + e_struct)
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

    def fuse_noisy_evidence(self, e_seq_noisy, e_struct_noisy):
        # 两个视图先分别通过其共享 T，再在 noisy-evidence 空间进行融合。
        return ds_evidence_fusion(e_seq_noisy, e_struct_noisy)

    def evidence_from_features(self, h_seq, h_struct):
        e_seq = self.seq_head(h_seq)
        e_struct = self.struct_head(h_struct)
        e_clean_fused = self.fuse_clean_evidence(e_seq, e_struct)
        return e_seq, e_struct, e_clean_fused

    def transfer_evidence(self, evidence, T):
        """Apply one global row-stochastic transition matrix to every sample."""
        return evidence @ T

    def forward(self, batch, apply_noise_correction=False, return_features=False):
        h_seq, h_struct = self.encode(batch)
        e_seq, e_struct, e_clean_fused = self.evidence_from_features(h_seq, h_struct)

        if apply_noise_correction:
            e_seq_noisy = self.transfer_evidence(e_seq, self.T_seq)
            e_struct_noisy = self.transfer_evidence(e_struct, self.T_struct)
            e_noisy_fused = self.fuse_noisy_evidence(e_seq_noisy, e_struct_noisy)
        else:
            # 验证/测试：直接使用 clean evidence，不引入训练期 noise correction。
            e_seq_noisy = e_seq
            e_struct_noisy = e_struct
            e_noisy_fused = e_clean_fused

        if return_features:
            return e_seq, e_struct, e_clean_fused, e_seq_noisy, e_struct_noisy, e_noisy_fused, h_seq, h_struct
        return e_clean_fused, e_noisy_fused

    def t_consistency_loss(self):
        """Encourage the two view-level transition matrices to remain compatible."""
        return F.mse_loss(self.T_seq, self.T_struct)

    @torch.no_grad()
    def normalize_T_(self):
        for T in [self.T_seq, self.T_struct]:
            T.data.clamp_(min=0.0)
            T.data = T.data / (T.data.sum(dim=-1, keepdim=True) + 1e-12)

# =============================================================================
# SharedT candidate scoring and training backbone
# =============================================================================

def build_similarity_and_neighbors(mean_feat, train_idx, k, device):
    x = torch.tensor(mean_feat, dtype=torch.float32, device=device)
    n = x.size(0)
    dist2 = torch.cdist(x, x, p=2) ** 2
    nearest = torch.zeros((n, k), dtype=torch.long, device=device)
    sigma = torch.zeros((n, 1), dtype=torch.float32, device=device)

    train_list = [int(i) for i in train_idx.tolist()]
    train_set = set(train_list)

    for i in range(n):
        cand = [j for j in train_set if j != i]
        if len(cand) == 0:
            cand = [i]
        ct = torch.tensor(cand, dtype=torch.long, device=device)
        d = dist2[i, ct]
        order = torch.argsort(d)
        sel = ct[order[:min(k, len(cand))]]
        if sel.numel() < k:
            pad = sel[-1:].repeat(k - sel.numel()) if sel.numel() > 0 else torch.tensor([i] * k, device=device)
            sel = torch.cat([sel, pad], dim=0)
        nearest[i] = sel[:k]
        sigma[i] = torch.mean(dist2[i, nearest[i]]).clamp(min=1e-5)

    sim = torch.exp(-dist2 / (sigma ** 2))
    return sim, nearest

def js_divergence(p, q, eps=1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * torch.sum(p * (p.log() - m.log()), dim=-1) + \
           0.5 * torch.sum(q * (q.log() - m.log()), dim=-1)

@torch.no_grad()
def calibrate(threshold, evidences_all, nearest_indices, similarity_matrix, train_idx, noisy_labels, device, keep_soft_label=True):
    y = torch.tensor(noisy_labels, dtype=torch.long, device=device)
    y_oh = F.one_hot(y, num_classes=2).float()
    train_t = torch.tensor(train_idx, dtype=torch.long, device=device)

    probs = F.softmax(evidences_all, dim=1)

    self_nei = torch.cat([train_t.view(-1, 1), nearest_indices[train_t]], dim=1)
    neighbor_probs = probs[self_nei]
    label_expand = y_oh[train_t].unsqueeze(1).expand_as(neighbor_probs)

    js = js_divergence(label_expand, neighbor_probs)

    rows = train_t.unsqueeze(1).expand(-1, nearest_indices.size(1))
    sim_nei = similarity_matrix[rows, nearest_indices[train_t]]
    sim = torch.cat([torch.ones_like(sim_nei[:, :1]), sim_nei], dim=1)
    sim = torch.softmax(sim, dim=1)

    score = torch.sum(sim * js, dim=1)
    norm_score = (score - score.min()) / (score.max() - score.min() + 1e-12)

    candidate = train_t[norm_score > threshold]
    corrected = y_oh.clone()

    if candidate.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device), corrected.cpu().numpy(), {
            "noisy_candidate_count": 0,
            "changed_count": 0,
            "threshold": float(threshold),
            "score_mean": float(norm_score.mean().item()),
            "score_std": float(norm_score.std().item()),
        }

    err_nei = torch.cat([candidate.view(-1, 1), nearest_indices[candidate]], dim=1)
    err_e = evidences_all[err_nei]

    err_rows = candidate.unsqueeze(1).expand(-1, nearest_indices.size(1))
    err_sim_nei = similarity_matrix[err_rows, nearest_indices[candidate]]
    err_sim = torch.cat([torch.ones_like(err_sim_nei[:, :1]), err_sim_nei], dim=1)
    err_sim = torch.softmax(err_sim, dim=1).unsqueeze(-1)

    agg_e = torch.sum(err_sim * err_e, dim=1)
    pseudo = torch.softmax(agg_e, dim=1)

    pseudo_hard = torch.argmax(pseudo, dim=1)
    noisy_hard = y[candidate]
    changed_mask = pseudo_hard != noisy_hard

    error_index = candidate[changed_mask]
    pseudo = pseudo[changed_mask]

    if error_index.numel() > 0:
        if keep_soft_label:
            corrected[error_index] = pseudo
        else:
            corrected[error_index] = F.one_hot(torch.argmax(pseudo, dim=1), num_classes=2).float()

    return error_index, corrected.cpu().numpy(), {
        "noisy_candidate_count": int(candidate.numel()),
        "changed_count": int(error_index.numel()),
        "threshold": float(threshold),
        "score_mean": float(norm_score.mean().item()),
        "score_std": float(norm_score.std().item()),
    }

def noise_det_metrics(error_index, injected_noise_mask):
    pred = np.zeros_like(injected_noise_mask, dtype=bool)
    if error_index is not None and error_index.numel() > 0:
        pred[error_index.detach().cpu().numpy().astype(np.int64)] = True
    true = injected_noise_mask.astype(bool)
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {
        "noise_det_precision": float(precision),
        "noise_det_recall": float(recall),
        "noise_det_f1": float(f1),
        "noise_det_tp": tp,
        "noise_det_fp": fp,
        "noise_det_fn": fn,
    }

def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

@torch.no_grad()
def evaluate(model, seq_data, graph_data, args, labels_override=None):
    device = torch.device(args.device)
    ds = DualModalDataset(seq_data, graph_data, labels_override=labels_override)
    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_dual,
        pin_memory=True,
    )
    model.eval()

    ys, probs, ids, seqs = [], [], [], []
    for b in loader:
        b = move_batch_to_device(b, device)
        e_fused, _ = model(b, apply_noise_correction=False)
        alpha = e_fused + 1.0
        p = alpha / alpha.sum(dim=1, keepdim=True)
        probs.extend(p[:, 1].detach().cpu().numpy().tolist())
        ys.extend(b["y"].detach().cpu().numpy().astype(np.int64).tolist())
        ids.extend(b["id"])
        seqs.extend(b["seq"])

    return binary_metrics(np.asarray(ys), np.asarray(probs), ids=ids, seqs=seqs)

def train_dual_tmnr2(
    args,
    train_seq,
    train_graph,
    test_seq,
    test_graph,
    train_idx,
    val_idx,
    seq_input_dim,
    graph_input_dim,
    clean_labels_override=None,
    file_label_noise_mask_override=None,
):
    device = torch.device(args.device)
    # These are the labels actually supplied to the model. For predefined CSV
    # training splits, they come from the ``target`` column.
    observed_input_labels = train_seq["labels"].numpy().astype(np.int64)
    sample_num = len(observed_input_labels)

    if clean_labels_override is not None:
        clean_labels = np.asarray(clean_labels_override, dtype=np.int64)
        if clean_labels.shape != observed_input_labels.shape:
            raise ValueError(
                "clean_labels_override shape mismatch: "
                f"{clean_labels.shape} vs {observed_input_labels.shape}"
            )
        if not np.isin(clean_labels, [0, 1]).all():
            raise ValueError("clean_labels_override must contain only 0/1 values.")
        clean_ref_matched_mask = np.ones(sample_num, dtype=bool)
        clean_ref_conflict_count = 0
    else:
        # FASTA compatibility path: derive clean labels from optional clean
        # reference FASTA files.
        clean_labels, clean_ref_matched_mask, clean_ref_conflict_count = (
            build_clean_labels_from_reference(
                train_seq["seqs"], args, observed_input_labels
            )
        )

    derived_file_noise_mask = observed_input_labels != clean_labels
    if file_label_noise_mask_override is not None:
        file_label_noise_mask = np.asarray(
            file_label_noise_mask_override, dtype=bool
        )
        if file_label_noise_mask.shape != observed_input_labels.shape:
            raise ValueError(
                "file_label_noise_mask_override shape mismatch: "
                f"{file_label_noise_mask.shape} vs {observed_input_labels.shape}"
            )
        if not np.array_equal(file_label_noise_mask, derived_file_noise_mask):
            raise ValueError(
                "CSV is_noisy values disagree with target and clean_target."
            )
    else:
        file_label_noise_mask = derived_file_noise_mask

    # CSV inputs are predefined noisy-label files and must never be poisoned
    # again. FASTA inputs retain the previous file/internal/auto behavior.
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
            clean_labels, train_idx, args.noise_rate, args.seed
        )
        file_label_noise_mask = np.zeros_like(clean_labels, dtype=bool)
        internal_noise_rate_used = float(args.noise_rate)
    else:
        raise ValueError(f"Unsupported resolved_noise_source: {resolved_noise_source}")

    # Store bookkeeping on args so it can be reported in metrics.
    args._resolved_noise_source = resolved_noise_source
    args._internal_noise_rate_used = internal_noise_rate_used
    args._input_file_noise_count_all = int(file_label_noise_mask.sum())
    args._input_file_noise_count_train = int(file_label_noise_mask[train_idx].sum())
    args._input_file_noise_count_val = int(file_label_noise_mask[val_idx].sum())
    args._input_file_noise_rate_all = float(file_label_noise_mask.mean())
    args._input_file_noise_rate_train = float(file_label_noise_mask[train_idx].mean())
    args._input_file_noise_rate_val = float(file_label_noise_mask[val_idx].mean())
    args._effective_train_noise_rate = float(injected_noise_mask[train_idx].mean())
    args._clean_ref_matched_count = int(clean_ref_matched_mask.sum())
    args._clean_ref_missing_count = int((~clean_ref_matched_mask).sum())
    args._clean_ref_conflict_count = int(clean_ref_conflict_count)

    print(
        "📌 noise setting | "
        f"noise_source={resolved_noise_source} | "
        f"input_file_noise_all={args._input_file_noise_count_all}/{sample_num} "
        f"({args._input_file_noise_rate_all:.4f}) | "
        f"input_file_noise_train={args._input_file_noise_count_train}/{len(train_idx)} "
        f"({args._input_file_noise_rate_train:.4f}) | "
        f"input_file_noise_val_before_clean_override={args._input_file_noise_count_val}/{len(val_idx)} "
        f"({args._input_file_noise_rate_val:.4f}) | "
        f"internal_noise_rate_used={internal_noise_rate_used:.4f} | "
        f"effective_train_noise={args._effective_train_noise_rate:.4f} | "
        f"clean_ref_matched={args._clean_ref_matched_count}/{sample_num} | "
        f"clean_ref_conflicts={args._clean_ref_conflict_count}"
    )

    corrected_probs = F.one_hot(torch.tensor(noisy_labels, dtype=torch.long), num_classes=2).float().numpy()

    knn_feat = build_dual_knn_features(train_seq, train_graph, args)
    similarity_matrix, nearest_indices = build_similarity_and_neighbors(knn_feat, train_idx, args.knn_k, device)

    model = DualModalTMNR2(
        seq_input_dim=seq_input_dim,
        graph_input_dim=graph_input_dim,
        args=args,
    ).to(device)

    backbone_params = []
    t_params = []
    for name, p in model.named_parameters():
        if name in ["T_seq", "T_struct"]:
            t_params.append(p)
        else:
            backbone_params.append(p)

    opt_model = torch.optim.AdamW(backbone_params, lr=args.lr, weight_decay=args.weight_decay)
    opt_T = torch.optim.AdamW(t_params, lr=args.t_lr, weight_decay=args.weight_decay)

    best_score = -1.0
    best_state = None
    best_epoch = -1
    bad_epochs = 0

    total_error_index = torch.empty(0, dtype=torch.long, device=device)
    history = []
    last_cal = {}

    for epoch in range(1, args.epochs + 1):
        train_ds = DualModalDataset(
            train_seq,
            train_graph,
            indices=train_idx.tolist(),
            labels_override=noisy_labels,
            corrected_probs=corrected_probs,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_dual,
            pin_memory=True,
        )

        val_ds = DualModalDataset(train_seq, train_graph, indices=val_idx.tolist(), labels_override=clean_labels)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_dual,
            pin_memory=True,
        )

        model.train()
        evidences_all = torch.zeros(sample_num, 2, device=device)
        losses, ce_ls, aux_ls, conf_ls, sim_ls, tcons_ls, mse_ls, mix_ls = [], [], [], [], [], [], [], []

        for b in train_loader:
            b = move_batch_to_device(b, device)
            y = b["y"].long()
            idxs = b["idx"].long()
            cy = b["corrected_y"].float()
            yoh = to_onehot(y, 2).to(device)

            e_seq, e_struct, e_fused, e_seq_noisy, e_struct_noisy, e_noisy_fused, h_seq, h_struct = model(
                b, apply_noise_correction=True, return_features=True
            )

            alpha_fused_clean = e_fused + 1.0
            alpha_noisy_fused = e_noisy_fused + 1.0

            # 主损失：两个视图各自过 T 后得到 noisy evidence，
            # 再用 Dempster-Shafer 证据理论融合成 noisy fused evidence，
            # 最后拟合 noisy label。
            loss_ce = edl_ce_loss(yoh, alpha_noisy_fused, 2, epoch, args.annealing_epoch, device).mean()

            # 辅助损失：每个视图自己的 noisy evidence 也拟合 noisy label。
            loss_aux = torch.tensor(0.0, device=device)
            if args.lambda_aux > 0:
                loss_seq_aux = edl_ce_loss(yoh, e_seq_noisy + 1.0, 2, epoch, args.annealing_epoch, device).mean()
                loss_struct_aux = edl_ce_loss(yoh, e_struct_noisy + 1.0, 2, epoch, args.annealing_epoch, device).mean()
                loss_aux = 0.5 * (loss_seq_aux + loss_struct_aux)

            loss_tcons = model.t_consistency_loss()

            uncertainty = 2.0 / torch.sum(alpha_noisy_fused, dim=1)
            conf = 1.0 - uncertainty
            loss_conf = 0.5 * (
                conf_loss(conf, model.T_seq, y, 2, device).mean()
                + conf_loss(conf, model.T_struct, y, 2, device).mean()
            )
            # Global view-level T is identical for all samples, so the former
            # neighbor transition-matrix smoothness term is mathematically zero
            # and is intentionally disabled. KNN is still used by calibrate().
            loss_sim = torch.tensor(0.0, device=device)

            if total_error_index.numel() > 0:
                noisy_mask = torch.isin(idxs, total_error_index)
                lm = corrected_mse_loss(cy, alpha_fused_clean, noisy_mask)
                loss_mse = lm if lm is not None else torch.tensor(0.0, device=device)
            else:
                noisy_mask = torch.zeros_like(y, dtype=torch.bool, device=device)
                loss_mse = torch.tensor(0.0, device=device)

            loss_mix = torch.tensor(0.0, device=device)
            if args.use_mixup and noisy_mask.any() and (~noisy_mask).any():
                noisy_pos = torch.nonzero(noisy_mask, as_tuple=False).view(-1)
                clean_pos_pool = torch.nonzero(~noisy_mask, as_tuple=False).view(-1)
                rand = torch.randint(0, clean_pos_pool.numel(), (noisy_pos.numel(),), device=device)
                clean_pos = clean_pos_pool[rand]

                lam = torch.distributions.Beta(args.mixup_alpha, args.mixup_alpha).sample((noisy_pos.numel(),)).to(device)
                lam = torch.max(lam, 1.0 - lam).view(-1, 1)

                h_seq_mix = lam * h_seq[noisy_pos] + (1.0 - lam) * h_seq[clean_pos]
                h_struct_mix = lam * h_struct[noisy_pos] + (1.0 - lam) * h_struct[clean_pos]
                y_mix = lam * cy[noisy_pos] + (1.0 - lam) * yoh[clean_pos]

                _, _, e_mix = model.evidence_from_features(h_seq_mix, h_struct_mix)
                alpha_mix = e_mix + 1.0
                p_mix = alpha_mix / alpha_mix.sum(dim=1, keepdim=True)
                loss_mix = torch.sum((y_mix - p_mix) ** 2, dim=1).mean()

            loss = (
                loss_ce
                + args.lambda_aux * loss_aux
                + args.lambda_t_consistency * loss_tcons
                + args.lambda_conf * loss_conf
                + args.lambda_mse * loss_mse
                + args.lambda_mix * loss_mix
            )

            opt_model.zero_grad()
            opt_T.zero_grad()
            loss.backward()

            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            opt_model.step()
            if epoch > args.warmup_epochs:
                opt_T.step()
                model.normalize_T_()

            evidences_all[idxs] = e_fused.detach()

            losses.append(float(loss.item()))
            ce_ls.append(float(loss_ce.item()))
            aux_ls.append(float(loss_aux.item()))
            conf_ls.append(float(loss_conf.item()))
            sim_ls.append(0.0)
            tcons_ls.append(float(loss_tcons.item()))
            mse_ls.append(float(loss_mse.item()))
            mix_ls.append(float(loss_mix.item()))

        # validation
        model.eval()
        val_probs, val_y = [], []
        with torch.no_grad():
            for b in val_loader:
                b = move_batch_to_device(b, device)
                e_fused, _ = model(b, apply_noise_correction=False)
                alpha = e_fused + 1.0
                p = alpha / alpha.sum(dim=1, keepdim=True)
                val_probs.extend(p[:, 1].detach().cpu().numpy().tolist())
                val_y.extend(b["y"].detach().cpu().numpy().astype(np.int64).tolist())

        val_metrics, _ = binary_metrics(np.asarray(val_y), np.asarray(val_probs))
        score = val_metrics["AUC"] if args.monitor == "val_auc" else (val_metrics["F1"] if args.monitor == "val_f1" else val_metrics["Accuracy"])

        # Do not let early epochs before label correction lock the best checkpoint.
        # Correction at epoch=start_correct affects training from the next epoch,
        # so --best_after_start_correct starts best selection from start_correct + 1.
        min_best_epoch = int(getattr(args, "min_best_epoch", 1))
        if getattr(args, "best_after_start_correct", False):
            min_best_epoch = max(min_best_epoch, int(args.start_correct) + 1)
        can_save_best = epoch >= min_best_epoch

        if can_save_best and score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        elif can_save_best:
            bad_epochs += 1

        if epoch >= args.start_correct and (epoch - args.start_correct) % args.correct_every == 0:
            error_index, corrected_probs, last_cal = calibrate(
                threshold=args.threshold,
                evidences_all=evidences_all.detach(),
                nearest_indices=nearest_indices,
                similarity_matrix=similarity_matrix,
                train_idx=train_idx,
                noisy_labels=noisy_labels,
                device=device,
                keep_soft_label=args.keep_soft_label,
            )
            total_error_index = torch.unique(torch.cat([total_error_index, error_index])) if total_error_index.numel() > 0 else error_index
            if args.threshold_decay != 1.0:
                args.threshold = max(args.min_threshold, args.threshold * args.threshold_decay)

        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "loss_ce": float(np.mean(ce_ls)),
            "loss_aux": float(np.mean(aux_ls)),
            "loss_conf": float(np.mean(conf_ls)),
            "loss_sim": float(np.mean(sim_ls)),
            "loss_tcons": float(np.mean(tcons_ls)),
            "loss_mse": float(np.mean(mse_ls)),
            "loss_mix": float(np.mean(mix_ls)),
            "val_acc": val_metrics["Accuracy"],
            "val_f1": val_metrics["F1"],
            "val_auc": val_metrics["AUC"],
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "detected_noisy_total": int(total_error_index.numel()),
        }
        history.append(row)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == best_epoch:
            print(
                f"[dual-tmnr2-{args.fusion_type} {epoch:03d}/{args.epochs}] "
                f"loss={row['loss']:.5f} ce={row['loss_ce']:.5f} aux={row['loss_aux']:.5f} "
                f"conf={row['loss_conf']:.5f} sim={row['loss_sim']:.5f} "
                f"tcons={row['loss_tcons']:.5f} mse={row['loss_mse']:.5f} mix={row['loss_mix']:.5f} | "
                f"val_acc={row['val_acc']:.4f} val_f1={row['val_f1']:.4f} val_auc={row['val_auc']:.4f} | "
                f"detected={row['detected_noisy_total']} | best={best_score:.4f}@{best_epoch}"
            )

        if bad_epochs >= args.patience:
            print(f"⏹ Early stopping at epoch={epoch}, best_epoch={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, pred_rows = evaluate(model, test_seq, test_graph, args)
    test_metrics.update({
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "min_best_epoch": int(getattr(args, "min_best_epoch", 1)),
        "best_after_start_correct": bool(getattr(args, "best_after_start_correct", False)),
        "noise_rate": float(args.noise_rate),
        "noise_source": getattr(args, "_resolved_noise_source", getattr(args, "noise_source", "unknown")),
        "internal_noise_rate_used": float(getattr(args, "_internal_noise_rate_used", args.noise_rate)),
        "input_file_noise_count_all": int(getattr(args, "_input_file_noise_count_all", 0)),
        "input_file_noise_count_train": int(getattr(args, "_input_file_noise_count_train", 0)),
        "input_file_noise_count_val_before_clean_override": int(getattr(args, "_input_file_noise_count_val", 0)),
        "input_file_noise_rate_all": float(getattr(args, "_input_file_noise_rate_all", 0.0)),
        "input_file_noise_train_rate": float(getattr(args, "_input_file_noise_rate_train", 0.0)),
        "input_file_noise_val_rate": float(getattr(args, "_input_file_noise_rate_val", 0.0)),
        "effective_train_noise_rate": float(getattr(args, "_effective_train_noise_rate", 0.0)),
        "clean_ref_matched_count": int(getattr(args, "_clean_ref_matched_count", 0)),
        "clean_ref_missing_count": int(getattr(args, "_clean_ref_missing_count", 0)),
        "clean_ref_conflict_count": int(getattr(args, "_clean_ref_conflict_count", 0)),
        "warmup_epochs": int(args.warmup_epochs),
        "start_correct": int(args.start_correct),
        "correct_every": int(args.correct_every),
        "knn_k": int(args.knn_k),
        "knn_feature": args.knn_feature,
        "fusion_type": args.fusion_type,
        "view_T_noisy_evidence_fusion": True,
        "transition_matrix_mode": "global_view_shared",
        "T_seq_shape": list(model.T_seq.shape),
        "T_struct_shape": list(model.T_struct.shape),
        "lambda_aux": float(args.lambda_aux),
        "lambda_t_consistency": float(args.lambda_t_consistency),
        "lambda_sim_effective": 0.0,
        "threshold_final": float(args.threshold),
        "detected_noisy_total": int(total_error_index.numel()),
        "last_calibration": last_cal,
        **noise_det_metrics(total_error_index, injected_noise_mask),
    })

    return model, test_metrics, pred_rows, history, corrected_probs, noisy_labels, injected_noise_mask, observed_input_labels, file_label_noise_mask, clean_labels

def save_outputs(args, model, metrics, pred_rows, history, corrected_probs, noisy_labels, injected_noise_mask, train_seq, train_graph, observed_input_labels=None, file_label_noise_mask=None, clean_labels_override=None):
    os.makedirs(args.result_dir, exist_ok=True)

    paths = {
        "metrics": os.path.join(args.result_dir, "metrics_dual_tmnr2.json"),
        "pred": os.path.join(args.result_dir, "test_predictions_dual_tmnr2.csv"),
        "model": os.path.join(args.result_dir, "best_dual_tmnr2.pt"),
        "history": os.path.join(args.result_dir, "train_history_dual_tmnr2.csv"),
        "train_info": os.path.join(args.result_dir, "train_dual_tmnr2_info.csv"),
    }

    with open(paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(paths["pred"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "seq", "true_label", "prob_amp", "pred_label"])
        w.writeheader()
        w.writerows(pred_rows)

    with open(paths["history"], "w", newline="", encoding="utf-8") as f:
        fieldnames = list(history[0].keys()) if history else ["epoch"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(history)

    clean = train_seq["labels"].numpy().astype(np.int64)
    if clean_labels_override is not None:
        clean = np.asarray(clean_labels_override).astype(np.int64)
    if observed_input_labels is None:
        observed_input_labels = train_seq["labels"].numpy().astype(np.int64)
    if file_label_noise_mask is None:
        file_label_noise_mask = np.zeros(len(clean), dtype=bool)

    rows = []
    for i in range(len(clean)):
        g = train_graph["graphs"][i]
        rows.append({
            "id": train_seq["ids"][i],
            "seq": train_seq["seqs"][i],
            "observed_input_label": int(observed_input_labels[i]),
            "clean_label": int(clean[i]),
            "file_label_is_noisy": int(bool(file_label_noise_mask[i])),
            "noisy_label": int(noisy_labels[i]),
            "injected_noise": int(bool(injected_noise_mask[i])),
            "corrected_nonamp": float(corrected_probs[i, 0]),
            "corrected_amp": float(corrected_probs[i, 1]),
            "pdb_path": g.get("pdb_path", ""),
            "plddt_mean": float(g.get("plddt_mean", 0.0)),
            "plddt_min": float(g.get("plddt_min", 0.0)),
            "length_struct": int(g.get("length_struct", 0)),
        })

    with open(paths["train_info"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id", "seq", "observed_input_label", "clean_label", "file_label_is_noisy",
                "noisy_label", "injected_noise",
                "corrected_nonamp", "corrected_amp",
                "pdb_path", "plddt_mean", "plddt_min", "length_struct",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    torch.save(model.state_dict(), paths["model"])

    print("\n📊 Test metrics")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    for k, p in paths.items():
        print(f"✅ saved {k}: {p}")

# =============================================================================
# EC-RML-Safe configuration and state
# =============================================================================

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {raw!r}") from exc


CFG = {
    # Candidate bands. args.threshold is the high anchor and should be 0.80.
    "low_threshold": _env_float("EC_LOW_THRESHOLD", 0.70),

    # pLDDT-derived sample-level reliability gate.
    "r_low_q": _env_float("EC_RLOW_Q", 0.15),
    "r_high_q": _env_float("EC_RHIGH_Q", 0.70),

    # Borderline soft-update strength and filters.
    "eta_max": _env_float("EC_ETA_MAX", 0.30),
    "seq_conf_min": _env_float("EC_SEQ_CONF_MIN", 0.60),
    "nbr_conf_min": _env_float("EC_NBR_CONF_MIN", 0.60),
    "pseudo_conf_min": _env_float("EC_PSEUDO_CONF_MIN", 0.55),
    "margin_min": _env_float("EC_MARGIN_MIN", 0.10),
    "min_eta": _env_float("EC_MIN_ETA", 0.02),

    # EMA prediction memory for borderline stability.
    "ema_alpha": _env_float("EC_EMA_ALPHA", 0.70),
    "ema_conf_min": _env_float("EC_EMA_CONF_MIN", 0.55),
    "ema_stability_floor": _env_float("EC_EMA_STABILITY_FLOOR", 0.60),

    # Pseudo-label fusion weights for borderline candidates.
    "w_seq": _env_float("EC_W_SEQ", 0.35),
    "w_nbr": _env_float("EC_W_NBR", 0.35),
    "w_ema": _env_float("EC_W_EMA", 0.15),
    "w_struct": _env_float("EC_W_STRUCT", 0.15),

    # If reliable structure conflicts with the pseudo-label, reduce eta.
    # If structure is unreliable, this penalty is small; unreliable structure
    # should not veto sequence-neighbor supported correction.
    "reliable_conflict_lambda": _env_float("EC_RELIABLE_CONFLICT_LAMBDA", 0.50),
    "conflict_floor": _env_float("EC_CONFLICT_FLOOR", 0.35),
}

if not (0.0 <= CFG["r_low_q"] < CFG["r_high_q"] <= 1.0):
    raise ValueError("Require 0 <= EC_RLOW_Q < EC_RHIGH_Q <= 1.")
if not (0.0 <= CFG["low_threshold"] <= 1.0):
    raise ValueError("Require 0 <= EC_LOW_THRESHOLD <= 1.")
if not (0.0 <= CFG["eta_max"] <= 1.0):
    raise ValueError("Require 0 <= EC_ETA_MAX <= 1.")
if min(CFG["w_seq"], CFG["w_nbr"], CFG["w_ema"], CFG["w_struct"]) < 0:
    raise ValueError("EC pseudo-label fusion weights must be non-negative.")
if CFG["w_seq"] + CFG["w_nbr"] + CFG["w_ema"] + CFG["w_struct"] <= 0:
    raise ValueError("At least one EC pseudo-label fusion weight must be positive.")

STATE = {}

def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, device=device, dtype=torch.float32)

def _reliability_from_graphs(graphs, train_idx):
    """
    Robust sample-level predicted-structure reliability.

    r = 0.50 * mean(pLDDT) + 0.30 * q20(pLDDT)
        + 0.20 * (1 - fraction[pLDDT < 0.50])

    In the unmodified graph builder, the first node feature is the normalised
    pLDDT value unless --disable_plddt_feature is used. Missing/fallback graphs
    become low-reliability by construction.
    """
    values = []
    for graph in graphs:
        x = graph.get("x")
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.numel() == 0 or x.size(1) == 0:
            values.append(0.0)
            continue
        p = x[:, 0].detach().cpu().numpy().astype(np.float32)
        p = np.clip(p, 0.0, 1.0)
        if p.size == 0:
            values.append(0.0)
            continue
        p_mean = float(np.mean(p))
        p_q20 = float(np.quantile(p, 0.20))
        p_low_ratio = float(np.mean(p < 0.50))
        r = 0.50 * p_mean + 0.30 * p_q20 + 0.20 * (1.0 - p_low_ratio)
        values.append(float(np.clip(r, 0.0, 1.0)))

    r = np.asarray(values, dtype=np.float32)
    tr = r[np.asarray(train_idx, dtype=np.int64)]
    r_low = float(np.quantile(tr, CFG["r_low_q"]))
    r_high = float(np.quantile(tr, CFG["r_high_q"]))
    if r_high <= r_low + 1e-8:
        r_high = min(1.0, r_low + 1e-6)

    # Tri-level participation gate: 0 = abstain, (0,1) = downweight, 1 = normal.
    rho = np.zeros_like(r, dtype=np.float32)
    mid = (r >= r_low) & (r < r_high)
    high = r >= r_high
    rho[mid] = (r[mid] - r_low) / (r_high - r_low + 1e-12)
    rho[high] = 1.0
    return r, rho, r_low, r_high

def _initialise_state(train_graph, train_idx, device):
    r, rho, r_low, r_high = _reliability_from_graphs(train_graph["graphs"], train_idx)
    n = int(len(r))
    STATE.clear()
    STATE.update({
        "r": _to_tensor(r, device),
        "rho": _to_tensor(rho, device),
        "r_low": r_low,
        "r_high": r_high,
        "e_seq": torch.zeros((n, 2), device=device),
        "e_struct": torch.zeros((n, 2), device=device),
        "e_fused": torch.zeros((n, 2), device=device),
        "ema_probs": torch.zeros((n, 2), device=device),
        "ema_initialized": False,
        "last_score": torch.full((n,), float("nan"), device=device),
        "last_candidate": torch.zeros(n, dtype=torch.bool, device=device),
        "last_changed": torch.zeros(n, dtype=torch.bool, device=device),
        "last_high_candidate": torch.zeros(n, dtype=torch.bool, device=device),
        "last_border_candidate": torch.zeros(n, dtype=torch.bool, device=device),
        "last_border_pass": torch.zeros(n, dtype=torch.bool, device=device),
        "last_eta": torch.zeros(n, dtype=torch.float32, device=device),
        "last_region_code": torch.zeros(n, dtype=torch.long, device=device),
    })

# Cache per-view evidence required by reliability-aware routing.

_ORIG_FORWARD = DualModalTMNR2.forward

def _patched_forward(self, batch, apply_noise_correction=False, return_features=False):
    output = _ORIG_FORWARD(
        self,
        batch,
        apply_noise_correction=apply_noise_correction,
        return_features=return_features,
    )

    if return_features and STATE and "idx" in batch:
        e_seq, e_struct, e_fused = output[0], output[1], output[2]
        idx = batch["idx"].detach().long()
        STATE["e_seq"][idx] = e_seq.detach()
        STATE["e_struct"][idx] = e_struct.detach()
        STATE["e_fused"][idx] = e_fused.detach()

    return output

DualModalTMNR2.forward = _patched_forward

# =============================================================================
# EC-RML-Safe calibration
# =============================================================================

def _normalise_local(score):
    return (score - score.min()) / (score.max() - score.min() + 1e-12)

def _binary_margin(probs):
    # For binary probabilities, top1-top2 equals |p1-p0|.
    return torch.abs(probs[:, 1] - probs[:, 0])

def _discount_struct_probs(e_struct, rho):
    """
    Subjective-logic style structural evidence discount.

    Evidence e -> belief b=e/S and uncertainty u=K/S.
    Discount by external structure confidence rho:
      b' = rho * b
      u' = 1 - rho + rho * u
      p' = b' + u'/K

    When rho=0, p' is uniform; when rho=1, p' is the original EDL probability.
    """
    k = e_struct.size(1)
    e = torch.clamp(e_struct, min=0.0)
    alpha = e + 1.0
    strength = torch.sum(alpha, dim=1, keepdim=True).clamp_min(1e-12)
    belief = e / strength
    uncertainty = float(k) / strength

    rho = rho.view(-1, 1).clamp(0.0, 1.0)
    belief_d = rho * belief
    uncertainty_d = 1.0 - rho + rho * uncertainty
    probs = belief_d + uncertainty_d / float(k)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

@torch.no_grad()
def _aggregate_neighbor_probs(indices, evidences_all, nearest_indices, similarity_matrix):
    """Aggregate current fused evidence over self + KNN neighbors."""
    if indices.numel() == 0:
        return torch.empty((0, evidences_all.size(1)), device=evidences_all.device)
    err_nei = torch.cat([indices.view(-1, 1), nearest_indices[indices]], dim=1)
    err_e = evidences_all[err_nei]

    err_rows = indices.unsqueeze(1).expand(-1, nearest_indices.size(1))
    err_sim_nei = similarity_matrix[err_rows, nearest_indices[indices]]
    err_sim = torch.cat([torch.ones_like(err_sim_nei[:, :1]), err_sim_nei], dim=1)
    err_sim = torch.softmax(err_sim, dim=1).unsqueeze(-1)

    agg_e = torch.sum(err_sim * err_e, dim=1)
    return torch.softmax(agg_e, dim=1)

_ORIGINAL_CALIBRATE = calibrate

@torch.no_grad()
def ec_rml_calibrate(
    threshold,
    evidences_all,
    nearest_indices,
    similarity_matrix,
    train_idx,
    noisy_labels,
    device,
    keep_soft_label=True,
):
    """
    EC-RML-Safe calibration.

    Safe principle
    --------------
    Keep the original SharedT-080 candidate set unchanged:
        candidate iff original SharedT score S >= threshold.

    Do NOT update the 0.70-0.80 borderline band. The goal of this version is
    not to increase candidate recall, but to reduce false corrections caused by
    unreliable predicted structures inside the original SharedT-080 candidates.

    For each SharedT-080 high-confidence candidate, the original neighbor pseudo
    label is treated as the candidate proposal. Before updating the soft label,
    EC-RML-Safe reconstructs the pseudo-label by fusing:
        sequence evidence + neighbor consensus + EMA memory + pLDDT-discounted structure evidence.

    A high candidate is updated only if the reliability-routed pseudo-label still
    supports flipping the observed label. This makes the update set a subset of
    the original SharedT-080 proposal set and prevents unreliable structure from
    adding extra corrections.
    """
    if not STATE:
        return _ORIGINAL_CALIBRATE(
            threshold, evidences_all, nearest_indices, similarity_matrix,
            train_idx, noisy_labels, device, keep_soft_label,
        )

    high_threshold = float(threshold)
    low_threshold = float(CFG["low_threshold"])

    y = torch.as_tensor(noisy_labels, dtype=torch.long, device=device)
    y_oh = F.one_hot(y, num_classes=2).float()
    train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    n = int(y.numel())

    # Original SharedT candidate score: KNN-weighted JS divergence between
    # observed labels and current fused predictions. This intentionally ignores
    # pLDDT, so external structure confidence does not affect candidate detection.
    probs = torch.softmax(evidences_all, dim=1)

    # Update EMA prediction memory on training samples.
    was_ema_initialized = bool(STATE["ema_initialized"])
    if not was_ema_initialized:
        STATE["ema_probs"][train_t] = probs[train_t].detach()
        STATE["ema_initialized"] = True
    else:
        alpha = float(CFG["ema_alpha"])
        STATE["ema_probs"][train_t] = (
            alpha * STATE["ema_probs"][train_t] + (1.0 - alpha) * probs[train_t].detach()
        )
        STATE["ema_probs"][train_t] = STATE["ema_probs"][train_t] / STATE["ema_probs"][train_t].sum(dim=1, keepdim=True).clamp_min(1e-12)

    self_nei = torch.cat([train_t.view(-1, 1), nearest_indices[train_t]], dim=1)
    neighbor_probs = probs[self_nei]
    label_expand = y_oh[train_t].unsqueeze(1).expand_as(neighbor_probs)
    js = js_divergence(label_expand, neighbor_probs)

    rows = train_t.unsqueeze(1).expand(-1, nearest_indices.size(1))
    sim_nei = similarity_matrix[rows, nearest_indices[train_t]]
    sim = torch.cat([torch.ones_like(sim_nei[:, :1]), sim_nei], dim=1)
    sim = torch.softmax(sim, dim=1)

    score_local = torch.sum(sim * js, dim=1)
    norm_score_local = _normalise_local(score_local)
    norm_score = torch.zeros(n, device=device)
    norm_score[train_t] = norm_score_local

    # Candidate set is exactly anchored at SharedT-080.
    high_candidate = train_t[norm_score[train_t] >= high_threshold]

    # Borderline band is recorded only for diagnostics. It is NOT updated.
    border_candidate = train_t[
        (norm_score[train_t] >= low_threshold) & (norm_score[train_t] < high_threshold)
    ]

    corrected = y_oh.clone()
    changed_mask_global = torch.zeros(n, dtype=torch.bool, device=device)
    candidate_mask_global = torch.zeros(n, dtype=torch.bool, device=device)
    high_mask_global = torch.zeros(n, dtype=torch.bool, device=device)
    border_mask_global = torch.zeros(n, dtype=torch.bool, device=device)
    border_pass_global = torch.zeros(n, dtype=torch.bool, device=device)
    eta_global = torch.zeros(n, dtype=torch.float32, device=device)
    region_code = torch.zeros(n, dtype=torch.long, device=device)  # 0 none, 1 high, 2 border, 3 blocked-high

    high_mask_global[high_candidate] = True
    border_mask_global[border_candidate] = True
    candidate_mask_global[high_candidate] = True
    region_code[high_candidate] = 1
    region_code[border_candidate] = 2

    # ------------------------------------------------------------------
    # High-confidence candidates only: original proposal + EC evidence routing.
    # ------------------------------------------------------------------
    high_changed = torch.empty(0, dtype=torch.long, device=device)
    high_original_changed = torch.empty(0, dtype=torch.long, device=device)
    high_blocked = torch.empty(0, dtype=torch.long, device=device)
    high_routed_conf = torch.empty(0, dtype=torch.float32, device=device)
    high_routed_margin = torch.empty(0, dtype=torch.float32, device=device)

    if high_candidate.numel() > 0:
        # Original SharedT neighbor proposal. This defines which high candidates
        # would have been changed by the baseline.
        p_nbr = _aggregate_neighbor_probs(
            high_candidate, evidences_all, nearest_indices, similarity_matrix
        )
        nbr_hard = torch.argmax(p_nbr, dim=1)
        obs = y[high_candidate]
        original_flip = nbr_hard != obs
        high_original_changed = high_candidate[original_flip]

        # External-confidence routed pseudo-label.
        p_seq = torch.softmax(STATE["e_seq"][high_candidate], dim=1)
        e_struct = STATE["e_struct"][high_candidate]
        rho = STATE["rho"][high_candidate].clamp(0.0, 1.0)
        p_struct = _discount_struct_probs(e_struct, rho)
        p_ema = STATE["ema_probs"][high_candidate]
        p_ema = p_ema / p_ema.sum(dim=1, keepdim=True).clamp_min(1e-12)

        w_seq = float(CFG["w_seq"])
        w_nbr = float(CFG["w_nbr"])
        w_ema = float(CFG["w_ema"])
        w_struct = float(CFG["w_struct"])
        denom = max(w_seq + w_nbr + w_ema + w_struct, 1e-12)
        pseudo = (w_seq * p_seq + w_nbr * p_nbr + w_ema * p_ema + w_struct * p_struct) / denom
        pseudo = pseudo / pseudo.sum(dim=1, keepdim=True).clamp_min(1e-12)

        pseudo_hard = torch.argmax(pseudo, dim=1)
        pseudo_conf = torch.max(pseudo, dim=1).values
        pseudo_margin = _binary_margin(pseudo)
        high_routed_conf = pseudo_conf
        high_routed_margin = pseudo_margin

        # EC-RML-Safe never adds changes beyond original SharedT-080 proposals.
        # It only keeps the original flip if the reliability-routed pseudo-label
        # still supports flipping the observed label with enough confidence.
        safe_update = (
            original_flip
            & (pseudo_hard != obs)
            & (pseudo_conf >= float(CFG["pseudo_conf_min"]))
            & (pseudo_margin >= float(CFG["margin_min"]))
        )

        high_changed = high_candidate[safe_update]
        high_pseudo_changed = pseudo[safe_update]
        high_blocked = high_candidate[original_flip & (~safe_update)]
        if high_blocked.numel() > 0:
            region_code[high_blocked] = 3

        if high_changed.numel() > 0:
            if keep_soft_label:
                corrected[high_changed] = high_pseudo_changed
            else:
                corrected[high_changed] = F.one_hot(
                    torch.argmax(high_pseudo_changed, dim=1), num_classes=2
                ).float()
            changed_mask_global[high_changed] = True
            eta_global[high_changed] = 1.0

    error_index = high_changed if high_changed.numel() else torch.empty(0, dtype=torch.long, device=device)

    # Diagnostics for last-round audit.
    train_rho = STATE["rho"][train_t]
    high_rho = STATE["rho"][high_candidate] if high_candidate.numel() else torch.empty(0, device=device)
    border_rho = STATE["rho"][border_candidate] if border_candidate.numel() else torch.empty(0, device=device)
    changed_rho = STATE["rho"][high_changed] if high_changed.numel() else torch.empty(0, device=device)
    blocked_rho = STATE["rho"][high_blocked] if high_blocked.numel() else torch.empty(0, device=device)

    def _count_regions(x):
        if x.numel() == 0:
            return 0, 0, 0
        abstain = int((x <= 1e-8).sum().item())
        normal = int((x >= 1.0 - 1e-8).sum().item())
        down = int(x.numel() - abstain - normal)
        return abstain, down, normal

    train_abstain, train_down, train_normal = _count_regions(train_rho)
    high_abstain, high_down, high_normal = _count_regions(high_rho)
    border_abstain, border_down, border_normal = _count_regions(border_rho)
    changed_abstain, changed_down, changed_normal = _count_regions(changed_rho)
    blocked_abstain, blocked_down, blocked_normal = _count_regions(blocked_rho)

    diagnostics = {
        "ec_rml_variant": "EC-RML-Safe",
        "threshold_high": high_threshold,
        "threshold_low": low_threshold,
        "threshold": high_threshold,
        "score_mean": float(norm_score_local.mean().item()),
        "score_std": float(norm_score_local.std().item()),
        "reliability_low_threshold": float(STATE["r_low"]),
        "reliability_high_threshold": float(STATE["r_high"]),
        "train_struct_abstain_count": train_abstain,
        "train_struct_downweight_count": train_down,
        "train_struct_normal_count": train_normal,
        "train_struct_reliability_mean": float(STATE["r"][train_t].mean().item()),
        "train_struct_rho_mean": float(train_rho.mean().item()),
        "noisy_candidate_count": int(high_candidate.numel()),
        "changed_count": int(error_index.numel()),
        "high_candidate_count": int(high_candidate.numel()),
        "high_original_changed_count": int(high_original_changed.numel()),
        "high_changed_count": int(high_changed.numel()),
        "high_blocked_by_routing_count": int(high_blocked.numel()),
        "high_routed_conf_mean": float(high_routed_conf.mean().item()) if high_routed_conf.numel() else float("nan"),
        "high_routed_margin_mean": float(high_routed_margin.mean().item()) if high_routed_margin.numel() else float("nan"),
        "border_candidate_count": int(border_candidate.numel()),
        "border_pass_agreement_count": 0,
        "border_updated_count": 0,
        "border_eta_mean": float("nan"),
        "border_eta_std": float("nan"),
        "high_candidate_struct_abstain_count": high_abstain,
        "high_candidate_struct_downweight_count": high_down,
        "high_candidate_struct_normal_count": high_normal,
        "high_changed_struct_abstain_count": changed_abstain,
        "high_changed_struct_downweight_count": changed_down,
        "high_changed_struct_normal_count": changed_normal,
        "high_blocked_struct_abstain_count": blocked_abstain,
        "high_blocked_struct_downweight_count": blocked_down,
        "high_blocked_struct_normal_count": blocked_normal,
        "border_candidate_struct_abstain_count": border_abstain,
        "border_candidate_struct_downweight_count": border_down,
        "border_candidate_struct_normal_count": border_normal,
        "border_updated_struct_abstain_count": 0,
        "border_updated_struct_downweight_count": 0,
        "border_updated_struct_normal_count": 0,
        "border_updated_struct_rho_mean": float("nan"),
        "ema_alpha": float(CFG["ema_alpha"]),
        "eta_max": 1.0,
        "seq_conf_min": float(CFG["seq_conf_min"]),
        "nbr_conf_min": float(CFG["nbr_conf_min"]),
        "pseudo_conf_min": float(CFG["pseudo_conf_min"]),
        "margin_min": float(CFG["margin_min"]),
        "min_eta": float(CFG["min_eta"]),
        "uses_external_confidence_for_candidate_detection": False,
        "uses_external_confidence_for_evidence_routing": True,
        "uses_borderline_weak_soft_update": False,
    }

    STATE["last_score"] = norm_score.detach().clone()
    STATE["last_candidate"] = candidate_mask_global.detach().clone()
    STATE["last_changed"] = changed_mask_global.detach().clone()
    STATE["last_high_candidate"] = high_mask_global.detach().clone()
    STATE["last_border_candidate"] = border_mask_global.detach().clone()
    STATE["last_border_pass"] = border_pass_global.detach().clone()
    STATE["last_eta"] = eta_global.detach().clone()
    STATE["last_region_code"] = region_code.detach().clone()

    return error_index, corrected.cpu().numpy(), diagnostics

calibrate = ec_rml_calibrate

# =============================================================================
# Safe training/output wrappers
# =============================================================================

_ORIGINAL_TRAIN = train_dual_tmnr2

def _patched_train(*args, **kwargs):
    if kwargs:
        train_args = kwargs["args"]
        train_graph = kwargs["train_graph"]
        train_idx = kwargs["train_idx"]
    else:
        train_args = args[0]
        train_graph = args[2]
        train_idx = args[5]

    _initialise_state(train_graph, train_idx, torch.device(train_args.device))
    result = _ORIGINAL_TRAIN(*args, **kwargs)

    metrics = result[1]
    metrics.update({
        "variant_name": "EC-RML-Safe",
        "ec_rml_variant": "EC-RML-Safe",
        "ec_rml_principle": "Decouple noisy-label candidate detection from unreliable structural evidence routing.",
        "ec_rml_reliability_formula": "0.50*mean_plddt + 0.30*q20_plddt + 0.20*(1-low_plddt_ratio)",
        "ec_rml_r_low_quantile": float(CFG["r_low_q"]),
        "ec_rml_r_high_quantile": float(CFG["r_high_q"]),
        "ec_rml_r_low_value": float(STATE["r_low"]),
        "ec_rml_r_high_value": float(STATE["r_high"]),
        "ec_rml_low_threshold": float(CFG["low_threshold"]),
        "ec_rml_high_threshold_from_args": float(getattr(train_args, "threshold", float("nan"))),
        "ec_rml_eta_max": float(CFG["eta_max"]),
        "ec_rml_seq_conf_min": float(CFG["seq_conf_min"]),
        "ec_rml_nbr_conf_min": float(CFG["nbr_conf_min"]),
        "ec_rml_pseudo_conf_min": float(CFG["pseudo_conf_min"]),
        "ec_rml_margin_min": float(CFG["margin_min"]),
        "ec_rml_min_eta": float(CFG["min_eta"]),
        "ec_rml_ema_alpha": float(CFG["ema_alpha"]),
        "ec_rml_w_seq": float(CFG["w_seq"]),
        "ec_rml_w_nbr": float(CFG["w_nbr"]),
        "ec_rml_w_ema": float(CFG["w_ema"]),
        "ec_rml_w_struct": float(CFG["w_struct"]),
        "ec_rml_final_prediction_uses_original_sharedt_fusion": True,
        "ec_rml_external_confidence_affects_candidate_detection": False,
        "ec_rml_external_confidence_affects_pseudolabel_routing": True,
    })
    return result

train_dual_tmnr2 = _patched_train

_ORIGINAL_SAVE = save_outputs

def _patched_save_outputs(*args, **kwargs):
    _ORIGINAL_SAVE(*args, **kwargs)

    train_args = args[0] if args else kwargs["args"]
    info_fp = Path(train_args.result_dir) / "train_dual_tmnr2_info.csv"
    if not info_fp.exists() or not STATE:
        return

    r = STATE["r"].detach().cpu().numpy()
    rho = STATE["rho"].detach().cpu().numpy()
    score = STATE["last_score"].detach().cpu().numpy()
    candidate = STATE["last_candidate"].detach().cpu().numpy().astype(int)
    changed = STATE["last_changed"].detach().cpu().numpy().astype(int)
    high = STATE["last_high_candidate"].detach().cpu().numpy().astype(int)
    border = STATE["last_border_candidate"].detach().cpu().numpy().astype(int)
    border_pass = STATE["last_border_pass"].detach().cpu().numpy().astype(int)
    eta = STATE["last_eta"].detach().cpu().numpy()
    region_code = STATE["last_region_code"].detach().cpu().numpy().astype(int)

    with open(info_fp, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    extra = [
        "ec_rml_struct_reliability", "ec_rml_struct_rho", "ec_rml_region",
        "ec_rml_score_last_round", "ec_rml_candidate_last_round",
        "ec_rml_changed_last_round", "ec_rml_high_candidate_last_round",
        "ec_rml_border_candidate_last_round", "ec_rml_border_pass_last_round",
        "ec_rml_eta_last_round",
    ]
    for col in extra:
        if col not in fieldnames:
            fieldnames.append(col)

    code_to_region = {0: "none", 1: "high", 2: "border_diagnostic", 3: "blocked_high"}
    for i, row in enumerate(rows):
        row.update({
            "ec_rml_struct_reliability": f"{float(r[i]):.8f}",
            "ec_rml_struct_rho": f"{float(rho[i]):.8f}",
            "ec_rml_region": code_to_region.get(int(region_code[i]), "none"),
            "ec_rml_score_last_round": "" if not np.isfinite(score[i]) else f"{float(score[i]):.8f}",
            "ec_rml_candidate_last_round": int(candidate[i]),
            "ec_rml_changed_last_round": int(changed[i]),
            "ec_rml_high_candidate_last_round": int(high[i]),
            "ec_rml_border_candidate_last_round": int(border[i]),
            "ec_rml_border_pass_last_round": int(border_pass[i]),
            "ec_rml_eta_last_round": f"{float(eta[i]):.8f}",
        })

    with open(info_fp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

save_outputs = _patched_save_outputs

# =============================================================================
# Train-only entry point
# =============================================================================

def assert_alignment(sequence_data: Mapping[str, Any], graph_data: Mapping[str, Any], name: str) -> None:
    """Check sample count, order, sequence and labels across both views."""
    seq_count = len(sequence_data["labels"])
    graph_count = len(graph_data["labels"])
    if seq_count != graph_count:
        raise RuntimeError(f"{name}: seq/graph length mismatch: {seq_count} vs {graph_count}")

    for index in range(seq_count):
        if norm_seq(sequence_data["seqs"][index]) != norm_seq(graph_data["seqs"][index]):
            raise RuntimeError(
                f"{name}: sequence mismatch at index={index}: "
                f"{sequence_data['seqs'][index]!r} vs {graph_data['seqs'][index]!r}"
            )
        if int(sequence_data["labels"][index]) != int(graph_data["labels"][index]):
            raise RuntimeError(
                f"{name}: label mismatch at index={index}: "
                f"{int(sequence_data['labels'][index])} vs {int(graph_data['labels'][index])}"
            )
    print(f"✅ {name} seq/graph alignment checked: {seq_count} samples")

def subset_feature_data(
    sequence_data: Mapping[str, Any],
    graph_data: Mapping[str, Any],
    indices: Sequence[int],
    labels_override: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create aligned feature subsets without recomputing ESM2 or PDB graphs."""
    selected = [int(index) for index in indices]
    if labels_override is None:
        labels = sequence_data["labels"][selected].clone().long()
    else:
        override = np.asarray(labels_override, dtype=np.int64)
        if len(override) != len(selected):
            raise ValueError(
                f"labels_override length mismatch: {len(override)} vs {len(selected)}"
            )
        labels = torch.as_tensor(override, dtype=torch.long)

    sequence_subset = {
        "ids": [sequence_data["ids"][index] for index in selected],
        "seqs": [sequence_data["seqs"][index] for index in selected],
        "labels": labels.clone(),
        "embeddings": [sequence_data["embeddings"][index] for index in selected],
    }
    graph_subset = {
        "ids": [graph_data["ids"][index] for index in selected],
        "seqs": [graph_data["seqs"][index] for index in selected],
        "labels": labels.clone(),
        "graphs": [graph_data["graphs"][index] for index in selected],
    }
    return sequence_subset, graph_subset

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

def rename_training_outputs(result_dir: Path) -> Dict[str, Path]:
    """Rename legacy output names to explicit training/validation names."""
    mapping = {
        "metrics_dual_tmnr2.json": "val_metrics.json",
        "test_predictions_dual_tmnr2.csv": "val_predictions.csv",
        "best_dual_tmnr2.pt": "best_model.pt",
        "train_history_dual_tmnr2.csv": "train_history.csv",
        "train_dual_tmnr2_info.csv": "train_info.csv",
    }
    outputs: Dict[str, Path] = {}
    for old_name, new_name in mapping.items():
        old_path = result_dir / old_name
        new_path = result_dir / new_name
        if not old_path.exists():
            raise FileNotFoundError(f"Expected training output was not created: {old_path}")
        os.replace(old_path, new_path)
        outputs[new_name] = new_path
    return outputs

def save_run_config(
    args: argparse.Namespace,
    output_path: Path,
    sequence_input_dim: int,
    graph_input_dim: int,
    train_size: int,
    validation_size: int,
) -> None:
    public_args = {
        key: _json_ready(value)
        for key, value in vars(args).items()
        if not key.startswith("_")
    }
    config: Dict[str, Any] = {
        "format_version": 2,
        "model": "RISE",
        "algorithm": "EC-RML-Safe",
        "sequence_input_dim": int(sequence_input_dim),
        "graph_input_dim": int(graph_input_dim),
        "train_size": int(train_size),
        "validation_size": int(validation_size),
        "arguments": public_args,
        "safe_configuration": _json_ready(CFG),
    }
    output_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML configuration not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {yaml_path}")
    return data


def _nested_get(mapping: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
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
        f"Unsupported dataset {name!r}. Expected XUAMP or GenPept-Curated-2025."
    )


def _format_noise(noise: float) -> str:
    return f"{float(noise):.3f}"


def _expand_template(value: Any, context: Mapping[str, Any]) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    try:
        return text.format(**context)
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder {exc} in path template: {text}") from exc


def resolve_experiment_paths(
    experiment_config: Mapping[str, Any],
    paths_config: Mapping[str, Any],
    dataset_override: Optional[str],
    noise: float,
    rep: int,
) -> Dict[str, str]:
    dataset_name = dataset_override or _nested_get(
        experiment_config, ("experiment", "dataset")
    )
    if not dataset_name:
        raise ValueError("Dataset is missing from both CLI and experiment config.")
    key = _dataset_key(str(dataset_name))

    dataset_paths = _nested_get(paths_config, ("datasets", key))
    if not isinstance(dataset_paths, Mapping):
        raise ValueError(f"Missing datasets.{key} section in paths YAML.")

    root = str(dataset_paths.get("root", ""))
    noise_text = _format_noise(noise)
    base_context: Dict[str, Any] = {
        "root": root,
        "noise": noise_text,
        "rep": int(rep),
    }
    run_dir = _expand_template(dataset_paths.get("run_dir_template"), base_context)
    if not run_dir:
        raise ValueError(f"datasets.{key}.run_dir_template is required.")
    context = dict(base_context)
    context["run_dir"] = run_dir

    resolved: Dict[str, str] = {
        "dataset_key": key,
        "dataset_name": str(dataset_name),
        "noise_text": noise_text,
        "run_dir": run_dir,
    }
    for field in (
        "train_csv", "val_csv", "test_csv",
        "train_amp", "train_nonamp", "val_amp", "val_nonamp",
        "test_amp", "test_nonamp",
    ):
        template_key = f"{field}_template"
        value = _expand_template(dataset_paths.get(template_key), context)
        if value:
            resolved[field] = value

    for field in ("pdb_dir", "clean_ref_amp", "clean_ref_nonamp"):
        value = _expand_template(dataset_paths.get(field), context)
        if value:
            resolved[field] = value

    runtime = paths_config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime section in paths YAML must be a mapping.")
    for field in ("device", "esm_model_path", "output_root", "cache_root"):
        value = runtime.get(field)
        if value is not None:
            resolved[field] = str(value)

    output_root = Path(resolved.get("output_root", "./outputs")).expanduser()
    cache_root = Path(resolved.get("cache_root", "./cache")).expanduser()
    resolved["result_dir"] = str(
        output_root / key / f"noise_{noise_text}" / f"rep{int(rep)}"
    )
    resolved["cache_dir"] = str(
        cache_root / key / f"noise_{noise_text}" / f"rep{int(rep)}"
    )
    return resolved


_CONFIG_ARGUMENT_MAP: Dict[str, Tuple[str, ...]] = {
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
    "lambda_sim": ("loss", "lambda_sim"),
    "lambda_mse": ("loss", "lambda_mse"),
    "lambda_mix": ("loss", "lambda_mix"),
    "batch_size": ("training", "batch_size"),
    "eval_batch_size": ("training", "eval_batch_size"),
    "num_workers": ("training", "num_workers"),
    "epochs": ("training", "epochs"),
    "warmup_epochs": ("training", "warmup_epochs"),
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

_SAFE_CONFIG_MAP: Dict[str, Tuple[str, ...]] = {
    "low_threshold": ("safe", "low_threshold"),
    "r_low_q": ("safe", "reliability_low_quantile"),
    "r_high_q": ("safe", "reliability_high_quantile"),
    "eta_max": ("safe", "eta_max"),
    "seq_conf_min": ("safe", "sequence_confidence_min"),
    "nbr_conf_min": ("safe", "neighbor_confidence_min"),
    "pseudo_conf_min": ("safe", "pseudo_label_confidence_min"),
    "margin_min": ("safe", "margin_min"),
    "min_eta": ("safe", "eta_min"),
    "ema_alpha": ("safe", "ema_alpha"),
    "ema_conf_min": ("safe", "ema_confidence_min"),
    "ema_stability_floor": ("safe", "ema_stability_floor"),
    "w_seq": ("safe", "weight_sequence"),
    "w_nbr": ("safe", "weight_neighbor"),
    "w_ema": ("safe", "weight_ema"),
    "w_struct": ("safe", "weight_structure"),
    "reliable_conflict_lambda": ("safe", "reliable_conflict_lambda"),
    "conflict_floor": ("safe", "conflict_floor"),
}


def configure_safe(experiment_config: Mapping[str, Any]) -> None:
    for key, path in _SAFE_CONFIG_MAP.items():
        value = _nested_get(experiment_config, path)
        if value is not None:
            CFG[key] = float(value)

    if not (0.0 <= CFG["r_low_q"] < CFG["r_high_q"] <= 1.0):
        raise ValueError("Require 0 <= reliability_low_quantile < reliability_high_quantile <= 1.")
    if not (0.0 <= CFG["low_threshold"] <= 1.0):
        raise ValueError("safe.low_threshold must be in [0, 1].")
    if not (0.0 <= CFG["eta_max"] <= 1.0):
        raise ValueError("safe.eta_max must be in [0, 1].")
    weights = [CFG["w_seq"], CFG["w_nbr"], CFG["w_ema"], CFG["w_struct"]]
    if min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("Safe pseudo-label weights must be non-negative and sum to > 0.")


def _set_from_config(
    cli: argparse.Namespace,
    experiment_config: Mapping[str, Any],
    resolved_paths: Mapping[str, str],
) -> argparse.Namespace:
    values = vars(cli).copy()
    for attr, path in _CONFIG_ARGUMENT_MAP.items():
        if values.get(attr) is None:
            config_value = _nested_get(experiment_config, path)
            if config_value is not None:
                values[attr] = config_value

    for attr in (
        "device", "esm_model_path", "pdb_dir",
        "train_csv", "val_csv",
        "train_amp", "train_nonamp", "val_amp", "val_nonamp",
        "clean_ref_amp", "clean_ref_nonamp",
        "result_dir", "cache_dir",
    ):
        if values.get(attr) is None and resolved_paths.get(attr) is not None:
            values[attr] = resolved_paths[attr]

    values["dataset"] = resolved_paths["dataset_key"]
    values["dataset_name"] = resolved_paths["dataset_name"]
    values["noise_rate"] = float(cli.noise)
    values["noise"] = float(cli.noise)
    values["rep"] = int(cli.rep)
    values["validation_mode"] = "predefined"
    return argparse.Namespace(**values)


def _require_paths(args: argparse.Namespace) -> None:
    csv_fields = ("train_csv", "val_csv")
    fasta_fields = ("train_amp", "train_nonamp", "val_amp", "val_nonamp")

    has_any_csv = any(getattr(args, field, None) for field in csv_fields)
    has_all_csv = all(getattr(args, field, None) for field in csv_fields)
    has_any_fasta = any(getattr(args, field, None) for field in fasta_fields)
    has_all_fasta = all(getattr(args, field, None) for field in fasta_fields)

    if has_all_csv:
        args.input_format = "csv"
        selected_fields = csv_fields
    elif has_all_fasta and not has_any_csv:
        args.input_format = "fasta"
        selected_fields = fasta_fields
    else:
        raise ValueError(
            "Configure either train_csv + val_csv, or all four FASTA paths "
            "(train_amp, train_nonamp, val_amp, val_nonamp)."
        )

    for field in selected_fields:
        path = Path(getattr(args, field)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{field} not found: {path}")
        setattr(args, field, str(path.resolve()))

    # CSV files already contain the controlled noisy labels. Never inject a
    # second round of synthetic noise into them.
    if args.input_format == "csv":
        if getattr(args, "noise_source", "file") == "internal":
            raise ValueError(
                "noise_source=internal is incompatible with predefined CSV "
                "noise splits. Use noise_source=file."
            )
        args.noise_source = "file"

    for field in ("esm_model_path", "pdb_dir"):
        value = getattr(args, field, None)
        if not value:
            raise ValueError(f"Missing required path: {field}")
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{field} not found: {path}")
        setattr(args, field, str(path.resolve()))

    for field in ("clean_ref_amp", "clean_ref_nonamp"):
        value = getattr(args, field, None)
        if value:
            path = Path(value).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"{field} not found: {path}")
            setattr(args, field, str(path.resolve()))


def _combine_feature_data(
    train_sequence: Mapping[str, Any],
    train_graph: Mapping[str, Any],
    validation_sequence: Mapping[str, Any],
    validation_graph: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    train_count = len(train_sequence["labels"])
    validation_count = len(validation_sequence["labels"])

    sequence_data: Dict[str, Any] = {
        "ids": list(train_sequence["ids"]) + list(validation_sequence["ids"]),
        "seqs": list(train_sequence["seqs"]) + list(validation_sequence["seqs"]),
        "labels": torch.cat(
            [train_sequence["labels"].long(), validation_sequence["labels"].long()],
            dim=0,
        ),
        "embeddings": list(train_sequence["embeddings"])
        + list(validation_sequence["embeddings"]),
    }
    graph_data: Dict[str, Any] = {
        "ids": list(train_graph["ids"]) + list(validation_graph["ids"]),
        "seqs": list(train_graph["seqs"]) + list(validation_graph["seqs"]),
        "labels": torch.cat(
            [train_graph["labels"].long(), validation_graph["labels"].long()],
            dim=0,
        ),
        "graphs": list(train_graph["graphs"]) + list(validation_graph["graphs"]),
    }
    train_idx = np.arange(train_count, dtype=np.int64)
    val_idx = np.arange(train_count, train_count + validation_count, dtype=np.int64)
    return sequence_data, graph_data, train_idx, val_idx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train RISE using predefined train and validation splits."
    )
    parser.add_argument("--config", required=True, help="Dataset experiment YAML.")
    parser.add_argument("--paths", required=True, help="Machine-specific paths YAML.")
    parser.add_argument("--dataset", default=None, help="Optional dataset override.")
    parser.add_argument("--noise", required=True, type=float)
    parser.add_argument("--rep", required=True, type=int)

    # Optional direct path overrides. CSV is the public benchmark format;
    # FASTA arguments are retained for backward compatibility.
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--train_amp", default=None)
    parser.add_argument("--train_nonamp", default=None)
    parser.add_argument("--val_amp", default=None)
    parser.add_argument("--val_nonamp", default=None)
    parser.add_argument("--result_dir", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--esm_model_path", default=None)
    parser.add_argument("--pdb_dir", default=None)
    parser.add_argument("--pdb_map_csv", default=None)
    parser.add_argument("--clean_ref_amp", default=None)
    parser.add_argument("--clean_ref_nonamp", default=None)

    parser.add_argument("--esm_batch_size", type=int, default=None)
    parser.add_argument("--contact_threshold", type=float, default=None)
    parser.add_argument("--graph_knn_k", type=int, default=None)
    parser.add_argument("--plddt_scale", choices=["auto", "100", "1"], default=None)
    parser.add_argument("--index_pdb_by_seq", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow_missing_pdb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--disable_plddt_feature", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_aa_onehot", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--delete_cache_after_run", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--lstm_layers", type=int, default=None)
    parser.add_argument("--gnn_hidden_dim", type=int, default=None)
    parser.add_argument("--gnn_layers", type=int, default=None)
    parser.add_argument("--classifier_hidden", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--fusion_type", choices=["ds", "sum", "mean"], default=None)
    parser.add_argument("--knn_feature", choices=["dual", "seq", "struct"], default=None)

    for name in (
        "lambda_aux", "lambda_t_consistency", "lambda_conf", "lambda_sim",
        "lambda_mse", "lambda_mix", "lr", "t_lr", "weight_decay",
        "clip_grad_norm", "min_delta", "threshold", "threshold_decay",
        "min_threshold", "mixup_alpha",
    ):
        parser.add_argument(f"--{name}", type=float, default=None)

    for name in (
        "epochs", "warmup_epochs", "start_correct", "correct_every", "patience",
        "min_best_epoch", "annealing_epoch", "knn_k", "seed", "log_every",
        "log_every_extract",
    ):
        parser.add_argument(f"--{name}", type=int, default=None)

    parser.add_argument("--monitor", choices=["val_auc", "val_f1", "val_acc"], default=None)
    parser.add_argument("--noise_source", choices=["auto", "file", "internal"], default=None)
    parser.add_argument("--best_after_start_correct", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--keep_soft_label", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_mixup", action=argparse.BooleanOptionalAction, default=None)
    return parser


def resolve_training_args(cli: argparse.Namespace) -> argparse.Namespace:
    experiment_config = _load_yaml(cli.config)
    paths_config = _load_yaml(cli.paths)
    resolved_paths = resolve_experiment_paths(
        experiment_config=experiment_config,
        paths_config=paths_config,
        dataset_override=cli.dataset,
        noise=cli.noise,
        rep=cli.rep,
    )
    args = _set_from_config(cli, experiment_config, resolved_paths)
    configure_safe(experiment_config)

    # Defaults only apply when neither YAML nor CLI specifies the value.
    defaults: Dict[str, Any] = {
        "device": "cuda:0", "esm_batch_size": 8,
        "pdb_map_csv": None, "index_pdb_by_seq": False,
        "allow_missing_pdb": False, "contact_threshold": 8.0,
        "graph_knn_k": 8, "plddt_scale": "auto",
        "disable_plddt_feature": False, "use_aa_onehot": False,
        "delete_cache_after_run": False, "batch_size": 16,
        "eval_batch_size": 64, "num_workers": 0, "hidden_dim": 128,
        "lstm_layers": 1, "gnn_hidden_dim": 128, "gnn_layers": 3,
        "classifier_hidden": 128, "dropout": 0.3, "fusion_type": "ds",
        "knn_feature": "dual", "lambda_aux": 0.05,
        "lambda_t_consistency": 0.05, "epochs": 100,
        "warmup_epochs": 10, "start_correct": 5, "correct_every": 5,
        "patience": 40, "monitor": "val_f1", "min_delta": 1e-4,
        "min_best_epoch": 1, "best_after_start_correct": True,
        "lr": 3e-4, "t_lr": 1e-3, "weight_decay": 1e-4,
        "clip_grad_norm": 1.0, "annealing_epoch": 80, "knn_k": 20,
        "lambda_conf": 0.01, "lambda_sim": 0.0, "lambda_mse": 1.0,
        "lambda_mix": 1.0, "threshold": 0.80, "threshold_decay": 1.0,
        "min_threshold": 0.60, "keep_soft_label": True,
        "use_mixup": True, "mixup_alpha": 0.3, "noise_source": "file",
        "seed": 42, "log_every": 5, "log_every_extract": 500,
    }
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    args.config = str(Path(cli.config).expanduser().resolve())
    args.paths = str(Path(cli.paths).expanduser().resolve())
    _require_paths(args)
    return args


def main() -> None:
    cli = build_parser().parse_args()
    args = resolve_training_args(cli)

    result_dir = Path(args.result_dir).expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir = str(result_dir)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir = str(cache_dir)

    initial_threshold = float(args.threshold)
    set_seed(args.seed)

    if abs(float(args.lambda_sim)) > 0:
        print(
            "⚠️ --lambda_sim is ignored: neighbor transition-matrix smoothing "
            "is not defined for global shared T."
        )

    clean_labels_override = None
    file_label_noise_mask_override = None

    if args.input_format == "csv":
        train_records = load_csv_split(
            args.train_csv,
            label_column="target",
            expected_split="train",
        )
        val_records = load_csv_split(
            args.val_csv,
            label_column="clean_target",
            expected_split="val",
        )
        clean_labels_override = np.asarray(
            [record["clean_target"] for record in train_records]
            + [record["clean_target"] for record in val_records],
            dtype=np.int64,
        )
        file_label_noise_mask_override = np.asarray(
            [record["is_noisy"] for record in train_records]
            + [record["is_noisy"] for record in val_records],
            dtype=bool,
        )
    else:
        train_records = load_binary_split(args.train_amp, args.train_nonamp)
        val_records = load_binary_split(args.val_amp, args.val_nonamp)

    print(
        f"✅ predefined {args.input_format} split | "
        f"train={len(train_records)} | validation={len(val_records)}"
    )

    train_sequence = extract_token_embeddings(
        records=train_records,
        esm_model_path=args.esm_model_path,
        cache_fp=str(cache_dir / "train_esm2_token.pt"),
        device=args.device,
        esm_batch_size=args.esm_batch_size,
    )
    val_sequence = extract_token_embeddings(
        records=val_records,
        esm_model_path=args.esm_model_path,
        cache_fp=str(cache_dir / "val_esm2_token.pt"),
        device=args.device,
        esm_batch_size=args.esm_batch_size,
    )
    train_graph = extract_graphs(
        train_records, str(cache_dir / "train_struct_graph.pt"), args, "train"
    )
    val_graph = extract_graphs(
        val_records, str(cache_dir / "val_struct_graph.pt"), args, "validation"
    )
    assert_alignment(train_sequence, train_graph, "train")
    assert_alignment(val_sequence, val_graph, "validation")

    sequence_data, graph_data, train_idx, val_idx = _combine_feature_data(
        train_sequence, train_graph, val_sequence, val_graph
    )
    assert_alignment(sequence_data, graph_data, "combined train/validation")

    sequence_input_dim = int(sequence_data["embeddings"][0].shape[1])
    graph_input_dim = int(graph_data["graphs"][0]["x"].shape[1])

    (
        model,
        val_metrics,
        val_prediction_rows,
        history,
        corrected_probs,
        noisy_labels,
        injected_noise_mask,
        observed_input_labels,
        file_label_noise_mask,
        returned_clean_labels,
    ) = train_dual_tmnr2(
        args=args,
        train_seq=sequence_data,
        train_graph=graph_data,
        test_seq=val_sequence,
        test_graph=val_graph,
        train_idx=train_idx,
        val_idx=val_idx,
        seq_input_dim=sequence_input_dim,
        graph_input_dim=graph_input_dim,
        clean_labels_override=clean_labels_override,
        file_label_noise_mask_override=file_label_noise_mask_override,
    )

    val_metrics.update({
        "evaluation_split": "validation",
        "validation_mode": "predefined",
        "model": "RISE",
        "algorithm": "EC-RML-Safe",
        "dataset": args.dataset_name,
        "noise": float(args.noise),
        "rep": int(args.rep),
        "threshold_initial": initial_threshold,
    })

    save_outputs(
        args=args,
        model=model,
        metrics=val_metrics,
        pred_rows=val_prediction_rows,
        history=history,
        corrected_probs=corrected_probs,
        noisy_labels=noisy_labels,
        injected_noise_mask=injected_noise_mask,
        train_seq=sequence_data,
        train_graph=graph_data,
        observed_input_labels=observed_input_labels,
        file_label_noise_mask=file_label_noise_mask,
        clean_labels_override=returned_clean_labels,
    )
    output_paths = rename_training_outputs(result_dir)

    np.savez_compressed(
        result_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
    )
    save_run_config(
        args=args,
        output_path=result_dir / "run_config.json",
        sequence_input_dim=sequence_input_dim,
        graph_input_dim=graph_input_dim,
        train_size=len(train_idx),
        validation_size=len(val_idx),
    )

    print("\n✅ Training completed")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")
    print(f"  run_config.json: {result_dir / 'run_config.json'}")

    if args.delete_cache_after_run and cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        print(f"🧹 deleted cache: {cache_dir}")


if __name__ == "__main__":
    main()
