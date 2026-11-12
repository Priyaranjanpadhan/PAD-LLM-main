import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from matplotlib.font_manager import FontProperties

def load_item_label_data(features_file, num_items):
    features_data = pd.read_csv(
        features_file, delimiter='::', engine='python', header=None,
        usecols=[0, 2], encoding='ISO-8859-1',
        names=['item_id', 'features']
    )
    features_data['item_id'] = features_data['item_id'].astype(int)
    all_features = set()
    for feat in features_data['features']:
        if pd.notnull(feat):
            all_features.update(feat.split('|'))
    feature_names = sorted(all_features)
    num_features = len(feature_names)
    tag_names_col = np.array(feature_names, dtype=object).reshape(-1, 1)
    encoded_features_matrix = np.zeros((num_items, num_features), dtype=np.int8)
    feature_to_idx = {f: i for i, f in enumerate(feature_names)}
    for _, row in features_data.iterrows():
        item_id_index = int(row['item_id']) - 1  # item_id 从 1 开始
        if 0 <= item_id_index < num_items and pd.notnull(row['features']):
            for f in row['features'].split('|'):
                j = feature_to_idx.get(f, None)
                if j is not None:
                    encoded_features_matrix[item_id_index, j] = 1
    return encoded_features_matrix, tag_names_col

def _random_pool_by_class(y_np, n0_pool=1200, n1_pool=1200, seed=42):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y_np == 0)[0]
    idx1 = np.where(y_np == 1)[0]
    p0 = rng.choice(idx0, size=min(n0_pool, len(idx0)), replace=False) if len(idx0) else np.array([], dtype=int)
    p1 = rng.choice(idx1, size=min(n1_pool, len(idx1)), replace=False) if len(idx1) else np.array([], dtype=int)
    return p0, p1

class _SHAP2DWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, L: torch.Tensor, force_logit: bool = True, debug_once: bool = True):
        super().__init__()
        self.base = base_model
        self.register_buffer("L", L)
        self.force_logit = force_logit
        self._debug_once = debug_once
        self._printed = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            out = self.base(x, self.L)
        except TypeError:
            out = self.base(x)

        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 2 and out.size(1) == 1:
            out = out[:, 0]
        elif out.dim() == 2 and out.size(1) > 1:
            out = out[:, 1]
        elif out.dim() > 2:
            out = out.view(out.size(0), -1)[:, 0]
        if self.force_logit:
            mn = float(out.min().detach().cpu())
            mx = float(out.max().detach().cpu())
            if mn >= -1e-4 and mx <= 1.0001:
                eps = 1e-6
                out = torch.clamp(out, eps, 1 - eps)
                out = torch.log(out / (1 - out))

        out2 = out.reshape(-1, 1)

        if self._debug_once and (not self._printed):
            self._printed = True
        assert out2.dim() == 2 and out2.size(1) == 1, f"wrapper must return [B,1], got {tuple(out2.shape)}"
        return out2

@torch.no_grad()
def _predict_logits(wrapper: torch.nn.Module, X: torch.Tensor, batch_size: int = 256):
    wrapper.eval()
    outs = []
    for i in range(0, X.size(0), batch_size):
        outs.append(wrapper(X[i:i + batch_size]).detach().cpu())
    out = torch.cat(outs, dim=0)
    return out[:, 0].numpy()

def shap_tag_heatmap_20users_from_cnn(
    cnn_model,
    rating_tensor,
    item_labels_tensor,
    y_tensor,
    bg_size=64,
    seed=42,
    tag="Figure23",
    tag_names=None,
    pick_mode="typical_by_logit",
    n_total=20,
    n0_pool=1200, n1_pool=1200,
    aggregate_only_rated=True,
    normalize_per_user=False,
    vmax_percentile=97,
    force_logit=True,
):
    device = next(cnn_model.parameters()).device
    cnn_model.eval()

    R_all = rating_tensor.detach().to(device).float()
    L = item_labels_tensor.detach().to(device).float()
    y_np = y_tensor.detach().cpu().numpy().reshape(-1)
    N, I = R_all.shape
    K = L.shape[1]
    if tag_names is None:
        x_names = [f"label {j+1}" for j in range(K)]
    else:
        x_names = [str(t).replace("\\", "") for t in tag_names]

    wrapped = _SHAP2DWrapper(cnn_model, L, force_logit=force_logit, debug_once=True).to(device)
    wrapped.eval()

    rng = np.random.default_rng(seed)
    if pick_mode == "random_balanced":
        pool0, pool1 = _random_pool_by_class(y_np, n0_pool=n0_pool, n1_pool=n1_pool, seed=seed)
        n0 = n_total // 2
        n1 = n_total - n0
        users0 = pool0[:min(n0, len(pool0))]
        users1 = pool1[:min(n1, len(pool1))]
    elif pick_mode == "typical_by_logit":
        pool0, pool1 = _random_pool_by_class(y_np, n0_pool=n0_pool, n1_pool=n1_pool, seed=seed)
        logits0 = _predict_logits(wrapped, R_all[pool0]) if len(pool0) else np.array([])
        logits1 = _predict_logits(wrapped, R_all[pool1]) if len(pool1) else np.array([])
        n0 = n_total // 2
        n1 = n_total - n0
        users0 = pool0[np.argsort(logits0)[:min(n0, len(pool0))]] if len(pool0) else np.array([], dtype=int)
        users1 = pool1[np.argsort(logits1)[-min(n1, len(pool1)) :]] if len(pool1) else np.array([], dtype=int)
    else:
        raise ValueError()

    users = np.concatenate([users0, users1]).astype(int)
    labels = np.array([0]*len(users0) + [1]*len(users1), dtype=int)

    if np.any(y_np == 0):
        bg_candidates = np.where(y_np == 0)[0]
    else:
        bg_candidates = np.arange(N)
    bg_idx = rng.choice(bg_candidates, size=min(bg_size, len(bg_candidates)), replace=False)

    X_bg = R_all[bg_idx].clone().detach().requires_grad_(True)
    X_ex = R_all[users].clone().detach().requires_grad_(True)

    with torch.enable_grad():
        explainer = shap.GradientExplainer(wrapped, X_bg)
        shap_vals = explainer.shap_values(X_ex)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals = np.array(shap_vals)

    if shap_vals.ndim == 3:
        shap_item = shap_vals[:, :, 0]
    else:
        shap_item = shap_vals

    if aggregate_only_rated:
        X_ex_np = X_ex.detach().cpu().numpy()
        rated_mask = (X_ex_np > 0).astype(np.float32)
        shap_item = shap_item * rated_mask

    L_np = L.detach().cpu().numpy().astype(np.float32)
    shap_tag = shap_item @ L_np

    if normalize_per_user:
        denom = np.sum(np.abs(shap_tag), axis=1, keepdims=True) + 1e-12
        shap_tag = shap_tag / denom
    vmax = np.percentile(np.abs(shap_tag), vmax_percentile)
    vmin = -vmax
    ytick_fs = 15
    fp_en = FontProperties(family="Times New Roman", size=ytick_fs)
    fig_h = 0.42 * len(users) + 2.2
    fig_w = 12.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    im = ax.imshow(shap_tag, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    fp_cb = FontProperties(family="Times New Roman", size=16)
    for t in cbar.ax.get_yticklabels():
        t.set_fontproperties(fp_cb)
    cbar.ax.tick_params(labelsize=16, length=0)
    fp_cn = FontProperties(family="Times New Roman", size=23)
    cbar.set_label("SHAP value", fontproperties=fp_cn, rotation=90, labelpad=13, fontsize=23)
    cbar.ax.tick_params(length=0)
    cbar.outline.set_visible(False)
    ax.set_xlabel("Item label", fontproperties=fp_cn, labelpad=8, fontsize=23)
    ax.set_ylabel("User sample", fontproperties=fp_cn, rotation=90, labelpad=10, fontsize=23)
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(x_names, fontsize=18, fontproperties=fp_en, rotation=45, ha="right")
    ytick = [f"{int(u)}({'malicious' if labels[i] == 1 else 'normal'})" for i, u in enumerate(users)]
    ax.set_yticks(np.arange(len(users)))
    ax.set_yticklabels(ytick, fontproperties=fp_cn, fontsize=15)
    for sp in ["top", "right", "left", "bottom"]:
        ax.spines[sp].set_visible(True)
    ax.tick_params(axis="both", which="both", length=0)
    plt.tight_layout()
    out_png = os.path.join(f"{tag}.png")
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_png)
