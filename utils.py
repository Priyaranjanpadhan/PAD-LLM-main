import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from transformers import AutoTokenizer, T5EncoderModel
import os
def extract_features(
    cnn_model,
    rating_tensor,
    item_labels_tensor,
    device,
    chunk_size=256,
    to_cpu=True,
    use_amp=True,
):
    cnn_model.eval()
    rating_tensor = rating_tensor.to(device)
    item_labels_tensor = item_labels_tensor.to(device)
    fixed_bs = getattr(cnn_model, "batch_size", None)
    eff_chunk = fixed_bs if (fixed_bs is not None and fixed_bs > 0) else chunk_size
    features_list = []
    num_samples = rating_tensor.size(0)
    autocast_enabled = (use_amp and device.type == "cuda")
    with torch.no_grad():
        for start in range(0, num_samples, eff_chunk):
            chunk = rating_tensor[start:start + eff_chunk]
            b = chunk.size(0)
            if fixed_bs is not None and b < fixed_bs:
                pad = chunk[-1:].repeat(fixed_bs - b, 1)
                chunk_in = torch.cat([chunk, pad], dim=0)
            else:
                chunk_in = chunk
            with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
                if hasattr(cnn_model, "forward_features"):
                    out = cnn_model.forward_features(chunk_in, item_labels_tensor)
                else:
                    out = cnn_model(chunk_in, item_labels_tensor)
            if fixed_bs is not None and b < fixed_bs:
                out = out[:b]
            out = out.detach().float()
            if to_cpu:
                out = out.cpu()
            features_list.append(out)
    features = torch.cat(features_list, dim=0)
    return features

def load_item_label_data(features_file, num_items,
                         model_name="t5-small",
                         out_dim=16,
                         batch_size=64,
                         max_length=64,
                         device=None):
    # 1. Disk cache check(loads in miliseconds on every future run)
    cache_path = f"{features_file}_t5_svd_{out_dim}d.pt"
    if os.path.exists(cache_path):
        print(f"[Cache Hit] Loading pre-computed T5 SVD item embeddings from {cache_path}")
        return torch.load(cache_path, weights_only=True).numpy()

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    df = pd.read_csv(
        features_file,
        delimiter="::",
        engine="python",
        header=None,
        usecols=[0, 2],
        encoding="ISO-8859-1",
        names=["item_id", "features"],
    )
    df["item_id"] = df["item_id"].astype(int)
    texts = df["features"].fillna("").astype(str).str.replace("|", ", ", regex=False).tolist()
    item_ids = df["item_id"].to_numpy()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval()
    with torch.no_grad():
        dummy = tokenizer(["test"], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        dummy = {k: v.to(device) for k, v in dummy.items()}
        out = model(**dummy)
        hidden_dim = out.last_hidden_state.shape[-1]
    # nn.linear(512, 16) Creates a matrix of untrained, random numbers. Multiplying rich 512 dimensional T5 language vectors by a random matrix destroys the semantic relationships. It reruns the T5 neural network from scratch every single we start the main file.
    # nn.linear(): creates a standard neural network layer that applies a mathematical matrix multplication to transform the 512 dimensional T5 vectors into 16 dimensional vectors.
    # proj = nn.Linear(hidden_dim, out_dim, bias=False).to(device)
    # .eval(): Sets the layer or model into evaluation mode, which disables training specific behaviours like driput to ensure the outputs are consistent every time.
    # proj.eval()
    # =============================HOW TO FIX=============================================
    # Collect raw 512D T5 embeddings first(No random linear layer!)
    raw_E = torch.zeros((num_items, hidden_dim), dtype=torch.float32, device="cpu")
    req_rows, req_texts = [],[]
    for iid, t in zip(item_ids, texts):
        row = iid - 1
        if 0 <= row < num_items and t.strip():
            req_rows.append(row)
            req_texts.append(t.strip())

    if not req_texts:
        return np.zeros((num_items, out_dim), dtype=np.float32)

    for s in range(0, len(req_texts), batch_size):
        batch_texts = req_texts[s:s + batch_size]
        batch_rows  = req_rows[s:s + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            raw_E[batch_rows] = pooled.detach().cpu()

    with torch.no_grad():
        mean_vec = raw_E[req_rows].mean(dim = 0, keepdim=True)
        centered = raw_E - mean_vec
        _, _, Vh = torch.linalg.svd(centered[req_rows], full_matrices=False)
        E_tensor = torch.matmul(centered, Vh[:out_dim].T)
        E_tensor = torch.nn.functional.normalize(E_tensor, p=2, dim=-1)

    torch.save(E_tensor, cache_path)
    print(f"[Cache Saved] Saved deterministic T5 SVD embeddings to {cache_path}")
    return E_tensor.numpy()


def amazon_item(
    category_file_path,
    brand_file_path,
    num_items=2752,
    model_name="t5-small",
    out_dim=16,
    batch_size=64,
    max_length=64,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    item_to_categories = defaultdict(list)
    with open(category_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            item_str, cat_str = parts
            try:
                item_id = int(item_str)
                cat_id = int(cat_str)
            except ValueError:
                continue
            item_to_categories[item_id].append(cat_id)
    for item_id in item_to_categories:
        item_to_categories[item_id] = list(dict.fromkeys(item_to_categories[item_id]))
    item_to_brand = {}
    with open(brand_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            item_str, brand_str = parts
            try:
                item_id = int(item_str)
                brand_id = int(brand_str)
            except ValueError:
                continue
            item_to_brand[item_id] = brand_id
    all_item_ids = sorted(set(item_to_categories.keys()) | set(item_to_brand.keys()))
    item_texts = {}
    for item_id in all_item_ids:
        parts = []
        if item_id in item_to_categories:
            for cat_id in item_to_categories[item_id]:
                parts.append(f"category_{cat_id}")

        if item_id in item_to_brand:
            parts.append(f"brand_{item_to_brand[item_id]}")
        text = ", ".join(parts).strip()
        if text:
            item_texts[item_id] = text
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = T5EncoderModel.from_pretrained(model_name).to(device)
    encoder.eval()
    with torch.no_grad():
        dummy = tokenizer(
            ["test"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        dummy = {k: v.to(device) for k, v in dummy.items()}
        hidden_dim = encoder(**dummy).last_hidden_state.shape[-1]
    proj = nn.Linear(hidden_dim, out_dim, bias=False).to(device)
    proj.eval()
    E = np.zeros((num_items, out_dim), dtype=np.float32)
    if not item_texts:
        return E
    item_ids = sorted(item_texts.keys())
    text_list = [item_texts[i] for i in item_ids]
    for s in range(0, len(text_list), batch_size):
        batch_ids = item_ids[s:s + batch_size]
        batch_texts = text_list[s:s + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = encoder(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            vecs = proj(pooled)
        vecs = vecs.detach().cpu().numpy().astype(np.float32)
        for item_id, vec in zip(batch_ids, vecs):
            if 0 <= item_id < num_items:
                E[item_id] = vec
    return E