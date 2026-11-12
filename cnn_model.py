from performer_pytorch import Performer
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from contextlib import nullcontext
import os

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_random_seed(42)

class DynamicWeightedFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(DynamicWeightedFocalLoss, self).__init__()
        self.gamma = gamma
    # Problem with the below code
    # let's say there is a very less malicious users so real users are almost zeros and the average of them will be zeros in the end. and when we multiply them with focal it will be zero 
    # and the 1% of malicious gets the free tickets that's the wrong thing about this code
    # def forward(self, inputs, targets):
    #     targets = targets.float()
    #     alpha = targets.mean()
    #     bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    #     pt = torch.exp(-bce)
    #     focal = alpha * (1 - pt) ** self.gamma * bce
    #     return focal.mean()
    # ============================== Here is the FIX(section 11.1 class-balanced Focal Loss) =========================================================
    def forward(self, inputs, targets):
        # Casts integer labels (0 or 1) to decimals (0.0 or 1.0) because PyTorch loss functions require floating-point data types.
        targets = targets.float().view(-1)
        # Flattens any multi-dimensional tensor into a single 1D array to ensure predictions and targets align perfectly for math operations.
        inputs = inputs.view(-1)
        # Calculates the exponential(eˆx) to convert the negative cross-entropy loss mathematically back into the predicted probability.
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        # torch.exp(-bce): converts negative cross-entropy loss back into predicted probability of true class.
        pt = torch.exp(-bce)

        # Compute class-balanced weights that NEVER collapse to 0.0 on all-normal batches
        # .mean(): Calculates the mathematical average of all elements in the tensor
        # .detach(): Disconnects the tensor from the computational graph so PyTorch does not calculate gradients for the class weightingg mechanism during backpropagation.
        # torch.clamp(..., min, max): Restricts all values in the tensor to fall strictly between the specified minimum and maximum bounds to prevent division by zero or dead graidents.
        pos_rate = torch.clamp(targets.mean().detach(), min = 0.05, max = 0.95)
        # Assigns higher weight (1 - pos_rate) to rate malicious users (target = 1) and lower weight (pos_rate) to normal users (targets=0).
        alpha_t = targets * (1.0 - pos_rate) + (1.0 - targets) * pos_rate
        # Multiples the class weight, the focal difficulty factor (1-pt)ˆgamma, and the base cross-entropy loss.
        focal = alpha_t * ((1.0 - pt) ** self.gamma) * bce
        # .mean(): averages the loss across all the users in the batch to return a single scalar value for backpropagation.
        return focal.mean()

class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, num_neg_candidates=8, hard_neg_k=3):
        super(HardNegativeContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_neg_candidates = num_neg_candidates
        self.hard_neg_k = hard_neg_k

    def forward(self, features, labels):
        device = features.device
        labels = labels.view(-1)
        features = F.normalize(features, p=2, dim=1)
        B = features.size(0)
        losses = []
        for i in range(B):
            anchor = features[i]
            pos_idx = torch.where(labels == labels[i])[0]
            pos_idx = pos_idx[pos_idx != i]
            neg_idx = torch.where(labels != labels[i])[0]
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue
            pos_j = pos_idx[torch.randint(0, pos_idx.numel(), (1,), device=device)].item()
            positive = features[pos_j]
            m = min(self.num_neg_candidates, neg_idx.numel())
            perm = torch.randperm(neg_idx.numel(), device=device)[:m]
            neg_candidates_idx = neg_idx[perm]
            neg_candidates = features[neg_candidates_idx]
            neg_sims = torch.matmul(neg_candidates, anchor)
            k = min(self.hard_neg_k, neg_candidates.size(0))
            hard_idx = torch.topk(neg_sims, k=k, largest=True).indices
            hard_negs = neg_candidates[hard_idx]
            pos_logit = torch.matmul(anchor, positive) / self.temperature
            neg_logits = torch.matmul(hard_negs, anchor) / self.temperature
            logits = torch.cat([pos_logit.unsqueeze(0), neg_logits], dim=0).unsqueeze(0)
            targets = torch.zeros(1, dtype=torch.long, device=device)
            loss_i = F.cross_entropy(logits, targets)
            losses.append(loss_i)
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(losses).mean()


def mixup(data, targets, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(data.size(0), device=data.device)
    shuffled_data = data[rand_index]
    shuffled_targets = targets[rand_index]
    mixed_data = lam * data + (1 - lam) * shuffled_data
    mixed_targets = lam * targets + (1 - lam) * shuffled_targets
    return mixed_data, mixed_targets

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def augment_data(data, targets, noise_factor=0.1):
    noise = torch.randn_like(data) * noise_factor
    noisy_data = data + noise
    noisy_data = torch.clamp(noisy_data, 0, 1)
    augmented_data = torch.cat([data, noisy_data], dim=0)
    augmented_targets = torch.cat([targets, targets], dim=0)
    return augmented_data, augmented_targets


def generate_adversarial_samples(
    model, inputs, targets, item_labels_matrix, epsilon=0.01, alpha=0.005, num_steps=1
):
    adv_inputs = inputs.clone().detach().requires_grad_(True)
    targets = targets.view(-1).float()
    for _ in range(num_steps):
        outputs = model(adv_inputs, item_labels_matrix).view(-1)
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        model.zero_grad(set_to_none=True)
        if adv_inputs.grad is not None:
            adv_inputs.grad.zero_()
        loss.backward()
        with torch.no_grad():
            adv_inputs = adv_inputs + alpha * adv_inputs.grad.sign()
            perturbation = torch.clamp(adv_inputs - inputs, min=-epsilon, max=epsilon)
            adv_inputs = torch.clamp(inputs + perturbation, 0, 1)
        adv_inputs.requires_grad_(True)
    return adv_inputs.detach()

def _auto_gn_groups(num_channels: int, max_groups: int = 4) -> int:
    g = min(max_groups, num_channels)
    for gg in range(g, 0, -1):
        if num_channels % gg == 0:
            return gg
    return 1


class ConvNeXt3DBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: int = 2, gn_max_groups: int = 4):
        super().__init__()
        hidden = max(channels * mlp_ratio, channels)
        self.dwconv = nn.Conv3d(
            channels, channels,
            kernel_size=(3, 3, 1),
            stride=(1, 1, 1),
            padding=(1, 1, 0),
            groups=channels,
            bias=True,
        )
        self.norm = nn.GroupNorm(_auto_gn_groups(channels, gn_max_groups), channels)
        self.pw1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x + residual

class ConvNeXt3DDownsample(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mlp_ratio: int = 2,
        gn_max_groups: int = 4,
        with_block: bool = True,
    ):
        super().__init__()
        self.dw_down = nn.Conv3d(
            in_ch, in_ch,
            kernel_size=(3, 3, 1),
            stride=(2, 2, 1),
            padding=(1, 1, 0),
            groups=in_ch,
            bias=True,
        )
        self.norm = nn.GroupNorm(_auto_gn_groups(in_ch, gn_max_groups), in_ch)
        self.pw_proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=True)
        self.block = (
            ConvNeXt3DBlock(out_ch, mlp_ratio=mlp_ratio, gn_max_groups=gn_max_groups)
            if with_block else nn.Identity()
        )

    def forward(self, x):
        x = self.dw_down(x)
        x = self.norm(x)
        x = self.pw_proj(x)
        x = self.block(x)
        return x

# ===================FIX 5 : Section 7.3 Learned Attention Pooling. =========================
class AttentionPooling3D(nn.Module):
    def __init__(self, channels: int, label_dim: int):
        super().__init__()
        # Multiplies channel count (8) by downsampled label dimension(4) to get the per-item feature size (32).
        self.feat_dim = channels * label_dim
        # nn.linear(): Leans a query vector that scores each item's feature vector down to 1 importance score.
        self.query = nn.Linear(self.feat_dim, 1)

    def forward(self, x):
        # Extracts tensor dimensions: Batch(B), Channels (c = 8), Items(I=988), Label (k = 4), Depth(1).
        B, C, I, K, _ = x.shape

        # .squeeze(-1): Removes the last dimension of size 1 -> shape becomes(B, C, I, K).
        # .permute(0, 2, 1, 3): swap axes so Items (i) comes before channels (c) -> shape becomes(B, I, C, K).
        # .contiguous().view(B, I, C * k): Combines channels and labels into one feature vector per items -> shape(B, 988, 32).
        h = x.squeeze(-1).permute(0,2,1,3).contiguous().view(B, I, C * K)
        # torch.tanh(h): Applies hyperbolic tangent activation to keep feature values bounded between -1 and 1.
        # self.query(...): Computes raw attention score for each of the 988 items -> shape(B, 988, 1).
        attn_logits = self.query(torch.tanh(h))
        # F.softmax(...., dim=1): convers raw scores across the 988 items into percentage weights that sum to 1.0
        alpha = F.softmax(attn_logits, dim=1)
        # torch.sum(.., dim=1): multiplies each item by its attention weight and sums across all 988 items -> shape(B, 32)
        pooled = torch.sum(alpha * h, dim=1)
        return pooled

class CNN3DModel(nn.Module):
    def __init__(self, device, item_number, item_label_number,
                 performer_heads, performer_dim=32, depth=1, use_attn_pool=True):
        super(CNN3DModel, self).__init__()
        self.device = device
        self.item_number = item_number
        self.item_label_number = item_label_number
        self.performer_heads = performer_heads
        # Boolean switch : True uses our attention pooling ; False uses the old baseline flattening.
        self.use_attn_pool = use_attn_pool

        self.performer = Performer(
            dim=item_label_number,
            heads=performer_heads,
            dim_head=max(1, item_label_number // max(1, performer_heads)),
            causal=False,
            nb_features=performer_dim,
            depth=depth,
            feature_redraw_interval=None
        ).to(device)
        in_ch = 2

        self.stage1 = ConvNeXt3DDownsample(in_ch, 4, mlp_ratio=2, gn_max_groups=4, with_block=True).to(device)
        self.stage2 = ConvNeXt3DDownsample(4, 8, mlp_ratio=2, gn_max_groups=4, with_block=True).to(device)
        self.attn_stage1 = ConvNeXt3DDownsample(in_ch, 4, mlp_ratio=2, gn_max_groups=4, with_block=True).to(device)
        self.attn_stage2 = ConvNeXt3DDownsample(4, 8, mlp_ratio=2, gn_max_groups=4, with_block=True).to(device)

        if self.use_attn_pool:
            # After 2 downsample stages, label_dim becomes 16 -> 8 -> 4, with 8 channels(8 * 4 = 32).
            downsampled_label_dim = item_label_number // 4
            self.local_pool = AttentionPooling3D(channels=8, label_dim=downsampled_label_dim).to(device)
            self.global_pool = AttentionPooling3D(channels=8, label_dim=downsampled_label_dim).to(device)
            # 32 features instead of 31,616 flattened numbers.
            self.conv_output_size = 8 * downsampled_label_dim
        else: 
            self.conv_output_size = self._get_conv_output_size(item_number, item_label_number)

        self.gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=device))
        self.dropout = nn.Dropout(p=0.6)
        # as we already set the value in if or else condition, we don't need to overwrite it.
        # self.conv_output_size = self._get_conv_output_size(item_number, item_label_number)
        self.fc1 = nn.Linear(self.conv_output_size, 64).to(device)
        self.fc2 = nn.Linear(64, 1).to(device)

    def _build_concat_tensor(self, rating_matrix, item_labels_matrix, use_attended_labels=False):
        B, I = rating_matrix.shape
        K = item_labels_matrix.shape[1]
        rating_map = rating_matrix.unsqueeze(-1).expand(B, I, K)
        labels_map = item_labels_matrix.unsqueeze(0).expand(B, I, K)
        if use_attended_labels:
            labels_map = self.performer(labels_map)
        x = torch.stack([rating_map, labels_map], dim=1)
        x = x.unsqueeze(-1)
        return x

    # def forward_features_for_contrastive(self, rating_matrix, item_labels_matrix):
    #     x_in = self._build_concat_tensor(rating_matrix, item_labels_matrix, use_attended_labels=False)
    #     feat = self.stage1(x_in)
    #     feat = feat.view(feat.size(0), -1)
    #     return feat

    # def forward(self, rating_matrix, item_labels_matrix):
    #     x_in = self._build_concat_tensor(rating_matrix, item_labels_matrix, use_attended_labels=False)
    #     x_original = self.stage1(x_in)
    #     x_original = self.stage2(x_original)
    #     x_original = x_original.view(x_original.size(0), -1)
    #     x_attn_in = self._build_concat_tensor(rating_matrix, item_labels_matrix, use_attended_labels=True)
    #     x_attention = self.attn_stage1(x_attn_in)
    #     x_attention = self.attn_stage2(x_attention)
    #     x_attention = x_attention.view(x_attention.size(0), -1)
    #     gate_value = torch.sigmoid(self.gate)
    #     fused_feature = (1 - gate_value) * x_original + gate_value * x_attention
    #     x = self.dropout(fused_feature)
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x

    def forward_features(self, rating_matrix, item_labels_matrix):
        # 1. Local branc(3D-DSConv)
        x_in = self._build_concat_tensor(rating_matrix, item_labels_matrix, use_attended_labels=False)
        x_local = self.stage2(self.stage1(x_in))
        # Uses attention pooling (B, 32) if enabled, otherwise falls back to flattening (B, 316616).
        x_local = self.local_pool(x_local) if self.use_attn_pool else x_local.view(x_local.size(0), -1)

        # 2. global branch(Performer + 3D-DSConv)
        x_attn_in = self._build_concat_tensor(rating_matrix, item_labels_matrix, use_attended_labels=True)
        x_global = self.attn_stage2(self.attn_stage1(x_attn_in))
        x_global = self.global_pool(x_global) if self.use_attn_pool else x_global.view(x_global.size(0), -1)

        # 3. Gated fusion of Local and Global branches
        gate_value = torch.sigmoid(self.gate)
        fused_feature = (1.0 - gate_value) * x_local + gate_value * x_global

        # 4. Project into the 64-dimensional latent representation space
        z_u = F.relu(self.fc1(self.dropout(fused_feature)))
        return z_u

    def forward_features_for_contrastive(self, rating_matrix, item_labels_matrix):
        # Routes contrastive learning through the full fused 64D representation so BOTH branches and the gate get trained!
        return self.forward_features(rating_matrix, item_labels_matrix)

    def forward(self, rating_matrix, item_labels_matrix):
        # Extract the 64D latent vector z_u first, then passes it to fc2 to output the 1d classification logit.
        z_u = self.forward_features(rating_matrix, item_labels_matrix)
        return self.fc2(z_u)

    def _get_conv_output_size(self, item_number, item_label_number):
        with torch.no_grad():
            rating_dummy = torch.randn(1, item_number, device=self.device)
            labels_dummy = torch.randn(item_number, item_label_number, device=self.device)
            x_in = self._build_concat_tensor(rating_dummy, labels_dummy, use_attended_labels=False)
            x = self.stage1(x_in)
            x = self.stage2(x)
            return x.view(1, -1).size(1)
def train_cnn_3d(
    model,
    rating_train_tensor,
    item_labels_tensor,
    y_train_tensor,
    learning_rate,
    epochs=50,
    batch_size=8,
    epsilon=0.01,
    alpha=0.005,
    num_steps=1,
    checkpoint_path=None,
):
    # ==========FIX 1===========================
    # even though in main_ml_1m.py we change to mps as cpu but it won't work as intended, these lines reads torch.cuda.is_available() which is obviously be false and moved entire model back to cpu which will lead to slow processing.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # to overcome this we use
    device = model.device
    model = model.to(device)
    item_labels_tensor = item_labels_tensor.to(device)
    rating_train_tensor = rating_train_tensor.float().to(device)
    y_train_tensor = y_train_tensor.view(-1).float().to(device)
    classification_criterion = DynamicWeightedFocalLoss(gamma=2).to(device)
    contrastive_criterion = HardNegativeContrastiveLoss(temperature=0.5,hard_neg_k=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_dataset = TensorDataset(rating_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader),
        epochs=epochs,
    )
    # AMP/GradScaler is intentionally disabled because Performer attention produced
    # NaN gradients with reduced-precision CUDA calculations; training stays in float32.
    scaler = torch.amp.GradScaler(enabled=False) 
    start_epoch = 0

    # A checkpoint contains more than weights: it also stores optimizer and scheduler
    # state, allowing an interrupted Colab run to continue after the last saved epoch.
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(
            f"[CNN] Resuming from checkpoint after epoch {start_epoch}.",
            flush=True,
        )
    else:
        model.apply(init_weights)

    # Anomaly detection was useful while locating the NaN source, but it slows normal
    # training considerably, so it is disabled for the full experiment.
    torch.autograd.set_detect_anomaly(False)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (rating_batch, y_batch) in enumerate(train_dataloader):

            debug = (epoch == 0 and batch_idx == 0)
            # Keep logs readable during long runs while still showing periodic progress.
            if batch_idx % 100 == 0:
                print(
                    f"[CNN] Epoch {epoch+1}/{epochs} | "
                    f"Batch {batch_idx+1}/{len(train_dataloader)}",
                    flush=True
                )

            rating_batch = rating_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            # amp_context = (
            #     torch.amp.autocast(device_type="cuda")
            #     if device.type == "cuda"
            #     else nullcontext()
            # )
            amp_context = nullcontext()

            with amp_context:
                rating_batch, y_batch = augment_data(rating_batch, y_batch)
                if debug:
                    print("[DEBUG] Augmentation completed.", flush=True)

                rating_batch, y_batch = mixup(rating_batch, y_batch)
                if debug:
                    print(f"[DEBUG] Mixup completed.", flush=True)

                adv_rating_batch = generate_adversarial_samples(
                    model,
                    rating_batch,
                    y_batch,
                    item_labels_tensor,
                    epsilon=epsilon,
                    alpha=alpha,
                    num_steps=num_steps,
                )
                if debug:
                    print("[DEBUG] Adversarial generation completed.", flush=True)

                outputs = model(rating_batch, item_labels_tensor).view(-1)
                if debug:
                    print("[DEBUG] Normal forward completed.", flush=True)
                
                adv_outputs = model(adv_rating_batch, item_labels_tensor).view(-1)
                if debug:
                    print("[DEBUG] Adversarial forward completed.", flush=True)
                
                cls_loss = classification_criterion(outputs, y_batch) + classification_criterion(adv_outputs, y_batch)

                feats = model.forward_features_for_contrastive(rating_batch, item_labels_tensor)
                if debug:
                    print("[DEBUG] Contrastive forward completed.", flush=True)

                hard_labels = (y_batch > 0.5).float()

                con_loss = contrastive_criterion(feats, hard_labels)

                loss = cls_loss + con_loss

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += float(loss.detach().cpu())
        avg_train_loss = total_loss / max(1, len(train_dataloader))
        
        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.4f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if checkpoint_path:
            # Save only after a complete epoch so a restart loses at most one epoch.
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                checkpoint_path,
            )
            print(
                f"[CNN] Checkpoint saved after epoch {epoch + 1}: "
                f"{checkpoint_path}",
                flush=True,
            )
    return model