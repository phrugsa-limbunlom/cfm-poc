"""Downstream few-shot landmark-detection probe for a pretrained CFM +
delta-alignment backbone. Importable: call run_probe(args).

Downstream protocol is aligned to CDPM-Align (the controlled variable in this
study, so the backbone is the ONLY difference vs the paper):
  * spatial-softmax + NLL heatmap loss (CDPM CustomNLLLoss), NOT background-
    dominated heatmap MSE;
  * full fine-tune of the backbone (freeze_backbone=False) by default;
  * AdamW (weight_decay) + early stopping on test MRE (patience).
An unnormalised-Gaussian MSE head is still selectable (loss="mse") as a
diagnostic.
"""

import copy
import random
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from deltaflow.models.conditional_unet import ConditionalUNetVelocityField

try:
    from cfm_pretrain import build_dataset
except ImportError:
    from pretrain_cfm_delta_align import build_dataset


def landmarks_to_target_heatmaps(landmarks, image_size, sigma=3.0):
    """(K, 2) landmarks -> (K, H, W) per-landmark Gaussian targets (peak 1)."""
    device = landmarks.device
    ys = torch.arange(image_size, device=device).view(1, -1, 1)
    xs = torch.arange(image_size, device=device).view(1, 1, -1)
    lx = landmarks[:, 0].view(-1, 1, 1)
    ly = landmarks[:, 1].view(-1, 1, 1)
    return torch.exp(-((xs - lx) ** 2 + (ys - ly) ** 2) / (2 * sigma**2))


def landmarks_to_prob_target(landmarks, image_size, sigma=3.0):
    """(K, 2) landmarks -> (K, H, W) per-landmark PROBABILITY targets that sum
    to 1 over the spatial map (a valid distribution for NLL / cross-entropy).

    sigma > 0 gives a normalised Gaussian (soft label); sigma <= 0 gives a
    one-hot target at the rounded pixel (CDPM cdpm_align uses sigma=0)."""
    K = landmarks.shape[0]
    device = landmarks.device
    if sigma and sigma > 0:
        hm = landmarks_to_target_heatmaps(landmarks, image_size, sigma)
        denom = hm.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
        return hm / denom
    hm = torch.zeros(K, image_size, image_size, device=device)
    xy = landmarks.round().long().clamp_(0, image_size - 1)
    hm[torch.arange(K, device=device), xy[:, 1], xy[:, 0]] = 1.0
    return hm


def two_d_softmax(logits):
    """(B, K, H, W) -> softmax over the H*W spatial dimension per (B, K)."""
    B, K, H, W = logits.shape
    return F.softmax(logits.view(B, K, -1), dim=-1).view(B, K, H, W)


def heatmap_nll_loss(logits, target_prob, eps=1e-10):
    """CDPM-style spatial cross-entropy: -sum_{H,W} target * log(softmax(logits)),
    averaged over batch and landmarks. Immune to background dominance because the
    prediction is normalised over the map."""
    pred = two_d_softmax(logits)
    nll = -(target_prob * torch.log(pred + eps)).sum(dim=(-1, -2))
    return nll.mean()


def heatmaps_to_coords(heatmaps, subpixel=True):
    """(B, K, H, W) -> (B, K, 2) peak coords as (x, y).

    CDPM-Align's ``get_hottest_point`` refines the integer argmax with a
    quadratic (parabolic) sub-pixel fit along each axis, rather than returning
    the raw argmax. At 256px one pixel is ~0.75-0.94 mm on ISBI2015, so the
    ~0.3 px average quantisation error of a plain argmax is a non-trivial slice
    of the mm-scale MRE we are trying to beat. ``subpixel=True`` (default)
    matches CDPM; set it False for the old integer-argmax behaviour.

    Parabolic offset along an axis from the peak value ``c0`` and its two
    neighbours ``cm1``/``cp1``:  ``delta = 0.5*(cm1 - cp1)/(cm1 - 2*c0 + cp1)``,
    clamped to [-0.5, 0.5]; the denominator is guarded against ~0 curvature.
    """
    B, K, H, W = heatmaps.shape
    idx = heatmaps.view(B, K, -1).argmax(dim=-1)
    ys = idx // W
    xs = idx % W
    xs_f = xs.float()
    ys_f = ys.float()
    if subpixel:
        bi = torch.arange(B, device=heatmaps.device).view(B, 1)
        ki = torch.arange(K, device=heatmaps.device).view(1, K)

        def _off(cm1, c0, cp1):
            denom = cm1 - 2.0 * c0 + cp1
            off = 0.5 * (cm1 - cp1) / denom
            off = torch.where(denom.abs() < 1e-6, torch.zeros_like(off), off)
            return off.clamp(-0.5, 0.5)

        xm1 = (xs - 1).clamp(0, W - 1)
        xp1 = (xs + 1).clamp(0, W - 1)
        ym1 = (ys - 1).clamp(0, H - 1)
        yp1 = (ys + 1).clamp(0, H - 1)
        c0 = heatmaps[bi, ki, ys, xs]
        xs_f = xs_f + _off(heatmaps[bi, ki, ys, xm1], c0, heatmaps[bi, ki, ys, xp1])
        ys_f = ys_f + _off(heatmaps[bi, ki, ym1, xs], c0, heatmaps[bi, ki, yp1, xs])
    return torch.stack([xs_f, ys_f], dim=-1)


class MultiScaleHeatmapHead(nn.Module):
    LEVELS = ("enc_1_4", "dec_1_8", "bottleneck")

    def __init__(self, num_landmarks, feature_dims, image_size):
        super().__init__()
        self.image_size = image_size
        in_ch = sum(feature_dims[k] for k in self.LEVELS)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
        )
        self.out = nn.Conv2d(64, num_landmarks, 1)

    def forward(self, feats):
        size = (self.image_size, self.image_size)
        parts = [
            F.interpolate(feats[k], size=size, mode="bilinear", align_corners=False)
            for k in self.LEVELS
        ]
        return self.out(self.fuse(torch.cat(parts, dim=1)))


class LandmarkProbe(nn.Module):
    def __init__(self, backbone, num_landmarks, feature_dims, image_size,
                 t_value=1.0, freeze=True):
        super().__init__()
        self.backbone = backbone
        self.head = MultiScaleHeatmapHead(num_landmarks, feature_dims, image_size)
        self.t_value = t_value
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        B = x.shape[0]
        t = torch.full((B,), self.t_value, device=x.device, dtype=x.dtype)
        if self.freeze:
            self.backbone.eval()
            with torch.no_grad():
                _, feats = self.backbone.forward_with_features(x, t, cond=None)
            feats = {k: v.detach() for k, v in feats.items()}
        else:
            _, feats = self.backbone.forward_with_features(x, t, cond=None)
        return self.head(feats)


def compute_metrics(pred, gt, mm_per_pixel, thresholds=(2.0, 2.5, 3.0, 4.0)):
    mm = torch.as_tensor(mm_per_pixel, dtype=torch.float32)
    err = torch.linalg.norm((pred - gt) * mm, dim=-1).flatten()
    metrics = {
        "MRE": err.mean().item(),
        "std": err.std(unbiased=False).item(),
        "P95": torch.quantile(err, 0.95).item(),
    }
    for z in thresholds:
        metrics[f"SDR@{z}mm"] = (err < z).float().mean().item() * 100.0
    return metrics


@torch.no_grad()
def evaluate(probe, loader, device, mm_per_pixel):
    probe.eval()
    preds, gts = [], []
    for x, lm in loader:
        x = x.to(device)
        preds.append(heatmaps_to_coords(probe(x)).cpu())
        gts.append(lm.float())
    return compute_metrics(torch.cat(preds), torch.cat(gts), mm_per_pixel)


def _split_indices(n, test_frac, n_shot, seed, val_frac=0.15):
    """Deterministic seeded split -> (train_idx, val_idx, test_idx).

    The permutation's first ``round(n*test_frac)`` entries are the fixed held-out
    TEST set (never touched during training/selection). The remaining pool yields
    the few-shot TRAIN set (``n_shot`` images) and a disjoint VAL set used for
    early stopping / model selection -- selecting on VAL rather than TEST avoids
    the optimistic bias of peeking at the evaluation set. ``val_frac`` is a
    fraction of ``n`` (clamped to what the pool can spare after the train shots).
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    test_size = max(1, int(round(n * test_frac)))
    test_idx = perm[:test_size]
    pool = perm[test_size:]
    train_idx = pool[:n_shot] if n_shot else pool
    if not train_idx:
        raise ValueError(f"empty train split (dataset size {n})")
    # VAL comes from the pool AFTER the few-shot train images (disjoint from both
    # train and test). Falls back to empty if the pool has nothing left to spare.
    val_start = len(train_idx)
    val_size = min(max(1, int(round(n * val_frac))), max(0, len(pool) - val_start))
    val_idx = pool[val_start:val_start + val_size]
    return train_idx, val_idx, test_idx


def build_eval_split(args):
    """Rebuild the exact (deterministic) train/val/test split used by run_probe.

    Returns (train_ds, val_ds, test_ds, mm_per_pixel, lm0). Because the split is
    fully seeded, calling this again reproduces the identical held-out test set --
    handy for qualitatively visualising a trained probe on its own test images.
    ``val_ds`` is used for early stopping / model selection so we never select on
    the test set. It may be an empty Subset if the pool has no spare images.
    """
    def _ds(phase, n_shot):
        return SimpleNamespace(
            dataset=args.dataset, root=args.root, landmarks_dir=args.landmarks_dir,
            landmark_ref_size=args.landmark_ref_size, split=args.split,
            image_size=args.image_size, n_shot=n_shot, phase=phase,
        )

    if args.dataset == "isbi2015":
        # CDPM-Align faithful protocol: sorted-filename split -- few-shot pool is
        # train=[:130], the fixed val=[130:150] (20 imgs, for model selection),
        # and the fixed test=[150:400] (250 imgs); mm is per-axis from the dataset
        # (NOT a single scalar).
        full_train = build_dataset(_ds("train", None))
        val_ds = build_dataset(_ds("val", None))
        test_ds = build_dataset(_ds("test", None))
        n = min(args.n_shot, len(full_train)) if args.n_shot else len(full_train)
        g = torch.Generator().manual_seed(args.seed)
        shot_idx = torch.randperm(len(full_train), generator=g).tolist()[:n]
        train_ds = Subset(full_train, shot_idx)
        mm_per_pixel = full_train.mm_per_pixel
        _, lm0 = full_train[shot_idx[0]]
        print(f"ISBI2015 CDPM split: train(n-shot)={len(train_ds)} "
              f"val={len(val_ds)} test={len(test_ds)} "
              f"mm/px={[round(m,4) for m in mm_per_pixel]}")
    else:
        dataset = build_dataset(_ds("all", None))
        train_idx, val_idx, test_idx = _split_indices(
            len(dataset), args.test_frac, args.n_shot, args.seed
        )
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        test_ds = Subset(dataset, test_idx)
        mm_per_pixel = args.mm_per_pixel
        _, lm0 = dataset[train_idx[0]]
        print(f"dataset={len(dataset)}  train(n-shot)={len(train_ds)}  "
              f"val={len(val_ds)}  test={len(test_ds)}")
    return train_ds, val_ds, test_ds, mm_per_pixel, lm0


def run_probe(args):
    """Train the probe and return (probe, history) with per-epoch metrics.

    Recognised (optional) CDPM-protocol knobs, read via getattr with defaults:
      loss          : "nll" (default) | "mse"
      weight_decay  : AdamW weight decay (default 1e-4)
      patience      : early-stopping patience on VAL MRE, 0 disables (default 15)

    Model selection (early stopping + best-checkpoint restore) is done on the
    held-out VAL split, NOT the test set, so the reported test metrics are not
    optimistically biased by peeking at the evaluation set. When no val images are
    available (empty val split) it transparently falls back to selecting on test.
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    loss_type = getattr(args, "loss", "nll")
    weight_decay = getattr(args, "weight_decay", 1e-4)
    patience = getattr(args, "patience", 15)

    train_ds, val_ds, test_ds, mm_per_pixel, lm0 = build_eval_split(args)

    K = lm0.shape[0]
    print(f"num landmarks: {K}")

    backbone = ConditionalUNetVelocityField(
        in_channels=1, cond_channels=1, num_classes=getattr(args, "num_classes", 0)
    )
    if args.backbone_ckpt:
        backbone.load_state_dict(torch.load(args.backbone_ckpt, map_location="cpu"))
        print(f"loaded pretrained backbone: {args.backbone_ckpt}")
    else:
        print("random-init backbone (from-scratch baseline)")
    backbone.to(device)

    feature_dims = backbone.feature_channels()
    probe = LandmarkProbe(
        backbone, K, feature_dims, args.image_size,
        t_value=args.t_value, freeze=args.freeze_backbone,
    ).to(device)
    trainable = [p for p in probe.parameters() if p.requires_grad]
    mode = "linear-probe (frozen)" if args.freeze_backbone else "fine-tune (full)"
    print(f"mode: {mode} | loss: {loss_type} | wd: {weight_decay} | patience: {patience}")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    use_val = len(val_ds) > 0
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if use_val else None
    sel_name = "val" if use_val else "test"
    if not use_val:
        print("[warn] empty val split -> selecting on TEST (biased); "
              "increase dataset/pool size for an unbiased val split.")

    history = []
    best_sel = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(args.epochs):
        probe.train()
        if args.freeze_backbone:
            probe.backbone.eval()
        ep_loss, nb = 0.0, 0
        for x, lm in train_loader:
            x = x.to(device)
            lm = lm.to(device).float()
            if loss_type == "mse":
                target = torch.stack([
                    landmarks_to_target_heatmaps(lm[i], args.image_size, args.sigma)
                    for i in range(x.shape[0])
                ])
                loss = F.mse_loss(probe(x), target)
            else:
                target = torch.stack([
                    landmarks_to_prob_target(lm[i], args.image_size, args.sigma)
                    for i in range(x.shape[0])
                ])
                loss = heatmap_nll_loss(probe(x), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1

        # Report metrics on TEST; select (early-stop) on VAL so we never peek at
        # the test set. ``val_MRE`` is stored alongside the test metrics so the
        # caller can pick the val-selected epoch without touching test.
        metrics = evaluate(probe, test_loader, device, mm_per_pixel)
        val_mre = (evaluate(probe, val_loader, device, mm_per_pixel)["MRE"]
                   if use_val else metrics["MRE"])
        sel = val_mre
        record = {"epoch": epoch, "train_loss": ep_loss / max(nb, 1),
                  "val_MRE": val_mre, **metrics}
        history.append(record)
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            unit = "mm"
            print(f"epoch={epoch:3d} loss={record['train_loss']:.4f} "
                  f"val_MRE={val_mre:.3f}{unit} "
                  f"test_MRE={metrics['MRE']:.3f}{unit} "
                  f"SDR@2mm={metrics['SDR@2.0mm']:.1f}% "
                  f"P95={metrics['P95']:.3f}{unit}")

        # Early stopping on the SELECTION metric (val MRE); keep best-so-far head.
        if sel < best_sel - 1e-6:
            best_sel = sel
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in probe.state_dict().items()}
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                print(f"early stop at epoch {epoch} (best {sel_name} "
                      f"MRE={best_sel:.3f}, no improvement for {patience} epochs)")
                break

    if best_state is not None:
        probe.load_state_dict(best_state)
        print(f"restored best epoch by {sel_name} (MRE={best_sel:.3f})")

    if args.out:
        torch.save(probe.head.state_dict(), args.out)
    return probe, history

@torch.no_grad()
def visualize_predictions(probe, test_ds, image_size, n=6, out=None, title=None,
                          seed=0):
    """Overlay predicted (red x) vs ground-truth (green o) landmarks on a few
    held-out test images for a trained probe. Yellow lines show per-point error.
    Returns the matplotlib Figure (also saved to ``out`` if given)."""
    import matplotlib.pyplot as plt

    device = next(probe.parameters()).device
    probe.eval()
    n = min(n, len(test_ds))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(test_ds), generator=g).tolist()[:n]

    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows),
                             squeeze=False)
    for a in axes.ravel():
        a.axis("off")

    for a, i in zip(axes.ravel(), idx):
        x, gt = test_ds[i]
        xb = x.unsqueeze(0).to(device)
        pred = heatmaps_to_coords(probe(xb))[0].cpu()   # (K, 2) as (x, y)
        gt = gt.float()
        a.imshow(x[0].cpu(), cmap="gray")
        for (gx, gy), (px, py) in zip(gt, pred):
            a.plot([gx, px], [gy, py], c="yellow", lw=0.6, alpha=0.8)
        a.scatter(gt[:, 0], gt[:, 1], s=20, c="lime", marker="o",
                  edgecolors="black", linewidths=0.4)
        a.scatter(pred[:, 0], pred[:, 1], s=24, c="red", marker="x")
        px_mre = torch.linalg.norm(pred - gt, dim=-1).mean().item()
        a.set_title(f"test idx {i}  px-MRE={px_mre:.1f}", fontsize=9)

    handles = [
        plt.Line2D([], [], marker="o", color="lime", ls="", label="ground truth"),
        plt.Line2D([], [], marker="x", color="red", ls="", label="predicted"),
    ]
    fig.legend(handles=handles, loc="upper right")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
    return fig
