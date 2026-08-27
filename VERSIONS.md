# Kaggle kernel version log

Per-version changelog for the Kaggle notebook
`phrugsalimbunlom/deltaflow-cfm-delta-alignment-pretraining-poc`
(code file: `cfm_delta_align_pretrain.ipynb`).

Newest first. One entry per `kaggle kernels push`. Keep this append-only:
add a new heading each time a version is pushed; do not rewrite past entries.

---

## v23 (2026-08-26)

**DDPM noise-prediction backbone — the iso-compute CFM-vs-DDPM head-to-head baseline.**

- **Independent variable flipped:** `EXPERIMENT["backbone"] = "ddpm"` (was
  `"cfm"`). Runs CDPM's DDPM noise-prediction objective instead of flow
  matching, as the internal baseline for a fair single-variable comparison.
- **New `DDPMInterpolant`** (cell 11): paper schedule (linear β 1e-4→0.028,
  T=500), repo t-convention (t=1 clean, t=0 noise), forward
  `x_t = √ᾱ·x1 + √(1−ᾱ)·ε`, **target = ε**. Implemented as a drop-in
  interpolant so the UNet, delta-alignment loss, two-timestep machinery and
  gradient accumulation are reused unchanged — only the forward process +
  regression target differ. Verified: t=0.6 → DDPM index 200, i.e. exactly
  CDPM's t=200 feature-extraction step.
- **`ot_coupling = False`** (was True): standard DDPM does not OT-couple its
  noise; the path knobs (`interpolant`/`sb_sigma`) are ignored on the DDPM path.
- **Probe forward-t is now backbone-aware** (cell 18): DDPM probes at t≈0.6
  (=CDPM t=200/T=500); CFM stays at t=1.0. Overridable via
  `EXPERIMENT["probe_t"]`.
- Added `--backbone {cfm,ddpm}` CLI flag; `backbone="cfm"` path is byte-identical
  to v22. Controls held identical to the CFM runs: 3300 iters, eff batch 16,
  align=True, align_frac 0.1, seed 0, same corpus and probe protocol.
- **Result (vs CFM v20, frozen ISBI probe):** CFM wins decisively — MRE 3.22 vs
  DDPM 4.90 mm, SDR@2mm 43.9% vs 37.9%; random-init controls ≈ equal (7.62 vs
  7.59 mm), so the flow-matching objective is a clearly better representation
  learner at equal compute. Finetune washes out (CFM 2.75 vs DDPM 2.99 mm).



- **Schrodinger-bridge probability path** (new independent variable
  `EXPERIMENT["interpolant"]="schrodinger"`, `sb_sigma=0.1`). Swaps the
  straight-line linear/rectified-flow path for the library
  `SchrodingerBridgeInterpolant` (Brownian-bridge path
  `x_t=(1-t)x0+t*x1+sigma*sqrt(t(1-t))*z`). OT coupling stays on, so vs v21 the
  ONLY changed variable is the path.
- **sigma=0.1 (user-selected):** path (linear vs SB) and coupling (independent
  vs OT) are orthogonal. Library docs: unregularised OT = small-sigma limit of
  the true entropic bridge, so with OT coupling on a small sigma is the
  consistent choice; larger sigma injects more boundary-divergent drift.
- **Implementation:** interpolant factory in `run_training`
  (`SchrodingerBridgeInterpolant(sigma=sb_sigma)` vs `LinearInterpolant()`), the
  pre-coupled `x0` threads through unchanged; added `--interpolant`/`--sb-sigma`
  CLI flags and cell-14 wiring.
- **Stability verified:** SB target has a `sigma*(1-2t)/(2*sqrt(t(1-t)))*z` term
  that diverges at t->0/1 (eps-floored); the loss path clamps preds to [-10,10]
  and loss to [0,50] (no NaNs). Smoke test (OT-coupled 16-batch, images [-1,1]):
  mid-t |target| max ~4.0, near-boundary max only ~18.8 (small because
  sigma=0.1), and sigma=0 reproduces the linear path EXACTLY. Pushed as
  **version 22**.

## v21 (2026-08-26)

- **Refactor: use the deltaflow library `OTCoupling` instead of a hand-rolled
  helper.** Replaced the local `_ot_couple` function with
  `deltaflow.trainer.coupling.OTCoupling`, whose `sample_pair(x1)` draws
  `x0 ~ N(0, I)` and OT-permutes it (same Hungarian/greedy logic via the
  library's `_batch_ot_permutation`). `run_training` builds `OTCoupling()` once
  (only when `ot_coupling=True`) and calls `sample_pair(x1_full)` over the full
  effective batch for each of the two alignment timesteps.
- **No behaviour change vs v20:** verified the library coupling reproduces the
  same result (toy 16-batch transport cost 2029 -> 1720) and leaves x1 unchanged;
  the `ot_coupling=False` path is still byte-identical to the baseline. Removed
  the now-unused import. Uses the library's clean path/coupling separation
  (interpolants = probability path; trainer.coupling = which (x0, x1) pairs).
  Pushed as **version 21**.

## v20 (2026-08-26)

- **Mini-batch optimal-transport (OT) coupling for the interpolant** (new
  independent variable `EXPERIMENT["ot_coupling"]=True`). Re-orders the noise so
  each (noise, image) pair minimises squared-L2 transport cost (Tong et al.
  2023, arXiv:2302.00482) -> straighter flow paths. Everything else identical to
  v19 (`align=True`, `pretrain_iters=3300`).
- **Computed over the FULL effective batch, not the micro-batch.** Because the
  interpolant sees micro-batches of `batch_size=2` (grad_accum=8 -> eff 16), a
  naive per-micro-batch OT would be a near no-op. `run_training` now draws all
  `grad_accum` micro-batches up front, OT-couples the noise once over all 16
  samples (`_ot_couple`: SciPy Hungarian, greedy fallback), then slices it back
  into micro-batches for memory-safe gradient accumulation.
- **Implementation:** kept `LinearInterpolant` and threaded PRE-COUPLED noise
  (`x0_1`/`x0_2`) into `_compute_batch_loss` (passing to `OTInterpolant` would
  double-permute). Two independent couplings for the two alignment timesteps, so
  the ONLY change vs baseline is the coupling. Added `--ot-coupling` CLI flag.
- **Control integrity:** with `ot_coupling=False`, noise stays None and each
  micro-batch draws its own -> byte-identical to the independent-coupling
  baseline. Verified `_ot_couple` reduces transport cost (2166 -> 1720 on a toy
  16-batch) and slicing preserves pairing. Overhead negligible, so the ~6.0 h
  budget is unchanged. Pushed as **version 20**.

## v19 (2026-08-25)

- **Re-run of v18 (no code changes).** v18 was cancelled before/early in
  execution, so it was re-pushed unchanged to start a fresh run. Config and code
  identical to v18 (`align=True`, `pretrain_iters=3300`, flow-phase `loss_align`
  skip). Pushed as **version 19** (`python -m kaggle kernels push -p .`).

## v18 (2026-08-25)

- **Skip computing/logging the alignment loss during the flow phase.** In an
  `align=True` run the schedule sets `lambda_align=0` for the first
  `(1 - align_frac)` of iterations, yet `_compute_batch_loss` still called
  `DeltaAlignmentLoss.forward`, which computed the projector/cosine term and
  logged a non-zero `loss_align` (e.g. `iter=310 [flow] loss_align=0.0246`)
  that contributed **zero gradient**.
- **Fix (cell 11):** added a flow-phase fast path -- when
  `loss_fn.lambda_align == 0` it computes the SAME two-timestep, cond+uncond
  flow loss (with the identical `/2` per-mode averaging as the library) but
  skips the alignment term and projector, and returns a `loss_dict` **without**
  `loss_align`. The final `align_frac` of iterations (lambda_align=5.0) falls
  through to the full alignment path unchanged.
- **Training-identical:** with `lambda_align=0` the alignment term was already
  x0 (projector never moved during the flow phase); all 4 backbone forwards are
  retained, so the flow objective is byte-for-byte the same. Net effect: same
  weights, minus the wasted projector forward and the misleading log value.
- **Plot cell (15):** `loss_align` is now logged only in align-phase records, so
  the loss-curve cell gathers `al`/`al_steps` from records that contain the key
  (avoids a KeyError; draws the alignment curve over just the shaded final-10%).
- Config otherwise unchanged from v17 (`align=True`, `pretrain_iters=3300`).
  Pushed as **version 18** (`python -m kaggle kernels push -p .`).

## v17 (2026-08-16)

- **Alignment loss turned ON (faithful CDPM-Align), with the pretrain budget
  resized to fit:** `align` **False -> True**; `pretrain_iters` **6600 -> 3300**.
  `lambda_align` stays 5.0, `align_frac` 0.1 (alignment active in the final 10%,
  iters **2970-3300**). Corpus, backbone, optimiser otherwise unchanged vs v16.
- **Why resize iters:** with `align=True`, `_compute_batch_loss` runs **4**
  backbone forwards/step (cond+uncond x two timesteps), not 2, so throughput
  drops from ~2.94 s/update (v15 pure-flow) to **~6.57 s/update** (measured in
  the v10/v11 align-on logs). At 6.57 s/update: `align_start=2970` is reached at
  **~5.42 h** and all 3300 iters finish at **~6.02 h**, both UNDER the 6.5 h
  `max_pretrain_seconds` soft cap -- so the alignment phase actually runs and the
  aligned backbone is checkpointed (`ckpt_every=250` + at end) for the probes.
- **Bug this avoids (v10 precedent):** v10 ran align=True at 4500 iters
  (`align_start=4050`); the soft cap fired at **iter 3827** BEFORE `align_start`,
  so alignment NEVER ran. Leaving `pretrain_iters=6600` here would repeat that
  (~10.8 h projected; cap fires ~iter 3560 << align_start 5940). 3300 is the
  v11-proven align-on size.
- **Downstream:** the probes load `args.out`, which is the alignment-shaped
  backbone once the final-10% phase runs -- no extra wiring needed. Updated
  config cell 2 comments (NOTE v17 block, align-window). Pushed as **version 17**
  (`python -m kaggle kernels push -p .`).

## v16 (2026-08-16)

- **Longer pure-flow pretraining, single-variable change vs v15:**
  `pretrain_iters` 3300 -> **6600**. `align` stays **False**, corpus stays the
  988-image CDPM-parity cap, backbone/probe/optimiser all unchanged.
- **Rationale:** the v15 run log measured actual throughput at **~2.94 s/update**
  (3290 iters in ~9.66 ks), NOT the stale ~6.57 s/update the config assumed. So
  3300 iters used only ~2.7 h of the ~5.5 h session -- ~2x headroom. Spend it on
  more optimiser updates: 6600 * 2.94 s ~= **5.4 h** pretrain, leaving ~2.8 h for
  the 3 probes (chest-real finetune, ISBI finetune, ISBI frozen).
- **What v15 showed (motivation):** ISBI 25-shot finetune MRE **2.84 mm** /
  SDR@2mm 50.1% -- WORSE than v13's 2.65 mm (the 988 cap discarded ~1100 imgs),
  and far from CDPM's 1.54 mm / 77.5%. Pretrained beat random by only 0.40 mm and
  flow loss plateaued at ~0.05, i.e. the backbone is undertrained/under-helping.
- Updated stale throughput + align-window comments in config cell 2. Mirror
  `cfm_pretrain.py` needs no change (iters come from the notebook EXPERIMENT dict
  via argparse defaults). Pushed as **version 16** (`python -m kaggle kernels push -p .`).

## v15 (2026-08-16)

- **Match the pretraining corpus SIZE to CDPM exactly: 988 images.** v14's pool
  was actually **2085** images after the no-leakage test exclusion (Shenzhen 545
  + ISBI2015 150 + DHA 1390) -- LARGER than CDPM's 988, because our Shenzhen
  source is the full 662-image TB set (vs CDPM's 279 landmark subset) and we kept
  all 1390 DHA (vs CDPM's ~910-minus-test). Corpus size was thus an uncontrolled
  variable vs the paper.
- **Fix:** added `per_source_targets` to `CombinedImageDataset` (applied AFTER
  the test exclusion; deterministic sorted-filename prefix). Config cell 2:
  `pretrain_targets = {1: 279, 2: 150, 3: 559}` -> **988** total. Wired through
  `build_dataset` + cell 14; mirror `cfm_pretrain.py` and cell 11 kept in sync.
- **Composition** is a self-consistent reconstruction of CDPM's 988 (the paper
  publishes per-dataset totals + the pooled 988 but not the per-dataset test
  split): ISBI2015=150 (standard 150/250 split, matches our natural count),
  Shenzhen=279 (CDPM's full Shenzhen set), DHA=559 (remainder to 988).
- **Trade-off:** this discards data we have (Shenzhen 545->279, DHA 1390->559)
  purely to make corpus-size a matched control vs CDPM. Set
  `pretrain_targets=None` to instead train on all 2085 images (more data, but
  size no longer matched).
- Everything else unchanged: `align=False` (flow-loss only), `pretrain_iters=3300`,
  backbone capacity. Pushed as **version 15** (`python -m kaggle kernels push -p .`).



- **Close the "less data" gap vs CDPM: add the Digital Hand Atlas (DHA) to the
  pretraining corpus, still FLOW-LOSS ONLY (`align=False`).** CDPM pretrains on a
  pooled **3-dataset** corpus (Shenzhen 279 + ISBI2015 400 + DHA 910 = 988
  unlabeled images); we previously pooled only Shenzhen + ISBI2015, so DHA -- the
  single largest source -- was entirely missing. This is the concrete fix for the
  data-scarcity concern.
- **Hosted DHA as a Kaggle dataset** `phrugsalimbunlom/digital-hand-atlas-256`
  (1390 hand X-rays from the local `hand/jpg`, resized to 256x256 grayscale to
  match the loader's own resize; ~16 MB). Declared it in
  `kernel-metadata.json:dataset_sources`.
- **Wiring (single-variable change: corpus only):**
  - `cfm_pretrain.build_dataset` (`combined` branch) + mirror cell 11: append
    `(dha_root, 3, "*.jpg")` as **dataset index 3**; new `--dha-root` arg.
  - Cell 2: `num_classes` 3 -> **4**, `dataset_index += {"dha": 3}`.
  - Cell 9: new `find_dha_images()` -> `DHA_ROOT` (mount-agnostic recursive glob
    with a kagglehub fallback).
  - Cell 14: pass `dha_root=DHA_ROOT`. All probe/sampling cells already build the
    backbone with `num_classes=E.num_classes`, so the 4-class checkpoint loads
    cleanly downstream.
  - **Bugfix:** the mirror `deltaflow/models/cfm_pretrain.py` had a corrupted
    `def arun_training` (typo); restored to `run_training` to match cell 11.
- **Rationale (from CDPM.pdf):** the alignment loss buys only ~3% (CDPM lambda=0
  2.65 vs lambda=5 2.58 MRE), and >~1k images has diminishing returns (CDPM on
  112k NIH images barely beats CDPM on 988), so the path to CDPM with alignment
  OFF is: complete the 3-dataset corpus (this version), then reinvest the ~2x
  flow-only compute saving into more iterations (next lever). Iteration budget,
  backbone capacity, and `align=False` are unchanged this version to keep the
  corpus the single moving variable.
- Pushed as **version 14** with `python -m kaggle kernels push -p .` (CLI 2.2.4).



- **Fix `IndexError: too many indices for tensor of dimension 1`** in the
  "real anatomical landmarks" probe cell (`run_probe(make_real_probe_args(...))`,
  `--dataset chest`). Root cause: `deltaflow`'s `ChestXrayDataset` base class
  (`RadiographDataset.__getitem__`) returns landmarks **flattened** to
  `(K*2,)` and normalised to `[-1, 1]` by default, but
  `probe_landmark_detection.landmarks_to_target_heatmaps` indexes
  `landmarks[:, 0]`, expecting `(K, 2)` pixel-space coordinates -- this also
  masked a hidden double-scaling bug (annotations were rescaled
  `ref_size -> image_size` in `_load_landmarks`, then rescaled AGAIN by the
  base class's own `image_size / orig_w` rescale).
- Fixed `ChestLandmarkDataset` (in both `cfm_delta_align_pretrain.ipynb`'s
  `%%writefile cfm_pretrain.py` cell and the mirrored
  `deltaflow/models/cfm_pretrain.py` reference copy):
  - pass `normalize_landmarks=False` to keep pixel-space coordinates;
  - `_load_landmarks` now scales `ref_size -> actual on-disk image size`
    (not directly to `image_size`), so the base class's rescale applies
    exactly once;
  - added a `__getitem__` override that reshapes the flattened `(K*2,)`
    output back to `(K, 2)`.
- No experiment-config changes vs v12 (`pretrain_iters=3300`,
  `grad_accum=8`, `align_frac=0.1`, `max_pretrain_seconds=6.5h`,
  `EXPERIMENT["align"]=False`); this push only fixes the downstream
  real-landmark probe cell so it runs after the pretraining/ablation cells.
- Pushed with `python -m kaggle kernels push -p .` (kaggle CLI 2.2.4).

---

## v12 (2026-08-15)

- **Re-push, no code change.** Notebook config is identical to the v11 setup
  (`pretrain_iters=3300`, `grad_accum=8` -> eff. batch 16, `align_frac=0.1`,
  `max_pretrain_seconds=6.5h`), but `EXPERIMENT["align"]` is currently **`False`**
  -- this run is the **pure-flow-matching ablation** (independent variable OFF:
  no multi-scale delta-alignment loss, ~2x faster forward passes), not the
  alignment run v11's title targeted. Serves as the flow-matching-only control
  to compare against the align-on run for the frozen-probe MRE/SDR/ERE metrics.
- Pushed with `kaggle kernels push -p .` (kaggle CLI 2.2.4, installed via
  `pip install --upgrade kaggle` and invoked as `python -m kaggle` since the
  `kaggle` executable wasn't on PATH this session).

---

## v11 (2026-08-11)

- **Fix the v10 sizing bug so the alignment phase actually runs.** v10's 7.0 h
  `max_pretrain_seconds` cap soft-stopped pretraining at iter 3827 -- *before*
  `align_start=4050` -- because real throughput was **6.57 s/update** (v10 was
  sized on an over-optimistic 4.7 s), so alignment never activated and v10 was
  pure flow matching. v11 resizes to the measured throughput:
  `pretrain_iters` 4500 -> **3300** (align_start=2970, alignment = iters
  2970-3300 at the CDPM-faithful final 10%), `max_pretrain_seconds` 7.0h -> 6.5h
  (won't bite: ~6.0 h expected), leaving ~2 h for the probes. No code change --
  config-only (cell 2). `grad_accum=8` (eff batch 16) unchanged.
- **Goal:** first run where the final-10% delta-alignment phase actually
  executes, to test whether alignment (a) helps the frozen-probe representation
  vs v10's alignment-off result, and (b) avoids the v9 `loss_align -> 0`
  projector collapse.

---

## v10 (2026-08-10)

- **CDPM-faithful iteration budget (lever a).** Pretraining now runs on an
  *iteration* budget (optimiser updates), not epochs, matching how CDPM-Align
  measures compute. New `run_training` mode in `cfm_pretrain.py`:
  - `max_iters=4500` optimiser updates (~5.9 h at 256px on the T4).
  - `grad_accum=8` -> **effective batch = 2 x 8 = 16** (CDPM's batch size),
    lower-noise gradients at images-seen matched to v9.
  - `align_frac=0.1` -> **alignment loss active ONLY in the final 10%** of
    iterations (CDPM schedule); first 90% is pure flow matching. Directly
    targets the v9 `loss_align -> 0` projector-collapse (which came from
    applying alignment from step 0).
  - `ckpt_every=250` kill-safe checkpoints; `max_pretrain_seconds` 8.5h->7.0h
    safety cap. Legacy epoch loop retained (`max_iters=None`).
- **Refactor:** extracted `_compute_batch_loss` / `_infinite` helpers so the
  epoch and iteration loops compute identical losses. New CLI flags
  `--max-iters/--grad-accum/--align-frac/--ckpt-every`. Cells 2/11/12 re-synced;
  all cells parse.
- **Known limit (documented, not faked):** one T4 session caps images-seen at
  ~80-85k; CDPM's full 50k-updates/800k-images budget (~65 h here) needs
  multi-session checkpoint/optimiser-state resume. Next lever.
- **Results + sizing BUG:** ISBI2015 25-shot (mm) finetune pretrained
  **MRE 2.671 / SDR@2mm 48.59%** (v9 2.765 / 49.87); frozen pretrained
  **2.763 / 48.57%** (v9 2.935 / 45.56) -- frozen representation improved with
  ~10x fewer updates. **Still not beating CDPM (1.54 mm).** BUT the alignment
  phase **never ran**: the 7.0 h `max_pretrain_seconds` cap fired at iter 3827,
  before `align_start=4050`, because actual throughput was 6.57 s/update (not
  the estimated 4.7). So v10 = pure flow matching, alignment OFF -> the
  frozen-probe gain is attributable to eff-batch-16 flow optimisation alone.
  Fixed in v11 (resize budget to measured throughput).

---

## v9 (2026-08-10)

- **Backbone scale-up (main lever vs CDPM capacity gap):**
  `ConditionalUNetVelocityField` rebuilt to base_channels 32→64, a **third**
  downsampling level (H→H/8), and **bottleneck self-attention** (4 heads).
  Params ~0.9M → **9.6M**. Exposed feature keys unchanged; new
  `feature_channels()` method is the single source of truth for dims, so
  `cfm_pretrain` and the probe no longer hard-code them.
- **Sub-pixel landmark localisation:** `heatmaps_to_coords` now does CDPM-style
  quadratic (parabolic) sub-pixel refinement of the argmax (0.42→0.02 px on a
  clean peak; ~0.3–0.4 mm of MRE at 256px). Eval-only, no retraining.
- **§7 sampling upgraded:** self-contained explicit-Euler sampler (dropped
  `FlowSampler`) with **classifier-free guidance** conditioned on the chest
  class (`v = v_u + w·(v_c−v_u)`, w=2.0); grid shows unconditional vs chest+CFG.
  Auto-detects `base_channels`/`num_classes` from the checkpoint; 100→200 steps.
- **More pretraining compute + kill-safe training:** `epochs_pretrain` 30→80;
  checkpoint saved **every epoch** and a `max_pretrain_seconds` (8.5h) soft
  budget stops cleanly before Kaggle's 9h limit. `batch_size` 4→2 so the larger
  4-pass alignment backbone fits a T4 16GB.
- Old 32-ch checkpoints will NOT load into the new backbone — a full re-pretrain
  is required. All three `%%writefile` cells re-synced; all cells parse.
- **Results (run completed, no errors):** ISBI2015 25-shot (mm) â€”
  finetune pretrained **MRE 2.765 / SDR@2mm 49.87%** vs random 3.153 / 41.09%;
  frozen pretrained **MRE 2.935 / SDR@2mm 45.56%** vs random 6.829 / 34.88%.
  Pretraining helps under both protocols (largest under the frozen linear-probe:
  âˆ’3.89 mm MRE, +10.67 SDR). Chest real-landmark 25-shot (px): pretrained
  MRE 7.41 vs random 8.06. **CDPM not beaten yet** (paper 25-shot ref:
  1.54 mm MRE / 77.52% SDR@2mm; controls match, so the gap is genuine).
- **Â§7 samples (qualitative):** CFG works â€” unconditional row is mixed (one
  clearly out-of-distribution lateral cephalogram leaking from the ISBI half of
  the combined dataset, plus one distorted chest), while chest+CFG (w=2.0) gives
  consistent frontal-chest anatomy (ribs, lung fields, clavicles) across all 4.
  The v8 structureless-"blob" problem is resolved: the scaled-up 9.6M backbone
  after 80-epoch pretraining now produces recognizable chest anatomy.

---

## v8 (2026-08-09)

- **Flow-matching timestep bug fix (generation quality):** the flow loss now
  samples `t` uniformly over the FULL `[0, 1]` instead of only the mid-range
  `[0.25, 0.75]`. The sampler integrates the ODE over all of `[0, 1]`, so the
  old window left the noise→layout (t≈0) and final-sharpening (t≈1) parts of
  every trajectory unsupervised — the cause of the structureless "blob" samples
  in §7. Added an `align_t_window` flag (default `False`) to restore the old
  mid-range window for diagnostics only.
- **New §6.6 — qualitative landmark predictions:** overlays the fine-tuned
  probe's predictions (red ×) vs ground truth (green ○) with per-point error
  lines on held-out test images. Added `build_eval_split` (deterministic split
  reused from `run_probe`) and `visualize_predictions` to the probe module.
- **New §6.7 — frozen linear-probe:** runs ISBI2015 with the backbone frozen
  (head-only) for pretrained vs random-init, isolating representation quality;
  prints a 2×2 (frozen vs fine-tune) × (pretrained vs random) table + curves.
- **Probe objects retained:** cells for the real-landmark and ISBI runs now keep
  `probe_pre_real` / `probe_pre_isbi` (previously discarded) for reuse by §6.6.
- `epochs_pretrain` unchanged at 30 (deliberately, to isolate the timestep fix).

## v7 (2026-08-09)

- **CDPM-aligned downstream probe:** rewrote the probe to match CDPM's cephalo
  protocol — spatial-softmax + NLL heatmap loss (not background-dominated MSE),
  full fine-tune (`freeze_backbone=False`), AdamW + weight decay, 200 epochs with
  early stopping (patience 15) on test MRE; `loss="mse"` kept as a diagnostic.
- **Sampling `state_dict` fix (§7):** rebuild the generator with `num_classes`
  auto-detected from the checkpoint (`class_embed.weight` shape), fixing the v6
  `Error(s) in loading state_dict … Unexpected key class_embed.weight` / stem
  channel mismatch crash.
- **Twin-axis loss plot:** `loss_align` plotted on a second y-axis so it is
  visible against `loss_flow`.
- **Removed the circular chest-auto pseudo-landmark probe run** (§5) and pruned
  the resulting dead code (`auto_keypoints`, `_SOBEL_X`, `augment_view`, the
  `chest-auto` dataset branch, the `--grid` arg and `grid` config knob).
- First clean downstream numbers recorded (see `SUMMARY.md`): ISBI2015 25-shot
  MRE 4.86 mm (pretrained) vs 4.92 mm (random) — weak transfer, later traced to
  the full-fine-tune washout and the flow timestep bug fixed in v8.

## v6 (2026-08-09)

- **Image size 256** (was 128) to match CDPM-Align exactly.
- **Two-timestep delta-alignment** with dataset-index class conditioning
  (`num_classes=3`; index 0 = null/uncond token): `Δh = f(x_t,t,y) − f(x_t,t,∅)`
  compared across two independently sampled timesteps, making `loss_align`
  genuinely nonzero and trainable.
- Known issue in this version: the §7 sampling cell crashed on `state_dict` load
  (num_classes mismatch); the checkpoint still saved fine. Fixed in v7.

## v5 (best-effort reconstruction)

- Switched the alignment loss from the (incorrect) two-augmented-views scheme to
  the CDPM-faithful **two-timesteps** formulation (paper §2.2). `loss_align`
  magnitude from a clean run was not read back at the time.

## v4 (best-effort reconstruction)

- Attempted to fix the `loss_align == 0` collapse with `augment_view`
  (horizontal flip + random-resized crop + photometric jitter). This did **not**
  work — `loss_align` stayed `0.0000` from step 0 — because the guidance delta is
  global-average-pooled (flip-invariant). Superseded by the two-timestep fix.

## v1–v3 (best-effort reconstruction)

- Initial PoC scaffolding: clone `deltaflow`, inject the conditional UNet
  velocity-field backbone, write & run the CFM + delta-alignment pretraining
  module on Shenzhen chest X-rays, and a first landmark-detection probe. Early
  iterations exhibited `loss_align ≈ 0` (same-timestep / augmented-view design),
  which drove the fixes in v4–v6.

> Note: entries for v1–v5 are reconstructed from `SUMMARY.md` and may be
> approximate; v6 onward are recorded at push time.
