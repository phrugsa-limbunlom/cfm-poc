# Copilot instructions

## Command approvals
- **Do not ask for approval on routine commands.** Run read/inspect/build/test/push
  commands (e.g. `Get-ChildItem`, `git status`, `kaggle kernels ...`, running Python
  scripts, editing files) autonomously.
- **Only ask for approval on critical actions you are not confident about**, such as:
  - deleting or overwriting files that aren't obviously safe to lose,
  - anything that could leak secrets/credentials,
  - destructive or irreversible operations.

## Project context
- This project is developed locally and executed on Kaggle via the Kaggle CLI.
- See `KAGGLE.md` for the full push/run workflow.
- After a `kaggle kernels push`, do NOT poll or check the run status
  (`kaggle kernels status`/output). The user checks progress manually.
- Keep `kernel-metadata.json` free of `id_no` and pinned `docker_image` (they cause
  `500` errors on push).
- The local file tree mirrors the Kaggle runtime layout (files land in the cloned
  `deltaflow` repo): `deltaflow/models/conditional_unet.py`, and `cfm_pretrain.py`,
  `probe_landmark_detection.py`, `prepare_shenzhen_landmarks.py` at the repo root.

## Documentation / explanations
- Maintain a running `SUMMARY.md` at the repo root as the author's reference.
- Whenever you investigate an issue, diagnose a bug, or make a non-trivial
  change, **iteratively append** a clear explanation to `SUMMARY.md` (what the
  problem was, the root cause with evidence, and the fix). Add to it as an
  ongoing log — do not overwrite prior entries.
- Maintain a per-version changelog in `VERSIONS.md` at the repo root. **After
  every `kaggle kernels push`, add a new dated entry** (newest first) with a
  short bullet list of what changed in that kernel version. Keep it append-only —
  never rewrite past version entries.

## Who you are working with (researcher persona)
- The author is a **DPhil researcher**. The research goal is to **adapt a
  flow-matching (rectified-flow / continuous-t) generative backbone in place of
  CDPM-Align's DDPM noise-prediction backbone, and beat the evaluation metrics
  reported in the CDPM paper** (`CDPM.pdf`, Di Via et al., "CDPM-Align").
- Treat every task as part of that research programme: prioritise scientific
  correctness and faithful reproduction of the paper's method over quick hacks.
  When the code diverges from the paper, flag it and explain the implication.

## How to do research (think like an experimentalist)
- **State the hypothesis** before coding: what do we expect flow matching to
  change, and why. Keep it falsifiable.
- **Identify the variables explicitly** for every experiment:
  - *Independent variable (what we change):* the generative backbone /
    training objective — DDPM noise prediction → flow-matching velocity field
    (and any change deliberately under study).
  - *Dependent variable (the answer we want):* the paper's evaluation metrics
    — **MRE** (Mean Radial Error, lower is better), **SDR** (Success Detection
    Rate at the paper's radius thresholds, higher is better), and **ERE**
    (Expected Radial Error / uncertainty). These are what "beating CDPM" means.
  - *Controlled variables (must be held identical to enable a fair claim):*
    dataset(s) and splits, shot budget (10-/25-shot), image size, landmark set
    and evaluation unit (**millimetres**, using the correct `mm_per_pixel`, not
    raw pixels), the alignment mechanism (multi-scale guidance delta
    `Δh = h_cond − h_uncond`, GAP→MLP→ℓ2, cosine alignment across **two
    timesteps**), UNet capacity, feature levels `S`, `λ_align`, optimiser, and
    the number of pretraining/fine-tuning iterations. Change **one thing at a
    time**; if a control differs from the paper, the comparison is not valid —
    call it out.
- **Baselines & ablations:** always compare against the paper's reported
  numbers and against internal controls (e.g. random-init / CDPM-Scratch, and
  DDPM-backbone reproduction) so any gain is attributable to flow matching and
  not to an incidental change.
- **Fair evaluation:** evaluate on the same metric, same unit, same dataset and
  split as CDPM; report mean ± spread over seeds where feasible; never compare
  a pixel-space error against the paper's millimetre numbers.
- **Reproducibility:** fix seeds, log the exact config (all controlled
  variables) with each run, and record results and their interpretation in
  `SUMMARY.md`. A result only counts once it is verified and its controls are
  documented.
- **Be honest about negative/insufficient results:** if a change does not beat
  CDPM, say so and diagnose why, rather than presenting a partial or
  non-comparable win.
