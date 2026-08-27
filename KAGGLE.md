# Running this notebook on Kaggle from your local machine

This project is developed locally and executed on Kaggle (for the free GPU). We
use the [Kaggle CLI](https://github.com/Kaggle/kaggle-cli) — the official
**`kaggle`** PyPI package, **v2.2.4** — to push the notebook and its settings
straight from this folder — no copy‑pasting into the web editor.

---

## 0. How the local → Kaggle loop works (TL;DR)

You edit locally, then push; the code actually runs on Kaggle's GPU:

1. **Edit locally** — change the notebook (`cfm_delta_align_pretrain.ipynb`) or
   `kernel-metadata.json` on your machine.
2. **Push** — `kaggle kernels push -p .` uploads the notebook + settings and
   triggers a fresh **batch run** on Kaggle's GPU.
3. **Monitor** — `kaggle kernels status ...` polls `RUNNING` →
   `COMPLETE` / `ERROR`.
4. **Pull results** — `kaggle kernels output ... -p .\out` downloads the
   produced files + run log. If it errored, read the log, fix locally, push
   again.

Two things to keep in mind:

- **Only `.ipynb` + `kernel-metadata.json` get pushed.** The local `.py` files
  under `deltaflow\` are just reference copies — the notebook regenerates them
  at runtime via `%%writefile`, and it `git clone`s the real `deltaflow` repo on
  Kaggle.
- **`push` runs headless (batch).** It does not touch the interactive editor
  session, which has its own separate internet toggle (see §6).

---

## 1. Project layout

The folder mirrors the runtime layout the notebook builds on Kaggle (it clones
the [`deltaflow`](https://github.com/phrugsa-limbunlom/deltaflow) repo into
`/kaggle/working/deltaflow` and writes these files into that tree):

```
cfm_experiment/
├── cfm_delta_align_pretrain.ipynb   # the notebook that gets pushed
├── kernel-metadata.json             # tells the CLI how to run it (GPU, internet, ...)
├── KAGGLE.md                        # this guide
├── .gitignore                       # keeps credentials + out/ out of git
│
├── deltaflow/
│   └── models/
│       └── conditional_unet.py      # written by notebook cell 4 -> deltaflow/models/
├── cfm_pretrain.py                  # written by notebook cell 9  (repo root)
├── probe_landmark_detection.py      # written by notebook cell 13 (repo root)
└── prepare_shenzhen_landmarks.py    # written by notebook cell 17 (repo root)
```

> The `.py` files here are the source‑of‑truth versions. Inside the notebook the
> same code is emitted with `%%writefile` cells at the paths shown above, so the
> local tree and the Kaggle runtime tree match.

---

## 2. One‑time setup

### Install / update the CLI

Install (or upgrade to the latest) with pip:

```powershell
pip install --upgrade kaggle
kaggle --version   # this project was built/tested with: Kaggle CLI 2.2.4
```

> The CLI used throughout this guide is the official **`kaggle`** PyPI package
> (a.k.a. [kaggle-cli](https://github.com/Kaggle/kaggle-cli)), version **2.2.4**.
> Always run `pip install --upgrade kaggle` to pick up the newest release; the
> commands below are stable across recent versions.

### Authenticate

You can use **either** an API token file **or** a short‑lived access token.

**Option A — `kaggle.json` API token (recommended, long‑lived):**
1. Kaggle → your avatar → **Settings** → **API** → **Create New Token**.
2. This downloads `kaggle.json`. Move it to:
   ```
   C:\Users\<you>\.kaggle\kaggle.json
   ```

**Option B — access token (`KGAT_...`, short‑lived):**
Write the token to `C:\Users\<you>\.kaggle\access_token`:
```powershell
Set-Content -Path "$env:USERPROFILE\.kaggle\access_token" -Value "KGAT_xxxxxxxx" -NoNewline
```
> ⚠️ KGAT access tokens expire quickly (often within minutes). If commands start
> returning `Not found` / `403`, generate a fresh one. Prefer Option A.

Verify auth:
```powershell
kaggle kernels list -m   # lists your own kernels
```

---

## 3. `kernel-metadata.json`

This file controls how the CLI runs the notebook:

```jsonc
{
  "id": "phrugsalimbunlom/deltaflow-cfm-delta-alignment-pretraining-poc",
  "title": "DeltaFlow: CFM + Delta-Alignment Pretraining (PoC)",
  "code_file": "cfm_delta_align_pretrain.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,               // GPU on
  "enable_tpu": false,
  "enable_internet": true,          // needed for the git clone + dataset download
  "machine_shape": "NvidiaTeslaT4"  // GPU type
}
```

Notes learned the hard way:
- **Do not** add `id_no` or a pinned `docker_image` — stale values cause a
  `500 Server Error` on push. Keep them out.
- `enable_internet: true` here only applies to **batch** ("Save & Run All")
  runs. The **interactive editor** has a *separate* internet toggle (see §6).
- **Datasets must be pre-declared** in `dataset_sources`. In a batch/push run,
  `kagglehub.dataset_download(...)` fails with
  `New Datasets cannot be attached in non-interactive sessions`. Declaring the
  dataset here mounts it at `/kaggle/input/...`, and cell 7 uses that mount
  first (skipping the download). That's why
  `raddar/tuberculosis-chest-xrays-shenzhen` is listed above.

---

## 4. Push the notebook

From this folder:

```powershell
cd C:\Users\plimbunlom\source\repos\myproject\cfm_experiment
kaggle kernels push -p .
```

Expected output:
```
Kernel version N successfully pushed.  Please check progress at https://www.kaggle.com/code/...
```

---

## 5. Check status & get output

```powershell
# run status (queued / running / complete / error)
kaggle kernels status phrugsalimbunlom/deltaflow-cfm-delta-alignment-pretraining-poc

# download rendered output + any files the run produced
kaggle kernels output phrugsalimbunlom/deltaflow-cfm-delta-alignment-pretraining-poc -p .\out
```

> The **logs** endpoint (`kernels output` log portion) sometimes returns `403`
> via the API. If so, read the run log in the **Kaggle web UI** instead — it's
> the reliable place for logs.

Pull the latest server‑side metadata/code (e.g. after editing in the web UI):
```powershell
kaggle kernels pull phrugsalimbunlom/deltaflow-cfm-delta-alignment-pretraining-poc -p . -m
```

---

## 6. Interactive editor vs. batch run

| | Batch ("Save & Run All" / `kernels push`) | Interactive editor |
|---|---|---|
| Internet | controlled by `enable_internet` in metadata | **separate toggle**, defaults **OFF** |
| Best for | full reproducible runs | poking at cells |

If you open the notebook in the interactive editor and the `git clone` in
**cell 2** fails, turn internet on:

> **right sidebar → Notebook options → Internet: On** (account must be
> phone‑verified), then re‑run from **cell 2**.

Cell 2 is idempotent and re‑run‑safe: it clones only if missing and anchors the
working directory with an absolute path, so re‑running it won't break the later
`%%writefile deltaflow/...` cells. If internet is off it now fails with a clear
message instead of a cryptic `FileNotFoundError`.

---

## 7. Quick reference

```powershell
# install / update CLI (kaggle 2.2.4+) + auth
pip install --upgrade kaggle
kaggle --version
kaggle kernels list -m

# push this project
kaggle kernels push -p .

# status + output
kaggle kernels status phrugsalimbunlom/deltaflow-cfm-delta-alignment-pretraining-poc
kaggle kernels output phrugsalimbunlom/deltaflow-cfm-delta-alignment-pretraining-poc -p .\out
```

---

## 8. Verified run (v3)

Version **3** ran end‑to‑end on a Kaggle **T4** GPU in ~58 min and finished with
status `COMPLETE`. Getting there took two fixes:

| Version | Result | Fix applied |
|---|---|---|
| v1 / v2 | `ERROR` | `FileNotFoundError` (cell 2 `%cd` not re‑run‑safe) **and** `BackendError: New Datasets cannot be attached in non-interactive sessions` |
| **v3** | **`COMPLETE`** | made cell 2 idempotent (clone/anchor cwd by absolute path) **+** declared the dataset in `dataset_sources` |

Artifacts pulled into `.\out`:

- **`cfm_delta_align_backbone.pt`** — the trained CFM + delta‑alignment backbone.
- `deltaflow/Chest-xray-landmark-dataset/figs/landmarks.png`, `info.png` —
  landmark figures.
- `deltaflow/Chest-xray-landmark-dataset/clinical.csv` — landmark metadata.
- `deltaflow-cfm-delta-alignment-pretraining-poc.log` — full run log.

Both the training half (cell 10) and the evaluation/landmark probe half ran
cleanly in this batch run.
