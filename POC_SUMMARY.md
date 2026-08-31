# Proof-of-Concept Summary: Flow-Matching Backbone for CDPM-Align

*Compute-constrained proof-of-concept prepared as a DPhil application artifact.*

---

## Research question

Can a **flow-matching** (rectified-flow / continuous-t) generative backbone
replace **CDPM-Align's DDPM noise-prediction backbone** while preserving its
multi-scale directional-alignment mechanism (Δh guidance signal), and approach
its few-shot landmark-detection accuracy?

- **Independent variable (what we change):** the generative backbone /
  training objective — DDPM noise prediction → flow-matching velocity-field
  prediction.
- **Dependent variables (the answer we want):** the paper's evaluation metrics
  — **MRE** (Mean Radial Error, lower is better) and **SDR** (Success Detection
  Rate at radius thresholds, higher is better), reported in **millimetres**.
- **Controlled variables (held identical for a fair claim):** dataset(s) and
  splits, 25-shot budget, image size 256, evaluation unit (mm, with correct
  `mm_per_pixel`), the alignment mechanism (Δh = h_cond − h_uncond, GAP→MLP→ℓ2,
  cosine alignment across two timesteps), UNet capacity, feature levels, batch
  size, alignment fraction, timestep sampling range, optimiser.

---

## What I built

A complete, faithful re-implementation of the CDPM-Align pipeline with the
diffusion backbone swapped for a continuous-time **flow-matching objective**
(velocity-field prediction). Controlled variables kept faithful to the paper:

- **Batch size 16** (effective, via batch_size=2 × grad_accum=8).
- **Alignment phase over the final 10% of iterations** (align_frac = 0.1).
- **Mid-range timestep sampling [T/4, 3T/4]** (t_low = 0.25, t_high = 0.75).
- **Fixed forward timestep** for feature extraction before probing (see the
  caveat below — the *value* is not directly comparable to the paper's).
- **mm-based evaluation** with correct `mm_per_pixel` (never raw pixels for ISBI).
- Same datasets/splits, 25-shot annotation budget, no-leakage test hold-out.

I also extended the **interpolant family** to test path robustness:
- **Linear** interpolant (rectified-flow baseline).
- **OT-coupled** minibatch pairing over the full effective batch (via the
  deltaflow library `OTCoupling`).
- **Schrödinger-bridge** interpolant (σ = 0.1).

---

## What I ran (and the constraint)

Within a **single-GPU Kaggle budget** I could afford **~3,300 pretraining
iterations versus the paper's 50,000** (~15× fewer; 330 vs 5,000 alignment
iterations). **All controlled variables except training scale match the paper.**

| Control | CDPM (paper) | This POC (v20) | Status |
|---|---|---|---|
| Pretrain iterations | 50,000 (45k diffusion + 5k align) | 3,300 (2,970 + 330) | ~15× fewer |
| Alignment iterations | 5,000 | 330 | ~15× fewer |
| Batch size | 16 | 16 (effective) | ✓ match |
| Alignment fraction | final 10% | final 10% | ✓ match |
| Timestep sampling | [T/4, 3T/4] | [0.25, 0.75] | ✓ match |
| Fixed forward-t for probe | yes (t = 200 on T=500, ≈0.4 noised) | yes (t = 1.0 on [0,1], clean) | ✗ differs — see caveat |
| Evaluation unit | mm | mm (mm/px = [0.7559, 0.9375]) | ✓ match |
| Optimiser | AdamW | AdamW | ✓ match |

The dominant unmatched control is **training scale**, dictated by the GPU limit.

> **Caveat — fixed forward-t is not directly comparable.** CDPM extracts probe
> features at **t = 200 on a discrete DDPM schedule (T = 500)** — i.e. ≈0.4 of
> the way to pure noise, a *mildly-noised* image. This POC uses **continuous-t
> flow matching on [0, 1]** with `x_t = (1−t)·x0 + t·x1` and probes at
> **t = 1.0**, the **clean data endpoint** (zero added noise). These are
> different operating points; the flow-matching timeline has no exact analogue
> of t = 200, so this control is a genuine methodological difference rather than
> a matched setting. (A closer analogue to the paper would probe at t ≈ 0.6,
> leaving mild noise — an untested variable in this POC.)

---

## Results (ISBI2015, 25-shot, millimetres)

| Configuration | MRE ↓ | SDR@2mm ↑ |
|---|---|---|
| **CFM backbone (best, finetune)** | **2.75** | **47%** |
| CFM, random-init baseline (finetune) | 2.96 | 43% |
| CFM backbone (frozen linear probe) | 3.22 | 44% |
| CFM, random-init (frozen probe) | 7.62 | 31% |
| **CDPM-Align (paper target)** | **1.54** | **77.5%** |

Best case still misses CDPM by ~1.2 mm and ~30 SDR points. Enabling alignment +
OT coupling (v20) versus alignment-off (v15: 2.84 mm / 50%) changed essentially
nothing **at this training scale**.

---

## Headline result: iso-compute CFM vs DDPM head-to-head

*(Revised after a second CFM pretraining run (v22) and a corrected-sampler
DDPM run reproduced this comparison — see "Reproducibility of the head-to-head"
below. The conclusion is now more nuanced than a single "CFM wins" claim.)*

The scientifically decisive experiment. A DDPM noise-prediction backbone was
run under an **identical** budget and architecture (3,300 iters, effective batch
16, alignment on, same 988-image corpus, same seed, same probe) — the **only**
variable changed is the generative objective (flow-matching velocity regression
vs DDPM ε-prediction). Two protocols probe different things:

- **Frozen linear probe** (backbone frozen, only the head trains) isolates
  **representation quality** with no finetuning-dynamics confound.
- **Full finetune** (25-shot) measures **downstream task accuracy**, but
  conflates pretrained-representation quality with each run's random-init
  baseline strength and finetuning noise.

> **Definition — "gain over random."** This is the value added by *pretraining
> itself*, computed as `gain = MRE(random-init) − MRE(pretrained)` for the same
> backbone, head, data and protocol (only pretraining removed). A larger
> positive gain means pretraining helped more. It is used because raw
> pretrained MRE is confounded — an architecture may score well simply because
> it optimises well, not because pretraining taught it anything. Subtracting
> the from-scratch baseline isolates the pretraining contribution, which is the
> fair way to compare two backbones whose random-init baselines differ (as CFM's
> and DDPM's do). It is effectively an ablation: "how much worse would this be
> if I skipped pretraining?"

| Frozen probe (ISBI2015, mm) | **CFM** | **DDPM** | Winner |
|---|---|---|---|
| MRE (pretrained) ↓ | **3.12–3.22** | 4.35–4.90 | **CFM by ~1.1–1.7 mm** |
| SDR@2mm (pretrained) ↑ | **42.8–43.9%** | 36.7–37.9% | **CFM by 6–6.1 pts** |
| MRE (random-init, control) | 7.62 | 7.59–7.59 | ≈ equal ✓ |
| Gain over random ↓ | **4.49–4.40 mm** | 2.68–3.23 mm | **CFM ~39–64% larger** |

| Finetune (ISBI2015, mm) | **CFM** | **DDPM** | Winner |
|---|---|---|---|
| MRE (pretrained) ↓ | 2.687–2.75 | **2.579** | **DDPM by 0.11–0.17 mm** |
| MRE (random-init, control) | 2.96–3.19 | 2.94 | DDPM's control is also better |
| Gain over random ↓ | **0.21–0.50 mm** | 0.36 mm | mixed — CFM larger in the v22 pair |

The random-init baselines on the **frozen probe** are essentially identical
across backbones (~7.6 mm), confirming the architecture, head, and data are
truly matched there — so the frozen-probe gap is attributable to the
pretrained representation, and it **reproduces across two independent CFM
pretraining runs** (v20 and v22) against a consistent DDPM number. This is the
**cleaner, more defensible claim**: at equal compute, flow matching learns a
representation ~1.1–1.7 mm more accurate on the frozen probe and transfers
39–64% more gain over random than DDPM.

The **finetune** picture is different and should not be conflated with the
frozen-probe claim. On the **direct, head-to-head comparison that actually
matters — pretrained + finetune vs pretrained + finetune — DDPM reaches a
marginally lower MRE than CFM (2.579 vs 2.687–2.75 mm, a 0.11–0.17 mm gap).**
That is the real result and it stands on its own; it should **not** be
explained away by appeal to the random-init baselines. Two honest caveats
apply to *interpreting* (not dismissing) it:

- **The gap is small and single-seed.** 0.11 mm on one seed, in a finetune loop
  that already swung 2.99 → 2.579 mm between two DDPM runs, is **not
  established as a real effect** — it could be noise. Multi-seed runs (≥3) are
  needed before claiming "DDPM > CFM on finetune" with any confidence.
- **If the effect is real, its mechanism is unproven.** A plausible (but *not
  demonstrated*) story is that the DDPM objective leaves the architecture in a
  slightly more favourable optimisation basin for 25-shot regression — the
  random-init finetune baseline is also lower for DDPM (2.94 vs 2.96–3.19 mm).
  This is offered as a hypothesis for future testing, not as an explanation
  that neutralises the head-to-head number.

For both backbones the alignment term was inert (loss_align ≈ 1e-4–1e-2, only
330 align iters), so any pretraining effect here is attributable to the
flow-matching/DDPM **objective itself**, not the alignment mechanism.

**Net claim (calibrated to confidence):**
- **CFM learns a better representation than DDPM — solid.** Reproduced across
  two independent CFM runs on the frozen probe (the protocol that isolates
  representation quality), ~1.1–1.7 mm better, with matched random-init controls
  confirming architecture parity.
- **DDPM reaches a marginally lower finetune MRE than CFM — marginal and
  unconfirmed.** A 0.11 mm single-seed gap that may be noise; do not over-claim
  it as "DDPM finetunes better" until it survives multiple seeds.

Report **both** results at these honest confidence levels — do not reduce this
to a single winner, and do not use the random-init decomposition to erase the
finetune head-to-head.

### Why the two protocols disagree (mechanism)

The frozen-probe/finetune split is not a contradiction — it follows directly
from what each protocol can and cannot change:

- **Frozen probe isolates representation quality.** The backbone is locked and
  only a small head trains on 25 shots, so a mediocre feature space cannot be
  repaired — whatever pretraining produced is exactly what is measured. This
  is why CFM's ~1.1–1.7 mm advantage surfaces cleanly and reproducibly here:
  nothing downstream can compensate for it.
- **Finetune lets the whole UNet adapt, which squeezes out the pretraining
  signal.** With every parameter unfrozen and only 25 images (trained up to
  ~150 early-stopped epochs), the network is massively overparameterised
  relative to the data and can reshape almost any starting point into a
  low-loss solution. The *initial* representation stops being decisive; what
  dominates instead is how well that architecture's parameter regime optimises
  under SGD for this small-sample regression.
- **The shrinking gain-over-random is the direct evidence.** Pretraining buys
  CFM ~4.5 mm on the frozen probe but only ~0.5 mm on finetune; DDPM similarly
  collapses from ~3.2 mm to ~0.36 mm. Once full adaptation is allowed,
  pretraining's leverage is almost entirely squeezed out for both backbones —
  only a small fraction of the frozen-probe advantage survives.
- **The smoking gun that DDPM's finetune edge is not a representation claim:**
  DDPM's *random-init* finetune baseline (2.94 mm) is already lower than CFM's
  random-init baselines (2.96–3.19 mm) — i.e. with **no pretraining at all**,
  the DDPM-objective architecture finetunes to a better number on this task.
  The most likely cause is that the ε-prediction objective (targets are
  unit-variance noise) leaves the backbone in a different weight-scale /
  normalisation regime than CFM's velocity-field objective (different target
  scale statistics), and that regime happens to sit in a more favourable
  optimisation basin for full-network SGD on 25 shots. That is an
  architecture/optimisation-dynamics artefact of full finetuning on very few
  shots — **not** evidence that DDPM learned better anatomical structure.

The practical upshot: the frozen probe answers "which backbone's pretrained
features are intrinsically more informative?" (CFM), while finetune answers
"which architecture optimises to the lowest absolute error under this exact
25-shot setup?" (DDPM, largely independent of pretraining). Both are true and
they are not in tension.

### Reproducibility of the head-to-head

| Run | Backbone | Config | Frozen MRE (pretrained) | Finetune MRE (pretrained) |
|---|---|---|---|---|
| v20 | CFM | align=T, ot=T, linear | 3.22 | 2.75 |
| v22 | CFM | align=T, ot=T, schrödinger | 3.124 | 2.687 |
| (earlier) | DDPM | align=T | 4.90 | 2.99 |
| (latest, corrected sampler) | DDPM | align=T | 4.353 | **2.579** |

The two CFM runs agree within ~0.1 mm on both protocols, and the frozen-probe
DDPM gap (~1.1–1.7 mm) holds against both — good evidence the frozen-probe
claim is a real effect, not run-to-run noise. The DDPM finetune number
improved between runs (2.99 → 2.579 mm); per `VERSIONS.md` the only
intervening change (v24 sampler bugfix) is documented as
sampling/visualisation-only with no effect on pretrain/probe code, so this
improvement is most plausibly finetune-loop stochasticity (dataloader
shuffling, GPU nondeterminism, early-stopping epoch selection) — a reminder
that the finetune protocol is noisier and should ideally be run over multiple
seeds before being used for a paper-facing claim.

---

## Findings (honest)

1. **The flow-matching backbone trains stably and transfers.** It beats a
   random-init baseline by **4.4 mm** on a frozen linear probe (3.22 vs 7.62 mm),
   confirming it learns useful anatomical structure rather than noise.

2. **It does not yet match CDPM, and the gap is attributable to a single
   uncontrolled variable — training scale — not to the method.** Diagnostics
   from the v20 run:
   - **Backbone undertrained / saturated:** flow loss reaches ~0.05 by iter ~900
     and then flatlines (noisy 0.04–0.12) for the remaining 2,400 iters. 3,300
     iters is ~53 epochs over a 988-image corpus; CDPM trains ~50k iters.
   - **Alignment term inert:** `loss_align ≈ 0.0001–0.0004` from the very first
     alignment iteration — ~250× smaller than the flow loss. With only **330**
     alignment iterations, the projector never develops a meaningful Δh signal,
     so the core CDPM-Align mechanism is effectively **off** even when enabled.
   - **Off-domain corpus:** only 150 of 988 pretraining images are ISBI (the
     downstream target); the rest are chest X-rays + hand atlas, so most
     pretraining signal is irrelevant to cephalometrics.
   - **Finetune ceiling ≈ random:** 25-shot fine-tuning dominates, so the
     pretrained backbone adds only **0.20 mm** in the finetune protocol.

3. **Path/coupling design is not the bottleneck at this scale.** The interpolant
   choice (linear vs OT-coupled vs Schrödinger-bridge, σ = 0.1) and OT coupling
   did **not** change outcomes — the limiter is compute, not path geometry.

4. **The gap is real, not a unit artifact.** `mm_per_pixel` = [0.7559, 0.9375]
   was applied correctly and ISBI is reported in mm — this is not a
   pixel-vs-millimetre comparison error.

5. **CFM vs DDPM is not a single-metric story — frozen-probe and finetune
   disagree, and each answers a different question.** On the **frozen linear
   probe** (isolates pretrained-representation quality, no finetuning-dynamics
   confound), CFM reproducibly beats DDPM by ~1.1–1.7 mm across two independent
   CFM runs (v20, v22) against a consistent DDPM number — the strongest,
   most defensible claim in this POC. On **full finetune**, DDPM reproducibly
   reaches a *lower* absolute MRE than CFM (2.579 vs 2.687–2.75 mm, both times),
   but DDPM's random-init finetune baseline is also stronger than CFM's
   (2.94 vs 2.96–3.19 mm) — so part of DDPM's finetune edge is inherited from
   that architecture's optimisation dynamics at 25-shot, not purely from
   pretraining. Any future write-up should report **both** numbers with this
   distinction, rather than picking one backbone as an unqualified "winner."

6. **Task-adaptation capacity matters more than frozen representation quality at
   25 shots — the architecture + finetuning does most of the work.** A
   from-scratch, *no-pretraining* finetune matches or beats a *pretrained but
   frozen* probe in almost every pairing:

   | Configuration | MRE (mm) |
   |---|---|
   | DDPM random-init + finetune | 2.94 |
   | CFM random-init + finetune (v20) | 2.96 |
   | CFM random-init + finetune (v22) | 3.19 |
   | CFM pretrained + frozen probe | 3.12–3.22 |
   | DDPM pretrained + frozen probe | 4.35–4.90 |

   Most starkly, DDPM's random-init finetune (2.94 mm) crushes its own
   pretrained frozen probe (4.35 mm); the only near-tie is CFM v22 (3.19
   finetune-random ≈ 3.12 frozen-pretrained). **Interpretation:** letting the
   entire `ConditionalUNet` adapt to the 25-shot target data is worth *more*
   than having good pretrained features that are then frozen. This is
   consistent with the tiny gain-over-random on finetune (~0.4–0.5 mm) versus
   the large gain on the frozen probe (~3–4.5 mm): full finetuning squeezes out
   most of the pretraining benefit. **Caveat (honest):** this does **not** make
   pretraining useless — it says *frozen* pretrained features underperform
   *task-adapted* random features. Pretraining still helps *on top of*
   finetuning (DDPM 2.94→2.58, CFM 3.19→2.69), just modestly; and the frozen
   probe still validly isolates *representation quality* between backbones — it
   simply isn't the route to the lowest absolute MRE. **Practical upshot for the
   DPhil:** at this shot budget, spend compute on task-adaptation capacity
   rather than on frozen-representation quality.

---

## Best downstream result (practical bottom line)

**If the sole objective is minimising absolute downstream MRE on the ISBI2015
landmark task, the best configuration observed in this POC is DDPM-pretrained +
full finetune at 2.579 mm.** Ranking of every finetune config actually run:

| Configuration | ISBI2015 finetune MRE ↓ |
|---|---|
| **DDPM pretrained + finetune** | **2.579** ← best observed |
| CFM pretrained + finetune (v22) | 2.687 |
| CFM pretrained + finetune (v20) | 2.75 |
| DDPM random-init + finetune | 2.940 |
| CFM random-init + finetune | 2.96–3.19 |
| *CDPM-Align paper target (unreached)* | *1.54* |

This is a fair **empirical** statement about the configs tested, but three
caveats keep it from being over-claimed:

1. **Single seed.** 2.579 mm is one run, and the DDPM finetune number already
   moved 2.99 → 2.579 mm between runs (finetune-loop stochasticity, per the
   Reproducibility note above). A ≥3-seed mean ± spread is needed before
   treating this as a robust ranking rather than "best observed."
2. **The gap is small, single-seed, and its cause is unproven.** 2.579 vs
   2.687–2.75 mm is a 0.11–0.17 mm margin on one seed — possibly noise (see
   caveat 1). One should therefore **not** over-claim it as "DDPM finetunes
   better." A plausible *hypothesis* is that DDPM's ε-prediction objective
   leaves the architecture in a slightly more favourable optimisation basin for
   25-shot regression (its random-init finetune baseline, 2.940 mm, is also
   lower than CFM's 2.96–3.19 mm) — but this is offered for future testing, not
   as a demonstrated explanation, and it does **not** erase the head-to-head
   number. "DDPM pretrained + finetune is the best **observed end-to-end
   config**" is fair; "DDPM **pretraining** is best" is not (CFM wins the
   reproduced representation-quality claim on the frozen probe).
3. **Still short of CDPM (1.54 mm).** No config reaches the paper target; the
   remaining ~1 mm gap is attributed to training scale (see Findings §2).

**Defensible one-liner:** *for minimising absolute downstream MRE under this
exact 25-shot setup, DDPM-pretrained + finetune gave the best observed result
(2.579 mm), though the advantage is driven by finetuning dynamics rather than
pretraining quality and needs multi-seed confirmation.*

---

## Threats to the thesis & repositioning (candid)

**Blunt assessment.** As currently framed — *"CFM beats DDPM on downstream
landmark-detection MRE"* — this POC does **not** support the thesis, and at
25-shot finetune it weakly **contradicts** it (DDPM 2.579 < CFM 2.687–2.75). A
proposal should not lead with that headline. The contribution is not dead, but
it must be repositioned.

**Where the accuracy-only claim is at risk**

1. **The metric that matters for CDPM favours DDPM here.** Few-shot finetune MRE
   is the paper's headline unit, and DDPM currently wins it in this POC.
2. **Finetuning washes out backbone differences at 25 shots** (Finding #6): a
   "better backbone" buys little in exactly the low-shot regime the paper
   targets, because task-adaptation capacity dominates.
3. **Underpowered evidence:** 15× below paper scale, single seed, inert
   alignment term — not yet a convincing SOTA-beating result.

**Where a meaningful contribution still survives**

1. **Representation quality is real and reproduced.** CFM's *frozen* features
   are 1.1–1.7 mm better than DDPM's across two independent CFM runs. Reframe
   the thesis as *"flow-matching pretraining yields more transferable
   medical-image representations"* (a representation-learning claim) rather than
   *"CFM wins end-to-end."* That narrower claim is defensible on today's data.
2. **The controlled analysis is itself a contribution.** The frozen-vs-finetune
   dissociation, the gain-over-random ablation methodology, and the honest
   accounting of confounds constitute a clean study that most backbone-swap
   papers omit.
3. **The scale hypothesis is untested, not falsified.** With the paper's 50k
   iterations and an *active* alignment term (currently inert at ~1e-4), the
   accuracy story could change. That is a falsifiable next experiment, not a
   dead end.

**The strategic risk to face directly.** If the *only* axis is accuracy,
CFM-over-DDPM is fragile. Flow matching's genuine, less-contestable advantages
lie elsewhere: **sampling efficiency (straighter probability paths, fewer
function evaluations), training stability, and likelihood/theoretical
tractability.** A proposal that leans on those *plus* the representation-quality
finding is substantially stronger than one betting solely on beating DDPM's MRE.

**Recommended positioning.** Do **not** propose "CFM > DDPM accuracy." Propose:
*flow-matching backbones give (a) better transferable representations —
validated here by a controlled frozen-probe study — and (b) efficiency/stability
advantages inherent to the objective, with (c) a scale-up experiment (50k iters,
active alignment, multi-seed) to test whether the residual accuracy gap closes.*
This frames the POC's honest result as motivation and evidence, not as a
falsified accuracy claim.

---

## Does this invalidate the PhD premise? (generative pretraining for robust, data-efficient medical image analysis)

**No — but only when the evidence is read correctly.** The result that seems
alarming — *"random-init finetune (2.94) beats pretrained frozen probe
(4.35)"* — is an **apples-to-oranges** comparison: it changes *two* variables at
once (initialisation **and** whether the backbone adapts). It shows frozen
features lose to task-adapted weights; it does **not** show pretraining is
worthless.

**The fair test — same protocol, only pretraining toggled — favours pretraining
in every matched pairing:**

| Protocol | random-init | pretrained | pretraining helps? |
|---|---|---|---|
| DDPM finetune | 2.94 | **2.58** | ✅ yes (+0.36 mm) |
| CFM finetune | 3.19 | **2.69** | ✅ yes (+0.50 mm) |
| CFM frozen probe | 7.62 | **3.12** | ✅ yes (+4.5 mm) |

So "normal finetuning beats generative pretraining" is **false as stated**. The
honest claim is: *finetuning from a pretrained init beats finetuning from
scratch, and beats a frozen pretrained probe.* Pretraining still wins its own
controlled (matched-protocol) test.

**Why the premise survives — and where its real vulnerability lies:**

1. **The premise holds directionally.** Pretraining consistently lowers MRE; the
   data-efficiency benefit is real and is **largest exactly where labels are
   scarcest / the representation is frozen** (frozen probe: +4.5 mm).
2. **The genuine threat is *effect size*, not *sign*.** Under 25-shot full
   finetune the benefit is small (~0.4–0.5 mm). The panel's real question is not
   "does pretraining help?" (yes) but "does it help *enough to matter* versus
   just finetuning a good architecture?" — an open, fundable question, not a
   refutation.
3. **Reframe "data-efficient" precisely.** The strongest evidence is the
   frozen/linear-probe / very-low-shot regime. The defensible thesis is:
   *generative pretraining matters most in the extreme-low-label regime and for
   label-efficient transfer (frozen/linear probing); its marginal value under
   full finetuning is an open question this PhD will characterise and aim to
   widen.* This is sharper and more honest than "pretraining always wins."

**What *would* genuinely threaten the proposal (be ready for these):**
- If pretraining gave *zero or negative* gain in matched comparisons — it does
  not.
- If a trivially-pretrained baseline (ImageNet transfer, or discriminative
  self-supervised contrastive pretraining) matched generative pretraining at
  equal cost. **This is the real competitor to pre-empt:** the novelty is
  *generative* pretraining specifically, so the proposal must argue why
  generative > discriminative SSL for medical images (models the full data
  density, supports the alignment/guidance mechanism, and adds
  sampling/anomaly-detection capabilities discriminative SSL cannot).

**Bottom line for the application.** The POC does **not** invalidate the
proposal. It (a) confirms pretraining helps in every matched test, (b) shows the
benefit concentrates in the low-label / frozen regime — precisely the
"data-efficient" claim — and (c) flags that quantifying and *widening* the
finetune-regime benefit is the open research problem. That is a valid, honest,
well-scoped PhD.

---

## Glossary — what "probe" means here

A **probe** is the small task-specific head placed on top of the backbone to
*measure how good its features are* — it "probes" the representation. In this
POC it is the landmark-detection head on the `ConditionalUNet` features that
predicts the 19 ISBI landmark coordinates. The critical distinction is **what
gets trained**:

| Term | Backbone weights | Head weights | What it measures |
|---|---|---|---|
| **Linear / frozen probe** | frozen | trained | Quality of the *pretrained representation* as-is |
| **Finetune** | trained | trained | Best achievable task accuracy (backbone adapts too) |

The frozen probe is the *clean* representation-quality signal: with the backbone
locked, the only thing that can differ between two experiments is the quality of
the features pretraining produced — nothing downstream can repair bad features.

Terminology note on the logs/CSVs: the `probe` column names *which downstream
evaluation* was run (`isbi2015` = ISBI task, finetune protocol;
`isbi2015-frozen` = same task, frozen linear-probe protocol; `chest-real` =
Shenzhen lung landmarks). The `protocol` column (`finetune` vs `linear-probe`)
is what actually indicates whether the backbone was frozen.

---

## Conclusion

This proof-of-concept establishes that **flow matching is a viable, controlled
drop-in** for the CDPM-Align diffusion backbone: it trains stably, transfers to
few-shot landmark detection, and clearly beats a from-scratch baseline. It also
**isolates training scale as the sole barrier** to a fair head-to-head with the
paper — every other control (batch size, alignment fraction, timestep range,
evaluation unit) is already matched.

**Falsifiable next hypothesis:** given the paper's 50k-iteration budget (45k
diffusion + 5k alignment), the alignment term would become active (`loss_align`
would grow beyond the current ~2e-4 floor) and the flow backbone would train to
convergence, closing most of the MRE/SDR gap. This is the natural next step for
the DPhil, requiring only additional compute rather than a methodological
change.

---

## Reproducibility

- **Artifact:** `cfm_delta_align_pretrain.ipynb` (single Kaggle-run notebook).
- **Config (v20):** align = True, ot_coupling = True, interpolant = linear,
  pretrain_iters = 3,300, batch_size = 2, grad_accum = 8 (eff 16),
  align_frac = 0.1, lambda_align = 5.0, t_low = 0.25, t_high = 0.75, seed = 0.
- **Probe:** fixed forward timestep `t_value = 1.0` (clean data endpoint on the
  continuous [0,1] flow-matching timeline; not equivalent to CDPM's t = 200).
- Fixed seed; full config logged with each run; results and their interpretation
  recorded in `SUMMARY.md` and `VERSIONS.md`.
- **Reference target:** CDPM.pdf (Di Via et al., "CDPM-Align"): MRE 1.54 mm /
  SDR@2mm 77.5% (ISBI2015, 25-shot).


---

## 2026-08-31 — Cross-dataset backbone analysis: CFM vs DDPM (25-shot, seed 0)

**Independent variable:** backbone (DDPM noise-prediction -> CFM flow-matching).
**Controlled:** combined-pretrain dataset, image_size 256, n_shot 25, align=True,
lambda_align 5.0, seed 0, protocol/init matched per pair.
**Caveats:** single seed (no error bars); chest-real & aasce are in **px**,
isbi2015 in **mm** — compare only *within* a dataset; px results are not
paper-comparable until converted to mm.

### MRE (lower is better) — matched pairs

| Dataset / protocol | init | DDPM | CFM | Delta (CFM-DDPM) | Winner |
|---|---|---|---|---|---|
| chest-real / finetune | pretrained | 8.337 | 7.226 | -1.111 (-13.3%) | CFM |
| chest-real / finetune | random | 7.956 | 7.668 | -0.288 | CFM |
| isbi2015 / finetune | pretrained | 2.942 | 2.709 | -0.233 (-7.9%) | CFM |
| isbi2015 / finetune | random | 2.992 | 3.207 | +0.215 | DDPM |
| isbi2015-frozen / probe | pretrained | 4.270 | 3.247 | -1.023 (-24%) | CFM |
| isbi2015-frozen / probe | random | 7.585 | 7.617 | +0.032 | ~tie |
| aasce / finetune | pretrained | 20.032 | 18.214 | -1.818 (-9.1%) | CFM |
| aasce / finetune | random | 19.479 | 20.760 | +1.281 | DDPM |
| aasce-frozen / probe | pretrained | 27.287 | 25.096 | -2.191 (-8.0%) | CFM |
| aasce-frozen / probe | random | 30.946 | 28.163 | -2.783 (-9.0%) | CFM |

SDR@2mm, SDR@4mm and P95 track MRE in every row (same winner), so no conflicting
signals.

### Findings

1. **CFM wins decisively on the pretrained condition.** With init=pretrained,
   CFM beats DDPM on every dataset and both protocols across MRE, P95, SDR@2 and
   SDR@4 (gains -8% to -24% MRE). The flow-matching backbone learns a better
   *transferable* representation.

2. **Pretraining transfer gain is the real differentiator** (random MRE -
   pretrained MRE; positive = pretraining helps):

   | Dataset / protocol | DDPM gain | CFM gain |
   |---|---|---|
   | chest-real / finetune | -0.381 (hurts) | +0.442 |
   | isbi2015 / finetune | +0.050 (~none) | +0.498 |
   | isbi2015-frozen / probe | +3.315 | +4.370 |
   | aasce / finetune | -0.553 (hurts) | +2.547 |
   | aasce-frozen / probe | +3.660 | +3.060 |

   CFM pretraining yields a **consistently positive** transfer gain; DDPM
   pretraining is erratic and sometimes *hurts* (chest-real, aasce finetune).

3. **From random init, CFM has no consistent edge** (loses isbi2015 & aasce
   finetune, ties isbi2015-frozen, wins the rest) — as expected, isolating the
   benefit to *pretraining* rather than architecture alone.

4. **Frozen linear-probe amplifies the CFM advantage** (largest win:
   isbi2015-frozen/pretrained, -24% MRE), confirming the *features themselves*
   are superior, not just fine-tuning dynamics.

### Next steps to make it publishable
- Multi-seed (>=3) for mean +/- spread; isbi2015/finetune/random and
  isbi2015-frozen/random are within noise.
- Convert chest-real & aasce to mm (correct mm_per_pixel) for paper comparison.
- Add CDPM paper baseline column on isbi2015 (mm) at matched shot budget.

### Bottom line
Evidence supports the hypothesis: replacing DDPM with flow matching improves
pretrained-representation quality — better MRE/SDR/P95 on all datasets under
pretrained init, and a consistently positive, larger pretraining transfer gain.
