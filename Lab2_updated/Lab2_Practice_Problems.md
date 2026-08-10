# COMP-4147 Lab 2 — Source Separation Practice Problems

**Materials:** [`Lab2_updated`](https://github.com/polinexian/COMP-4147-Audio-And-Speech-Processing/tree/jintian/Lab2_updated)

| Resource | Role |
|---|---|
| `Lab2 - Sound seperation.ipynb` | Classical methods, pretrained DL models, metrics, Demucs, DPRNN overview |
| `Lab2 - TasNet.ipynb` | Conv-TasNet architecture, SI-SNR / PIT training |
| `Dataset/` | Stereo speech mixtures, speaker tracks, music clips |
| `Images/` | Conv-TasNet, DPRNN, Demucs diagrams; hyperparameter table |
| `Conv-Tasnet-Customized.pth` | Customized Conv-TasNet checkpoint (use if assigned) |

**Environment tips:** Python 3 with `librosa`, `numpy`, `matplotlib`, `scikit-learn`, `torch`, `torchaudio`, `mir_eval`; for Demucs: `pip install demucs`. Google Colab is acceptable if GPU is needed.

**How to submit (if used as graded practice):** one notebook named `YourName_Lab2_Practice.ipynb` and a short report (`.docx` or markdown) with waveforms/spectrograms, metric tables, and written analysis.

---

## Learning objectives

After completing these problems you should be able to:

1. Apply ICA, RPCA, Conv-TasNet, and Demucs under the correct input assumptions (channels, sample rate, domain).
2. Visualize mixtures and separated sources in the time domain and as spectrograms.
3. Evaluate separation with SDR, SIR, SAR, and SI-SDR, and interpret the scores.
4. Explain the Conv-TasNet encoder–separator–decoder pipeline and permutation-invariant training (PIT).
5. Reason about failure cases (domain mismatch, number of sources, sample-rate mismatch).

---

## Problem 1 — ICA on stereo speech mixtures *(classical BSS)*

**Background.** Independent Component Analysis (ICA) treats multichannel observations as linear mixtures of statistically independent sources (cocktail-party setting). In the lab notebook, `FastICA` is applied to `mix_sample1.wav`.

**Tasks**

1. Load **two** stereo mixtures from `Dataset/`: `mix_sample2.wav` and `mix_sample3.wav` with `librosa.load(..., mono=False)` at the file’s native sample rate.
2. For each file:
   - Print shapes **before** and **after** the transpose required by ICA (`(C, T)` → `(T, C)`).
   - Run `sklearn.decomposition.FastICA` with `n_components=2`.
   - Listen to both estimated sources; plot original channels vs separated sources and their spectrograms (same layout as the lab notebook).
3. **Written analysis (short):**
   - Why does ICA require **at least as many channels as sources**?
   - What happens conceptually if you force `n_components=3` on a 2-channel file?
   - Compare separation quality across the three mixtures (`mix_sample1`–`3`). Which is easiest/hardest to separate by ear, and why might that be?

**Deliverables:** code + figures + ½–1 page analysis.

**Hints:** ICA does not use ground-truth speakers; quality is judged by listening and by spectrogram structure (e.g., one source quieter in high frequencies).

---

## Problem 2 — RPCA for singing-voice / accompaniment separation

**Background.** Robust PCA decomposes a magnitude spectrogram into a **low-rank** part (repetitive accompaniment) and a **sparse** part (often vocals / foreground events). The lab implements `RPCA` and `singing_voice_separation` on `Goodbye_Bolero.mp3`.

**Tasks**

1. Run the lab’s RPCA pipeline on **`Goodbye_Bolero.mp3`** and on **one other music file** of your choice: `Corine.mp3` or `ComfyCouches.mp3`.
   - Use mono audio, `sr=8000`, and a fixed duration (e.g. 8 s) for fair comparison.
2. Parameter study: keep all settings fixed except **`lmbda`** in `singing_voice_separation`. Try at least **three** values (e.g. `0.5`, `1.0`, `2.0`).
3. For each (file, λ) pair, save/plot:
   - time-domain waveforms of both outputs;
   - spectrograms of both outputs;
   - a brief note: which output sounds more like “vocals” vs “backing”?
4. **Written analysis:**
   - How does increasing λ change the sparse vs low-rank trade-off?
   - On which track does RPCA work better? Relate this to musical structure (repetitive vs changing accompaniment, vocal prominence).

**Deliverables:** code, a small results table (file × λ × qualitative score), and discussion.

---

## Problem 3 — Pretrained Conv-TasNet: domain and sample rate

**Background.** `CONVTASNET_BASE_LIBRI2MIX` is pretrained for **speech separation** at **8 kHz** on Libri2Mix. The lab also loads music (`Goodbye_Bolero`) at 8 kHz—an intentional domain mismatch worth studying.

**Tasks**

1. Separate the following with the pretrained pipeline (mono, `sr=8000`):
   - Speech-like mixture: build one yourself by mixing `speaker1.wav` and `speaker2.wav`  
     \[
     y = \frac{s_1 + s_2}{\max|s_1+s_2|}
     \]
     (trim/pad to equal length first).
   - Music: first 8 s of `Goodbye_Bolero.mp3` at 8 kHz (as in the notebook).
2. **Sample-rate ablation on the speech mixture only:** also run Conv-TasNet after loading the same mixture at `sr=16000` and `sr=22050` (still mono). Listen and note quality.
3. Plot waveforms + spectrograms of the two estimated sources for the **best** speech run and for the music run.
4. **Written analysis:**
   - Why is speech mixture separation expected to beat music separation with this checkpoint?
   - Why does wrong sample rate hurt time-domain models even if “you can still hear audio”?
   - Optional: load `Conv-Tasnet-Customized.pth` (if compatible with the lab’s `ConvTasNet` definition in `Lab2 - TasNet.ipynb`) and compare to the torchaudio pipeline on your speech mixture.

**Deliverables:** code, listening notes, figures, short discussion.

---

## Problem 4 — Evaluation metrics with ground truth *(core quantitative problem)*

**Background.** The notebook introduces **SDR / SIR / SAR** (`mir_eval.separation.bss_eval_sources`) and **SI-SDR** (`metric_sisdr`). For fair BSS evaluation you need **aligned reference sources**, not the mixture alone.

**Setup.** Use:

| File | Role |
|---|---|
| `Dataset/speaker1.wav` | Reference source 1 |
| `Dataset/speaker2.wav` | Reference source 2 |
| `Dataset/synthetic_mixture.wav` | Mixture (or rebuild \(s_1+s_2\)) |

**Tasks**

1. Load references and mixture; ensure identical sample rate and equal length (trim to `min` length).
2. Obtain estimated sources with **two methods**:
   - **Method A:** ICA on a **stereo** mixture. If you only have mono references, create a simple stereo observation, e.g.  
     \[
     x_1 = 0.8 s_1 + 0.4 s_2,\quad x_2 = 0.3 s_1 + 0.7 s_2
     \]
     then apply ICA. (Document your mixing matrix.)
   - **Method B:** Pretrained Conv-TasNet on the mono mixture (8 kHz).
3. Compute **SDR, SIR, SAR** (use `bss_eval_sources` with shape `(n_sources, n_samples)` for both reference and estimate). Handle **permutation**: report the assignment (source1↔est_i) that yields higher average SDR.
4. Implement or reuse **SI-SDR** for each source under the best permutation; report mean SI-SDR.
5. Fill a comparison table:

| Method | Mean SDR | Mean SIR | Mean SAR | Mean SI-SDR |
|---|---:|---:|---:|---:|
| ICA | | | | |
| Conv-TasNet | | | | |

6. **Written analysis:** Which metric best matches your listening impression? When would high SDR but low SIR be possible?

**Deliverables:** code, metric table, permutation note, analysis.

**Grading focus:** correct tensor/array shapes, permutation handling, and honest interpretation—not necessarily “ICA wins.”

---

## Problem 5 — Demucs music stem separation

**Background.** Demucs (lab: `htdemucs`) expects **stereo** audio, typically **44.1 kHz**, and outputs stems such as drums, bass, other/melody, vocals (order as in the notebook).

**Tasks**

1. Load `Dataset/falcon69.mp3` with `mono=False`, `sr=44100`, duration **10 s** (or full track if resources allow).
2. Convert to a `torch` tensor with shape expected by `apply_model` (batch dimension included).
3. Separate with `get_model('htdemucs')` and listen to all stems.
4. For each stem, plot (i) mono-downmixed waveform (normalized) and (ii) spectrogram for the first few seconds.
5. **Written analysis:**
   - Which stem is cleanest / noisiest on this clip?
   - Why is stereo + 44.1 kHz the right default for Demucs but not for Libri2Mix Conv-TasNet?
   - Name one use case for stem separation in production (remixing, karaoke, dataset building).

**Deliverables:** code, stem figures, short discussion.

**Note:** CPU inference is slow; use GPU if available. If CUDA is unavailable, document runtime and use a shorter clip (e.g. 5 s).

---

## Problem 6 — Conv-TasNet architecture, SI-SNR, and PIT *(from `Lab2 - TasNet.ipynb`)*

**Background.** The TasNet notebook defines `Conv1DBlock`, `Separator` (dilated TCN), and full `ConvTasNet` (encoder → masks → decoder), plus **SI-SNR** and **PIT loss**.

**Tasks**

1. **Architecture walk-through (with the hyperparameter figure in `Images/hyperparameters.png`):**
   - Draw or tabulate data shapes for a batch mixture of shape `(B, T)` through: encoder, separator masks, masked encoding, decoder output.
   - Explain the roles of \(N, L, B, H, P, X, R\) at a high level (what each controls).
2. **Forward-shape experiment:** instantiate `ConvTasNet` with default (or figure) hyperparameters; pass `torch.randn(2, 40000)` and print intermediate shapes (you may temporarily keep or add shape prints). Confirm output is `(B, num_spks, T')`.
3. **Loss reasoning:**
   - Show with a **tiny numerical example** (2 sources, short vectors) that without PIT, swapping estimates can make SI-SNR loss look terrible even if separation is perfect up to order.
   - Implement `si_snr` and `pit_loss` as in the notebook (or equivalent) and verify on that toy example that PIT picks the better permutation.
4. **Optional coding (if Libri2Mix-style folders are available):** train 1–2 epochs on a small subset and plot training loss; otherwise skip training and only complete (1)–(3).

**Deliverables:** shape table/diagram, toy PIT demonstration, code cells, brief written explanation of why PIT is necessary for multi-speaker separation.

---

## Problem 7 — Open discussion & multi-speaker extension *(synthesis / bonus)*

**Background.** The lab ends with: *if an audio mixes 3 or more speakers, how do we separate their voices?*

**Tasks (choose A or B, or both for bonus)**

**A. Conceptual design (required if 7 is assigned)**  
Write 1–1.5 pages covering:

1. Limits of **ICA** when \(n_{\text{sources}} > n_{\text{mics}}\).
2. How **Conv-TasNet / DPRNN-style** models choose `num_spks` at training time; what goes wrong at test time if the true speaker count differs.
3. At least **two** research/practical strategies for unknown or variable speaker count (e.g. recursive separation, attractor/clustering models, models trained with a maximum \(C\) speakers, target-speaker extraction with enrollment).
4. How **DPRNN’s dual-path** (intra-chunk vs inter-chunk) addresses long-sequence modeling compared to a pure dilated-TCN separator (refer to `Images/DPRNN.png`).

**B. Mini experiment (bonus)**  
Create a **3-speaker mono mixture** by mixing `speaker1.wav`, `speaker2.wav`, and a third track (e.g. a short segment of `Corine.mp3` speech-like region, or a second crop of the speakers with different delay). Run 2-speaker Conv-TasNet and report failure modes (who gets merged, residual interference). Propose one concrete next experiment.

**Deliverables:** written essay for A; optional notebook section for B.

---

## Suggested workload & mapping

| Problem | Focus | Est. time | Primary notebook |
|---|---|---|---|
| 1 | ICA + visualization | 45–60 min | Sound separation |
| 2 | RPCA + parameters | 60–75 min | Sound separation |
| 3 | Conv-TasNet domain/sr | 45–60 min | Sound separation |
| 4 | Metrics + fair comparison | 75–90 min | Sound separation |
| 5 | Demucs stems | 45–60 min | Sound separation |
| 6 | Architecture + PIT | 60–75 min | TasNet |
| 7 | Multi-speaker reasoning | 45–60 min | Discussion + DPRNN |

**Recommended set for a single lab session (≈3 hours):** Problems **1, 3, 4**, and short **7A**.  
**Recommended full take-home set:** Problems **1–6**, with **7** as bonus.

---

## Rubric (if graded)

| Criterion | Weight |
|---|---|
| Correct implementation & runnable code | 35% |
| Figures (waveforms / spectrograms) clearly labeled | 15% |
| Metrics computed correctly (shapes, permutation) | 20% |
| Analysis quality (limitations, domain, sample rate) | 20% |
| Clarity & organization of notebook/report | 10% |
| Problem 7 / extras | up to +10% bonus |

---

## Academic integrity

You may reuse lab template code with citation (`Lab2 - Sound seperation.ipynb`, `Lab2 - TasNet.ipynb`). Do not submit another student’s figures or metric tables. Use of pretrained weights provided in the course materials is allowed.

---

## Quick file checklist for students

```
Lab2_updated/
├── Dataset/
│   ├── mix_sample1.wav … mix_sample3.wav
│   ├── speaker1.wav, speaker2.wav, synthetic_mixture.wav
│   ├── Goodbye_Bolero.mp3, Corine.mp3, ComfyCouches.mp3, falcon69.mp3
│   └── …
├── Images/          # architecture figures
├── Conv-Tasnet-Customized.pth
├── Lab2 - Sound seperation.ipynb
├── Lab2 - TasNet.ipynb
└── Lab2_Practice_Problems.md   # this sheet
```
