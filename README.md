# TALEJI_CSIRO---Image2Biomass-Prediction

# Image2Biomass: A Research Textbook (Mini-Edition)

## Table of contents

1. Problem framing & evaluation
2. Pasture biomass science & field methods
3. Vegetation indices & spectral context
4. Dataset & labels (what you’re predicting)
5. Modeling recipes (image-only, tabular, multimodal)
6. Training, validation, and leakage traps
7. Metrics, error analysis, & uncertainty
8. Reproducibility & Kaggle constraints
9. Strong baselines (Python + R)
10. Research directions & readings

---

## 1) Problem framing & evaluation

**Goal.** Predict component-wise pasture biomass from top-view images plus limited metadata. The competition evaluates a **weighted average of per-target (R^2)** across five outputs with the following weights:

* `Dry_Green_g`: 0.1
* `Dry_Dead_g`: 0.1
* `Dry_Clover_g`: 0.1
* `GDM_g`: 0.2
* `Dry_Total_g`: 0.5

The official scoring is a weighted sum of (R^2) scores computed per target. Your submission must be a **long-format CSV** with one row per (image, target) pair. ([Kaggle][1])

**Why (R^2)?** It measures variance explained vs. a mean-only baseline and is sensitive to both bias and dispersion—useful when targets span broad numeric ranges. (You’ll still want residual plots and scale checks.)

---

## 2) Pasture biomass science & field methods (quick primer)

* **Clip-and-weigh (cut–dry–weigh)** is the classic, accurate but labor-intensive ground truth method. **Rising plate meters** infer herbage mass from compressed sward height via calibration; they’re fast but require species/season-specific equations. ([arxiv.org][2])

* Accurate biomass estimates improve grazing decisions, profitability, and environmental outcomes—one reason this dataset and challenge were created. ([arxiv.org][2])

---

## 3) Vegetation indices & spectral context

* **NDVI** (Normalized Difference Vegetation Index) tracks “greenness” using red and near-IR reflectance; higher values indicate denser, healthier vegetation. It’s a workhorse for agriculture and drought monitoring. Enhanced indices (e.g., **EVI**) reduce saturation in dense canopies. ([Earth Observatory][3])

* Long-running satellite NDVI records (e.g., NOAA CDR, GIMMS) provide historical context and seasonality if external data were allowed; here, you’ll typically rely on the NDVI provided in train metadata only. ([NCEI][4])

---

## 4) Dataset & labels (what you’re predicting)

The public dataset (released via CSIRO & partners) comprises **top-down quadrat images** (about 70 × 30 cm) with ground-truth biomass sorted by component, collected across seasons/sites in Australia. Five labels:

* **Dry_Green_g:** non-legume green vegetation
* **Dry_Dead_g:** senescent material
* **Dry_Clover_g:** clover component
* **GDM_g:** **green dry matter = green + clover**
* **Dry_Total_g:** total biomass

The accompanying paper describes collection, QA, and label structure, and explains why component-wise prediction supports feed-quality assessment. ([arxiv.org][2])

---

## 5) Modeling recipes

### 5.1 Image-only (vision backbones)

* **CNN baselines** (e.g., ResNet, EfficientNet) trained on cropped quadrats.
* **Vision Transformers** (ViT/DeiT) often excel with augmentation or SSL pretraining; patch-based attention handles mixed sward textures well. ([arxiv.org][5])
* **Self-supervised features** (e.g., **DINOv2**) can transfer strongly to small/medium datasets; freeze or fine-tune last blocks and predict 5 targets via a multi-head MLP. ([arxiv.org][6])

### 5.2 Tabular baselines (metadata only)

Use species, state, date features, NDVI (if allowed in split), and height to fit Ridge/GBDT; simple models provide sturdy anchors and useful ensembling diversity.

### 5.3 Multimodal fusion

Options:

* **Early fusion:** concatenate image embeddings with normalized tabular features; train a joint MLP head.
* **Late fusion:** blend predictions from image and tabular models (stacking or weighted average).
* **Co-training:** auxiliary losses for component consistency (e.g., enforce `GDM ≈ Dry_Green + Dry_Clover`).

### 5.4 Multi-output learning

If your regressor doesn’t support multi-target outputs, wrap it with **`MultiOutputRegressor`** (one regressor per target). Be aware this **doesn’t model inter-target correlations**—multi-task heads can. ([Scikit-learn][7])

---

## 6) Training, validation, and leakage traps

* **Group by image_id** (each image appears 5× in long format). Use **GroupKFold** to avoid leakage between folds.
* Respect the competition’s split rules: **some metadata appear only in train** and are not available for validation/test (e.g., sampling date, state, species, height, NDVI)—plan ablations accordingly to simulate test conditions fairly. ([arxiv.org][2])
* **Scale**: biomass spans orders of magnitude; consider log-space training targets (evaluate in original space as required by the competition). Keep an eye on negative predictions—clip at 0.

---

## 7) Metrics, error analysis, & uncertainty

* **Weighted (R^2)** emphasizes **Dry_Total** and **GDM**—your improvements will move the leaderboard most when they reduce errors there. ([Kaggle][1])
* Plot **per-component residuals** against predicted/true values; look for heteroscedasticity and seasonal/systematic patterns (e.g., specific species).
* Add **uncertainty** via **deep ensembles** or MC-dropout; calibrate with isotonic regression to know when not to trust a prediction.

---

## 8) Reproducibility & Kaggle constraints

* **Notebook limits**: single-notebook, ≤9 h, no internet at submission; file must be `submission.csv` in long format. External, publicly available models/data are allowed if included as dataset inputs. ([Kaggle][8])

---

## 9) Strong baselines (Python + R)

These are **competition-compliant** patterns you can adapt. (You already have a full working hybrid notebook from our previous step.)

### 9.1 Metric & submission helpers (Python)

Use a clean implementation of per-target (R^2), the official weights, and long-format submission builders. (See the notebook we generated; it writes `submission.csv` and reports fold scores.)

### 9.2 Tabular Ridge (Python) + Linear Models (R)

* **Python:** `OneHotEncoder + StandardScaler + Ridge`, `GroupKFold` by image_id; one model per target or a single multi-head.
* **R:** `lm()` or `glmnet` on the same engineered features (`Year`, `Month`, species/state factors, height, NDVI if allowed in train folds), returning per-target predictions.
* **Blend 50/50** or tuned weights; fallback to Python if R features are not available in test.
* This “orthogonal” ensemble often boosts leaderboard (R^2) with negligible complexity.

### 9.3 Image backbone (ViT/DINOv2)

* Extract 768- to 1024-D features from a ViT, fine-tune a small MLP head for five outputs. Modern SSL features are robust in small data regimes and may beat scratch training with strong augmentations. ([arxiv.org][5])

---

## 10) Research directions & readings

**A. Better vision features**

* Pretrain on pasture-like scenes (self-supervised) to improve texture sensitivity in mixed swards (grasses vs clover). ([arxiv.org][6])

**B. Component consistency & priors**

* Add a soft constraint or auxiliary loss: `GDM ≈ Dry_Green + Dry_Clover`; `Dry_Total ≥ GDM`. Penalize violations to reduce physically implausible outputs.

**C. Semi-/weak supervision**

* Use pseudo-labels from strong folds and consistency regularization across photometrically augmented crops.

**D. Geospatial & seasonal context (if allowed)**

* Encode **day-of-year** and simple harmonics ((\sin, \cos)) of season; cautious with features unavailable in test.

**E. Spectral & structure fusion**

* If additional sensors are permitted in future work, combine **active optical sensors** (AOS NDVI) and **height** with RGB imagery. Reviews show remote sensing (Sentinel-2) + ML can estimate pasture biomass and quality. ([mdpi.com][9])

**F. Error calibration for decisions**

* For grazing plans, deliver prediction **intervals** and risk-aware thresholds (e.g., with conformal prediction).

---

## Quick reference: target definitions (from the dataset paper)

* Dry_Green_g — non-legume green vegetation (grams)
* Dry_Dead_g — senescent material (grams)
* Dry_Clover_g — clover component (grams)
* GDM_g — **Green dry matter = green + clover** (grams)
* Dry_Total_g — total biomass (grams)
  These match the competition’s five outputs and motivate the evaluation weights. ([arxiv.org][2])

---

## Selected sources & further reading

* **Competition overview / rules / data** — CSIRO Image2Biomass on Kaggle. ([Kaggle][1])
* **Dataset paper** — *Estimating Pasture Biomass from Top-View Images* (includes label definitions and collection protocol). ([arxiv.org][2])
* **NDVI/EVI primers** — NASA Earth Observatory & Earthdata; USGS Landsat. ([Earth Observatory][3])
* **Remote sensing for pasture biomass** — Chen et al., *Remote Sensing* (Sentinel-2 + ML); Fernandes et al., *Sci. Reports* (quantity & nutritive value); 2025 review articles. ([mdpi.com][9])
* **Rising plate meter** — University of Kentucky Extension note. ([forages.mgcafe.uky.edu][10])
* **Vision Transformers & SSL** — ViT; DINOv2. ([arxiv.org][5])
* **Multi-output regression** — scikit-learn docs. ([Scikit-learn][7])
* **R spatial/remote sensing** — `terra` package documentation. ([CRAN][11])

---

## References

[1]: https://www.kaggle.com/competitions/csiro-biomass?utm_source=chatgpt.com "CSIRO - Image2Biomass Prediction"
[2]: https://arxiv.org/html/2510.22916v1 "Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture"
[3]: https://earthobservatory.nasa.gov/features/MeasuringVegetation?utm_source=chatgpt.com "Measuring Vegetation (NDVI & EVI)"
[4]: https://www.ncei.noaa.gov/products/climate-data-records/normalized-difference-vegetation-index?utm_source=chatgpt.com "Normalized Difference Vegetation Index CDR"
[5]: https://arxiv.org/abs/2010.11929?utm_source=chatgpt.com "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
[6]: https://arxiv.org/abs/2304.07193?utm_source=chatgpt.com "[2304.07193] DINOv2: Learning Robust Visual Features ..."
[7]: https://scikit-learn.org/stable/modules/multiclass.html?utm_source=chatgpt.com "1.12. Multiclass and multioutput algorithms"
[8]: https://www.kaggle.com/competitions/csiro-biomass/rules?utm_source=chatgpt.com "CSIRO - Image2Biomass Prediction"
[9]: https://www.mdpi.com/2072-4292/13/4/603?utm_source=chatgpt.com "Estimating Pasture Biomass Using Sentinel-2 Imagery and ..."
[10]: https://forages.mgcafe.uky.edu/sites/forages.ca.uky.edu/files/AGR277.pdf?utm_source=chatgpt.com "Using a Rising Plate Meter to Measure Pasture Growth"
[11]: https://cran.r-project.org/package%3Dterra?utm_source=chatgpt.com "CRAN: Package terra - Spatial Data Analysis"

Here you go — first I restate every equation from your images exactly (in clean LaTeX), then I review the logic in the Python+R notebook and helpers I built for you and point out a few edge-cases/improvements.

# Equations (from your screenshots)

## Per-target (R^2)

For each target (i),

## $$R_i^2 ;=; 1 ;-; \frac{SS_{\mathrm{res},i}}{SS_{\mathrm{tot},i}}$$.


Residual sum of squares:

## $$SS_{\mathrm{res},i} ;=; \sum_{j},\bigl(y_{ij}-\hat y_{ij}\bigr)^2$$,

Total sum of squares:

## $$SS_{\mathrm{tot},i} ;=; \sum_{j},\bigl(y_{ij}-\bar y_{i}\bigr)^2$$,

where (y_{ij}) is the ground-truth value for sample (j) and target (i), (\hat y_{ij}) is your prediction, and (\bar y_{i}) is the mean of the ground-truths for target (i).

## Final leaderboard score (weighted average of (R^2)s)

Let the five targets be indexed by (i=1,\dots,5) in the set
({\text{Dry_Green_g},\text{Dry_Dead_g},\text{Dry_Clover_g},\text{GDM_g},\text{Dry_Total_g}}).
With weights
$$
w_{\text{Dry_Green_g}}=0.1,\quad
w_{\text{Dry_Dead_g}}=0.1,\quad
w_{\text{Dry_Clover_g}}=0.1,\quad
w_{\text{GDM_g}}=0.2,\quad
w_{\text{Dry_Total_g}}=0.5,
$$
the final score is

## $$\text{Final Score} ;=; \sum_{i=1}^{5} w_i,R_i^2$$ .


## Submission key

Each row corresponds to a pair (image, target).

$$\texttt{sample_id} ;=; \texttt{<image_id>} ; + ; \texttt{"__"} ; + ; \texttt{<target_name>}$$.

The CSV has two columns: `sample_id,target`.

---

# Review of the software logic you have

Below I review the two main artifacts we produced for you:

1. **`image2biomass_eval.py`** (metric + formatting helpers)
2. **Hybrid Kaggle notebook (Python + R)**

## 1) Metric & helpers (`image2biomass_eval.py`)

**What’s right**

* Implements the official formula: merges ground truth and predictions on `sample_id`; computes (R^2) **per target** and then the **weighted sum**.
* Handles zero-variance edge case: if (SS_{\text{tot}}=0), returns (1.0) for perfect predictions else (0.0) (avoids div-by-zero).
* `preds_wide_to_long(...)` correctly builds `sample_id = image_id + "__" + target_name` and returns a valid long submission.
* `long_submission(...)` guarantees required two columns and a stable sort.

**Minor suggestions**

* Consider returning **NaN** for (R^2) if (SS_{\text{tot}}=0) (and then treating it as 0 in the final score) to make the behavior more explicit during debugging.
* Add a small unit test to assert the function yields **0.0** when predictions equal the **per-target mean** (sanity).

## 2) Hybrid Python+R baseline notebook

### Data & features

* Extracts `image_id` from `sample_id` correctly.
* Builds **meta features** (State, Species, Height, NDVI, Date→Year/Month) **only if present**, and detects when the test set lacks them. Good.
* Uses **GroupKFold by `image_id`** for OOF — prevents leakage across the 5 rows per image. ✅

### Python model (tabular)

* `OneHotEncoder(handle_unknown='ignore')` + `StandardScaler` + `Ridge` in a `Pipeline`: robust and simple.
* Multi-target handled by looping per target (equivalent to `MultiOutputRegressor` without cross-target coupling).
* OOF predictions → converted to long → evaluated with the **same metric**. Solid.

### R model

* Loaded via `rpy2` with graceful fallback.
* `lm()` per target with the same feature set; creates `Year/Month` if date exists.
* If test lacks features, returns NA and the **ensemble falls back** to Python predictions. Nice.

### Ensembling & submission

* Simple (0.5/0.5) blend with NA-aware fallback; produces valid long-format submission.
* Writes `submission.csv` and prints a preview. Good.

### Edge cases / improvements

1. **Date handling consistency**

   * You compute Year/Month in both Python and R; good. Just ensure **time zones** aren’t present (they aren’t here) and keep the same dtype (ints).

2. **Feature lists & ColumnTransformer**

   * You add `Year/Month` to `num_cols` **before** building the `ColumnTransformer` — correct.
   * In test-time, you correctly rebuild Year/Month and drop `Sampling_Date`. Good.

3. **When test lacks meta**

   * You correctly fall back to **per-target means** (format-correct, leaderboard-safe).
   * Optional: fall back to a **vision-only model** (e.g., ViT/DINOv2 head) if you later add image features — usually stronger than means.

4. **Target space**

   * Consider training in **log1p space** and `clip` negatives on inverse (`np.expm1()` then `np.maximum(…,0)`). This often improves (R^2) for skewed biomass.

5. **Physics/consistency penalties**

   * Add a soft penalty or post-processing to enforce ( \text{GDM} \approx \text{Dry_Green} + \text{Dry_Clover} ) and ( \text{Dry_Total} \ge \text{GDM} ).
   * Even a simple constrained least-squares adjustment on the 5 outputs per image can tighten errors.

6. **Calibration & intervals**

   * After ensembling, fit an **isotonic regression** per target on OOF to calibrate.
   * For uncertainty, consider a tiny **deep ensemble** or **conformal** intervals (use OOF residuals).

7. **Scoring cross-check**

   * For peace of mind, add a quick assertion that your (R^2) matches `sklearn.metrics.r2_score` per target on a toy split.

---

Here you go — first I restate every equation from your images exactly (in clean LaTeX), then I review the logic in the Python+R notebook and helpers I built for you and point out a few edge-cases/improvements.

# Equations (from your screenshots)

## Per-target (R^2)

For each target (i),
[
R_i^2 ;=; 1 ;-; \frac{SS_{\mathrm{res},i}}{SS_{\mathrm{tot},i}}.
]

Residual sum of squares:
[
SS_{\mathrm{res},i} ;=; \sum_{j},\bigl(y_{ij}-\hat y_{ij}\bigr)^2,
]
Total sum of squares:
[
SS_{\mathrm{tot},i} ;=; \sum_{j},\bigl(y_{ij}-\bar y_{i}\bigr)^2,
]
where (y_{ij}) is the ground-truth value for sample (j) and target (i), (\hat y_{ij}) is your prediction, and (\bar y_{i}) is the mean of the ground-truths for target (i).

## Final leaderboard score (weighted average of (R^2)s)

Let the five targets be indexed by (i=1,\dots,5) in the set
({\text{Dry_Green_g},\text{Dry_Dead_g},\text{Dry_Clover_g},\text{GDM_g},\text{Dry_Total_g}}).
With weights
[
w_{\text{Dry_Green_g}}=0.1,\quad
w_{\text{Dry_Dead_g}}=0.1,\quad
w_{\text{Dry_Clover_g}}=0.1,\quad
w_{\text{GDM_g}}=0.2,\quad
w_{\text{Dry_Total_g}}=0.5,
]
the final score is
[
\text{Final Score} ;=; \sum_{i=1}^{5} w_i,R_i^2 .
]

## Submission key

Each row corresponds to a pair (image, target).
[
\texttt{sample_id} ;=; \texttt{<image_id>} ; + ; \texttt{"__"} ; + ; \texttt{<target_name>}.
]
The CSV has two columns: `sample_id,target`.

---

# Review of the software logic you have

Below I review the two main artifacts we produced for you:

1. **`image2biomass_eval.py`** (metric + formatting helpers)
2. **Hybrid Kaggle notebook (Python + R)**

## 1) Metric & helpers (`image2biomass_eval.py`)

**What’s right**

* Implements the official formula: merges ground truth and predictions on `sample_id`; computes (R^2) **per target** and then the **weighted sum**.
* Handles zero-variance edge case: if (SS_{\text{tot}}=0), returns (1.0) for perfect predictions else (0.0) (avoids div-by-zero).
* `preds_wide_to_long(...)` correctly builds `sample_id = image_id + "__" + target_name` and returns a valid long submission.
* `long_submission(...)` guarantees required two columns and a stable sort.

**Minor suggestions**

* Consider returning **NaN** for (R^2) if (SS_{\text{tot}}=0) (and then treating it as 0 in the final score) to make the behavior more explicit during debugging.
* Add a small unit test to assert the function yields **0.0** when predictions equal the **per-target mean** (sanity).

## 2) Hybrid Python+R baseline notebook

### Data & features

* Extracts `image_id` from `sample_id` correctly.
* Builds **meta features** (State, Species, Height, NDVI, Date→Year/Month) **only if present**, and detects when the test set lacks them. Good.
* Uses **GroupKFold by `image_id`** for OOF — prevents leakage across the 5 rows per image. ✅

### Python model (tabular)

* `OneHotEncoder(handle_unknown='ignore')` + `StandardScaler` + `Ridge` in a `Pipeline`: robust and simple.
* Multi-target handled by looping per target (equivalent to `MultiOutputRegressor` without cross-target coupling).
* OOF predictions → converted to long → evaluated with the **same metric**. Solid.

### R model

* Loaded via `rpy2` with graceful fallback.
* `lm()` per target with the same feature set; creates `Year/Month` if date exists.
* If test lacks features, returns NA and the **ensemble falls back** to Python predictions. Nice.

### Ensembling & submission

* Simple (0.5/0.5) blend with NA-aware fallback; produces valid long-format submission.
* Writes `submission.csv` and prints a preview. Good.

### Edge cases / improvements

1. **Date handling consistency**

   * You compute Year/Month in both Python and R; good. Just ensure **time zones** aren’t present (they aren’t here) and keep the same dtype (ints).

2. **Feature lists & ColumnTransformer**

   * You add `Year/Month` to `num_cols` **before** building the `ColumnTransformer` — correct.
   * In test-time, you correctly rebuild Year/Month and drop `Sampling_Date`. Good.

3. **When test lacks meta**

   * You correctly fall back to **per-target means** (format-correct, leaderboard-safe).
   * Optional: fall back to a **vision-only model** (e.g., ViT/DINOv2 head) if you later add image features — usually stronger than means.

4. **Target space**

   * Consider training in **log1p space** and `clip` negatives on inverse (`np.expm1()` then `np.maximum(…,0)`). This often improves (R^2) for skewed biomass.

5. **Physics/consistency penalties**

   * Add a soft penalty or post-processing to enforce ( \text{GDM} \approx \text{Dry_Green} + \text{Dry_Clover} ) and ( \text{Dry_Total} \ge \text{GDM} ).
   * Even a simple constrained least-squares adjustment on the 5 outputs per image can tighten errors.

6. **Calibration & intervals**

   * After ensembling, fit an **isotonic regression** per target on OOF to calibrate.
   * For uncertainty, consider a tiny **deep ensemble** or **conformal** intervals (use OOF residuals).

7. **Scoring cross-check**

   * For peace of mind, add a quick assertion that your (R^2) matches `sklearn.metrics.r2_score` per target on a toy split.

---

## TL;DR

* The equations in your screenshots are correctly captured above and the code matches them.
* Your helpers and the hybrid notebook are **logically sound** for this competition’s rules and failure modes (missing test meta, no internet, single-notebook).
* If you want a quick lift: add **log-space training**, **post-hoc consistency adjustment**, and (optionally) a small **image backbone** to blend with the tabular models. Those three usually yield the biggest gains with minimal complexity.

