# Implementation Guide: Enhanced CSIRO Image2Biomass Baseline

## Quick Start

### Running the Enhanced Notebook

1. **Open** `CSIRO_Image2Biomass_Python_R_Baseline.ipynb`
2. **Configure** feature flags at the top of the setup cell:
   ```python
   USE_LOG_SPACE = True
   USE_ISOTONIC_CALIBRATION = True
   APPLY_PHYSICS_CONSTRAINTS = True
   ```
3. **Run all cells** sequentially (requires ~5-10 minutes on CPU)
4. **Check** `submission.csv` is generated with correct format

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Loading & Prep                      │
│  - Read train.csv, test.csv                                  │
│  - Extract image_id from sample_id                           │
│  - Pivot targets to wide format (1 row per image)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Feature Engineering                         │
│  - Extract metadata (State, Species, Date, Height, NDVI)     │
│  - Create Year, Month from Sampling_Date                     │
│  - One-hot encode categoricals, scale numerics               │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────┐
        │    Python Model (Ridge)             │
        │  ┌─────────────────────────────┐    │
        │  │ 1. GroupKFold(n=5)          │    │
        │  │ 2. Train in log1p space     │    │
        │  │ 3. Generate OOF predictions │    │
        │  │ 4. Fit isotonic calibrators │    │
        │  │ 5. Train final models       │    │
        │  └─────────────────────────────┘    │
        └─────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────┐
        │    R Model (lm)                     │
        │  ┌─────────────────────────────┐    │
        │  │ 1. Same features as Python  │    │
        │  │ 2. Train per-target lm()    │    │
        │  │ 3. Generate predictions     │    │
        │  └─────────────────────────────┘    │
        └─────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Ensemble (50/50)                        │
│  - Blend Python + R predictions                              │
│  - Fallback to Python if R fails                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               Apply Physics Constraints                      │
│  1. Clip all predictions ≥ 0                                 │
│  2. GDM = 0.7 * GDM_pred + 0.3 * (Green + Clover)            │
│  3. Dry_Total = max(Dry_Total_pred, GDM)                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│            Convert to Long Format & Submit                   │
│  - Pivot wide → long (sample_id, target)                     │
│  - Save submission.csv                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Code Structure

### Core Files

```
CSIRO_Image2Biomass_Python_R_Baseline.ipynb  # Main notebook
├── Cell 1: Title & Feature Overview
├── Cell 2: Setup & Metric Helpers (ENHANCED)
│   ├── Imports (sklearn, numpy, pandas)
│   ├── Configuration flags
│   ├── r2_manual() - Manual R² calculation
│   ├── weighted_r2_from_long() - Competition metric
│   ├── apply_physical_constraints() - NEW
│   └── Long/wide conversion helpers
├── Cell 3: Data Loading
│   ├── Resolve paths (Kaggle/local)
│   ├── Load train/test/sample_submission
│   └── Extract image_id
├── Cell 4: Feature Assembly
│   ├── Extract metadata
│   ├── Pivot targets to wide
│   └── Detect test feature availability
├── Cell 5: Python Ridge Model (ENHANCED)
│   ├── Feature engineering (Year, Month)
│   ├── ColumnTransformer (scaling + encoding)
│   ├── GroupKFold training loop
│   ├── Log-space training - NEW
│   ├── Isotonic calibration - NEW
│   ├── Physics constraints on OOF - NEW
│   └── Metric evaluation
├── Cell 6: R Extension Loading
├── Cell 7: R Data Preparation
├── Cell 8: R Model Training (%%R magic)
├── Cell 9: Ensemble & Evaluation
└── Cell 10: Test Prediction & Submission (ENHANCED)
    ├── Apply calibration to test preds - NEW
    ├── Blend Python + R
    ├── Apply physics constraints - NEW
    └── Generate submission.csv
```

---

## Key Functions Reference

### `r2_manual(y_true, y_pred)`
**Purpose:** Calculate R² matching competition formula  
**Inputs:** Arrays of true and predicted values  
**Output:** Float R² score  
**Edge cases:**
- Empty arrays → `np.nan`
- Zero variance (SS_tot=0) → `1.0` if perfect else `0.0`

### `weighted_r2_from_long(true_long, pred_long)`
**Purpose:** Compute official weighted R² metric  
**Inputs:** Long-format DataFrames with `sample_id`, `target_name`, `target`  
**Output:** Dict with per-target R² and final weighted score  
**Validation:** Cross-checks with `sklearn.metrics.r2_score`

### `apply_physical_constraints(preds)`
**Purpose:** Post-process predictions for physical consistency  
**Inputs:** Wide DataFrame with target columns  
**Output:** Constrained DataFrame  
**Operations:**
1. Clip negatives: `preds[t] = max(preds[t], 0)`
2. Soft GDM constraint: `GDM = 0.7 * GDM + 0.3 * (Green + Clover)`
3. Hard total constraint: `Total = max(Total, GDM)`

### `fit_predict_ridge_oof(train_wide)`
**Purpose:** Train Ridge models with GroupKFold and calibration  
**Returns:** `(oof_preds, models, calibrators)`  
**Process:**
1. For each target:
   - Transform to log-space if configured
   - 5-fold GroupKFold by image_id
   - Train Ridge on each fold
   - Generate out-of-fold predictions
   - Fit IsotonicRegression on OOF
   - Train final model on all data
2. Apply physics constraints to OOF predictions

---

## Configuration Deep-Dive

### USE_LOG_SPACE

**When to enable:**
- Target distributions are right-skewed (biomass almost always is)
- Range spans >2 orders of magnitude
- Heteroscedasticity in residuals

**When to disable:**
- Targets are already normalized/standardized
- You want to optimize MAE instead of RMSE/R²
- Many zero values (though `log1p` handles this)

**Implementation details:**
```python
# Training
y_train = np.log1p(y) if USE_LOG_SPACE else y

# Prediction
pred = model.predict(X)
if USE_LOG_SPACE:
    pred = np.expm1(pred)  # Inverse transform
    pred = np.maximum(pred, 0)  # Safety clip
```

### USE_ISOTONIC_CALIBRATION

**When to enable:**
- You have >500 samples per target (rule of thumb)
- OOF predictions show systematic bias
- Model is well-regularized (avoids overfitting calibrator)

**When to disable:**
- Very small datasets (<100 samples/target)
- OOF predictions are already well-calibrated
- You're ensembling many diverse models (calibration may conflict)

**Implementation details:**
```python
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(oof_pred_raw, y_true)  # Fit on OOF only!
calibrators[target] = iso

# At test time
pred = model.predict(X_test)
pred = calibrators[target].predict(pred)
```

**Critical:** Never fit calibrators on training data directly—this causes severe overfitting. Always use OOF predictions.

### APPLY_PHYSICS_CONSTRAINTS

**When to enable:**
- You're predicting physically-related quantities
- Ensembles sometimes produce inconsistent predictions
- Domain experts will review predictions

**When to disable:**
- No clear physical relationships between targets
- Constraints are too restrictive (hurting CV)
- Using a multi-task model that learns correlations

**Tuning the GDM blend ratio:**
```python
# Current: 70% model prediction, 30% component sum
gdm_from_components = preds['Dry_Green_g'] + preds['Dry_Clover_g']
preds['GDM_g'] = 0.7 * preds['GDM_g'] + 0.3 * gdm_from_components

# To optimize, grid search over alpha:
for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
    preds['GDM_g'] = alpha * preds['GDM_g'] + (1-alpha) * gdm_from_components
    # Evaluate OOF CV score
```

---

## Troubleshooting

### Issue: OOF R² is negative

**Possible causes:**
- Feature/target leakage in preprocessing
- GroupKFold not applied correctly
- Test fold distributions very different from train

**Debug steps:**
1. Check `groups` parameter in `gkf.split()` uses `image_id`
2. Verify no data leakage (e.g., scaling on full data before split)
3. Print fold-wise R² to identify problematic folds
4. Check for NaN/inf in predictions

### Issue: Physics constraints hurt performance

**Possible causes:**
- Constraints are too strict
- Ground truth labels have inconsistencies
- Model is already learning correct relationships

**Solutions:**
1. Soften GDM constraint (increase model weight from 0.7 to 0.8-0.9)
2. Add tolerance bands instead of hard equality
3. Apply constraints only to extreme predictions
4. Disable and see if multi-task learning helps instead

### Issue: Log-space predictions are way off

**Possible causes:**
- Forgot to `expm1()` on predictions
- Targets have negative values (shouldn't happen for biomass)
- Extreme outliers dominating log-space

**Debug steps:**
1. Print predictions before and after `expm1()`
2. Check for negative targets: `assert (y >= 0).all()`
3. Plot histogram of log-transformed targets
4. Try `np.log1p(y + epsilon)` with small epsilon

### Issue: Isotonic calibration causes overfitting

**Possible causes:**
- Too few OOF samples per target
- Base model predictions have high noise
- Calibrator is too flexible for the data

**Solutions:**
1. Check OOF sample size: `print(len(oof_pred))`
2. Use simpler calibration (e.g., linear: `y_cal = a * y_pred + b`)
3. Increase calibrator regularization (though scikit's isotonic has none)
4. Skip calibration or apply only to well-sampled targets

### Issue: R model fails silently

**Possible causes:**
- `rpy2` not installed
- R dependencies missing (tidyverse, etc.)
- Feature names with special characters

**Solutions:**
1. Check `have_r` flag and fallback logic
2. Install rpy2: `pip install rpy2`
3. Test R manually: `%R print("Hello from R")`
4. Sanitize feature names (remove spaces, special chars)

---

## Performance Optimization

### Speed Improvements

1. **Reduce folds:** `GroupKFold(n_splits=3)` instead of 5 (faster, slightly less robust)
2. **Parallelize models:** Use `joblib.Parallel` to train targets independently
3. **Cache preprocessing:** Fit `ColumnTransformer` once, reuse across folds
4. **Sparse encoding:** Use `sparse_output=True` in `OneHotEncoder`

### Memory Optimization

1. **Delete intermediate objects:**
   ```python
   del X_tr, X_va, y_tr
   gc.collect()
   ```
2. **Use float32:** Convert targets to `np.float32` instead of `float64`
3. **Avoid copying DataFrames:** Use `.loc` and `inplace=True` carefully

---

## Extension Ideas

### Adding Image Features

```python
# Pseudo-code for adding vision backbone
from transformers import AutoModel, AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

def extract_features(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0].numpy()  # CLS token

# Extract for all images
train['img_features'] = train['image_id'].apply(lambda x: extract_features(f'images/{x}.jpg'))

# Concatenate with tabular features
X_combined = np.hstack([X_tabular, img_features])
```

### Multi-Task Learning

Instead of training 5 separate models, train a single multi-output model:

```python
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(Ridge(alpha=1.0))
y_all_targets = train_wide[TARGETS].values  # Shape: (n_samples, 5)
model.fit(X_train, y_all_targets)
preds = model.predict(X_test)  # Shape: (n_test, 5)
```

**Pros:** Shares information across targets, fewer models to train  
**Cons:** Can't use different hyperparameters per target

### Pseudo-Labeling

```python
# 1. Train on labeled data
model.fit(X_train, y_train)

# 2. Predict on test set
test_preds = model.predict(X_test)

# 3. Keep high-confidence predictions (top 20%)
confident_idx = np.argsort(prediction_variance)[:int(0.2 * len(test_preds))]

# 4. Add to training set
X_train_augmented = np.vstack([X_train, X_test[confident_idx]])
y_train_augmented = np.hstack([y_train, test_preds[confident_idx]])

# 5. Retrain
model.fit(X_train_augmented, y_train_augmented)
```

**Warning:** Can amplify errors if initial predictions are poor. Use with caution.

---

## Validation Strategy

### Ensuring Fair CV

```python
# Check no image_id appears in multiple folds
from collections import Counter

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=image_ids)):
    tr_imgs = set(image_ids[tr_idx])
    va_imgs = set(image_ids[va_idx])
    assert len(tr_imgs & va_imgs) == 0, f"Fold {fold} has leakage!"
    print(f"Fold {fold}: {len(va_imgs)} unique images in validation")
```

### Simulating Test Conditions

If test set lacks metadata, simulate this in CV:

```python
# Drop metadata from validation set
X_va_no_meta = X_va[['Height_Ave_cm', 'Pre_GSHH_NDVI']]  # Only allowed features
pred_va = model.predict(X_va_no_meta)
```

---

## Competition-Specific Tips

### Leaderboard Shakeup Prevention

1. **Use GroupKFold:** Prevents overfitting to specific images
2. **Don't overfit to public LB:** Public test set is small, can be noisy
3. **Ensemble diverse models:** Python Ridge + R lm + LightGBM + Vision
4. **Check private LB distribution:** If public/private differ greatly, trust CV more

### Submission Format Validation

```python
# Must match sample_submission exactly
assert submission.shape == sample_submission.shape
assert (submission['sample_id'] == sample_submission['sample_id']).all()
assert submission['target'].notna().all()
assert (submission['target'] >= 0).all()  # Biomass can't be negative
```

---

## FAQ

**Q: Should I use log-space for all targets?**  
A: Yes, unless you find via ablation that it hurts a specific target. Biomass is almost always skewed.

**Q: How do I know if isotonic calibration is working?**  
A: Plot calibration curves (predicted vs actual in bins). Calibrated models should lie close to the diagonal.

**Q: Can I change the 70/30 blend for GDM?**  
A: Absolutely. Grid search it as a hyperparameter. 70/30 is just a reasonable default.

**Q: What if my R model has lower R² than Python?**  
A: That's fine. Ensembling often helps even if one model is weaker, due to diversity.

**Q: Should I add more features?**  
A: Carefully. The competition restricts what's available in test. Always simulate test conditions in CV.

---

## Further Reading

- **Isotonic Regression:** [Sklearn docs](https://scikit-learn.org/stable/modules/isotonic.html)
- **GroupKFold:** [Sklearn docs](https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold)
- **Multi-output Regression:** [Sklearn docs](https://scikit-learn.org/stable/modules/multiclass.html#multioutput-regression)
- **Competition discussion:** [Kaggle forums](https://www.kaggle.com/competitions/csiro-biomass/discussion)

---

## Contact & Contributions

For questions or suggestions about these enhancements:
- Open an issue on the GitHub repo
- Post in the Kaggle competition discussion
- Review the research textbook (README.md) for deeper context

**Happy modeling! 🌱**
