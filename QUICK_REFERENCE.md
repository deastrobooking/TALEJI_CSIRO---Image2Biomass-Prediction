# Quick Reference: Enhanced Baseline Features

## 🎯 Configuration Flags (Cell 2)

```python
USE_LOG_SPACE = True               # Train in log1p space for skewed distributions
USE_ISOTONIC_CALIBRATION = True    # Calibrate predictions post-hoc
APPLY_PHYSICS_CONSTRAINTS = True   # Enforce physical consistency rules
```

---

## 📐 Mathematical Formulations

### Log-Space Training

**Forward (training):**
$$y_{\text{train}} = \log(1 + y)$$

**Inverse (prediction):**
$$\hat{y} = \exp(\hat{y}_{\text{log}}) - 1$$
$$\hat{y} = \max(\hat{y}, 0)$$

### Official Weighted R²

$$\text{Final Score} = \sum_{i=1}^{5} w_i \cdot R_i^2$$

Where:
- $R_i^2 = 1 - \frac{SS_{\text{res},i}}{SS_{\text{tot},i}}$
- $SS_{\text{res},i} = \sum_j (y_{ij} - \hat{y}_{ij})^2$
- $SS_{\text{tot},i} = \sum_j (y_{ij} - \bar{y}_i)^2$

**Weights:**
- Dry_Green_g: 0.1
- Dry_Dead_g: 0.1
- Dry_Clover_g: 0.1
- GDM_g: 0.2
- Dry_Total_g: 0.5

### Physics Constraints

**1. Non-negativity:**
$$\hat{y}_i = \max(\hat{y}_i, 0) \quad \forall i$$

**2. GDM consistency (soft):**
$$\text{GDM}_{\text{adj}} = 0.7 \cdot \text{GDM}_{\text{pred}} + 0.3 \cdot (\text{Green} + \text{Clover})$$

**3. Total ≥ GDM (hard):**
$$\text{Total}_{\text{adj}} = \max(\text{Total}_{\text{pred}}, \text{GDM}_{\text{adj}})$$

---

## 🔧 Key Code Snippets

### Isotonic Calibration

```python
from sklearn.isotonic import IsotonicRegression

# Fit on out-of-fold predictions
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(oof_predictions, y_true)

# Apply to test predictions
calibrated_pred = iso.predict(test_predictions)
```

### Physics Constraints

```python
def apply_physical_constraints(preds):
    preds = preds.copy()
    
    # Clip negatives
    for t in TARGETS:
        preds[t] = np.maximum(preds[t], 0)
    
    # GDM = 0.7 * model + 0.3 * components
    gdm_components = preds['Dry_Green_g'] + preds['Dry_Clover_g']
    preds['GDM_g'] = 0.7 * preds['GDM_g'] + 0.3 * gdm_components
    
    # Total >= GDM
    preds['Dry_Total_g'] = np.maximum(preds['Dry_Total_g'], preds['GDM_g'])
    
    return preds
```

### Cross-Validation with GroupKFold

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
groups = train_wide['image_id']  # Group by image to prevent leakage

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    # Train and validate...
```

---

## 📊 Expected Performance Gains

| Enhancement | Typical Δ R² | When It Helps Most |
|-------------|-------------|--------------------|
| Log-space training | +0.01 to +0.03 | Skewed distributions, outliers |
| Isotonic calibration | +0.005 to +0.02 | Systematic bias in predictions |
| Physics constraints | +0.005 to +0.015 | Inconsistent ensemble predictions |
| **Combined** | **+0.02 to +0.06** | Most scenarios |

---

## ✅ Pre-Submission Checklist

- [ ] All configuration flags set appropriately
- [ ] GroupKFold used (no image_id leakage)
- [ ] OOF weighted R² calculated and logged
- [ ] Physics constraints applied to final predictions
- [ ] submission.csv format validated:
  - [ ] 2 columns: `sample_id`, `target`
  - [ ] Correct number of rows (n_images × 5)
  - [ ] All targets ≥ 0
  - [ ] No NaN/inf values
- [ ] Cross-check with `sample_submission.csv` shape

---

## 🐛 Quick Debug Commands

### Check for data leakage
```python
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=image_ids)):
    tr_imgs = set(image_ids[tr_idx])
    va_imgs = set(image_ids[va_idx])
    assert len(tr_imgs & va_imgs) == 0, f"Leak in fold {fold}!"
```

### Validate metric implementation
```python
from sklearn.metrics import r2_score
assert np.isclose(r2_manual(y_true, y_pred), r2_score(y_true, y_pred))
```

### Check submission format
```python
assert submission.shape == (len(test['image_id'].unique()) * 5, 2)
assert (submission['target'] >= 0).all()
assert submission['target'].notna().all()
```

### Plot calibration curve
```python
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

y_true_binned, y_pred_binned = calibration_curve(
    y_true, y_pred, n_bins=10, strategy='quantile'
)
plt.plot(y_pred_binned, y_true_binned, marker='o')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Calibration Curve')
```

---

## 🎓 When to Use Each Feature

### Use Log-Space When:
- ✅ Targets span >2 orders of magnitude
- ✅ Distribution is right-skewed
- ✅ You see heteroscedasticity in residuals
- ❌ Targets are already normalized
- ❌ Optimizing MAE (absolute errors)

### Use Isotonic Calibration When:
- ✅ OOF sample size >500 per target
- ✅ Systematic over/under-prediction visible
- ✅ Model is well-regularized
- ❌ Very small datasets (<100 samples)
- ❌ Already using many diverse ensemble members

### Apply Physics Constraints When:
- ✅ Clear physical relationships exist
- ✅ Ensemble produces inconsistent predictions
- ✅ Domain experts review outputs
- ❌ No known relationships between targets
- ❌ Constraints hurt CV performance
- ❌ Using multi-task learning (learns correlations)

---

## 🔍 Ablation Study Template

```python
configs = [
    {'log': False, 'iso': False, 'phys': False},  # Baseline
    {'log': True,  'iso': False, 'phys': False},  # +Log
    {'log': True,  'iso': True,  'phys': False},  # +Isotonic
    {'log': True,  'iso': True,  'phys': True},   # +Physics (full)
]

results = []
for cfg in configs:
    USE_LOG_SPACE = cfg['log']
    USE_ISOTONIC_CALIBRATION = cfg['iso']
    APPLY_PHYSICS_CONSTRAINTS = cfg['phys']
    
    # Train and evaluate...
    oof_score = evaluate_model()
    results.append({'config': cfg, 'score': oof_score})

# Compare results
import pandas as pd
pd.DataFrame(results)
```

---

## 📚 File Structure

```
.
├── CSIRO_Image2Biomass_Python_R_Baseline.ipynb  # Main notebook (ENHANCED)
├── README.md                                      # Research textbook
├── CHANGELOG.md                                   # Detailed changes & rationale
├── IMPLEMENTATION_GUIDE.md                        # Technical deep-dive
├── QUICK_REFERENCE.md                             # This file
└── submission.csv                                 # Generated output
```

---

## 💡 Pro Tips

1. **Always use GroupKFold** with `image_id` as groups to prevent leakage
2. **Fit calibrators on OOF only** — never on training data directly
3. **Monitor per-target R²** during training to catch issues early
4. **Validate submission format** before uploading (shape, no negatives, no NaNs)
5. **Trust CV over public LB** if public test set is small
6. **Plot residuals** to understand where your model struggles
7. **Check physics constraints** aren't over-restricting (compare CV with/without)
8. **Ensemble diverse models** (Python + R + GBM + Vision) for best results

---

## 🆘 Emergency Fixes

### Negative R² in validation
```python
# Check for leakage
assert len(set(train_imgs) & set(val_imgs)) == 0

# Check preprocessing order
# ❌ scaler.fit(all_data)  # WRONG
# ✅ scaler.fit(train_data)  # CORRECT
```

### NaN predictions
```python
# Find NaN source
print(X_train.isna().sum())
print(y_train.isna().sum())

# Fill or drop
X_train = X_train.fillna(X_train.median())
```

### Memory errors
```python
# Use float32
y_train = y_train.astype(np.float32)

# Delete intermediates
del X_tr, X_va
gc.collect()
```

### Submission rejected
```python
# Validate format exactly matches sample
assert submission.columns.tolist() == ['sample_id', 'target']
assert submission.shape[0] == sample_submission.shape[0]
assert (submission['sample_id'].values == sample_submission['sample_id'].values).all()
```

---

## 📞 Resources

- **Kaggle Competition:** https://www.kaggle.com/competitions/csiro-biomass
- **Dataset Paper:** arXiv:2510.22916
- **Sklearn Docs:** https://scikit-learn.org/stable/
- **Full Documentation:** See README.md, CHANGELOG.md, IMPLEMENTATION_GUIDE.md

---

**Version:** 2.0  
**Last Updated:** October 31, 2025  
**Quick Start:** Set flags in Cell 2 → Run all cells → Submit `submission.csv`
