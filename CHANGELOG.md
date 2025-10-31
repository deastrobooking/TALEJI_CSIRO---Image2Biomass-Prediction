# Changelog: CSIRO Image2Biomass Baseline Enhancements

## Version 2.0 - Enhanced Baseline (October 31, 2025)

### Overview

This document describes the enhancements made to the Python + R hybrid baseline notebook for the CSIRO Image2Biomass Prediction competition. These improvements are based on best practices outlined in the research textbook (README.md) and address common pitfalls in biomass prediction modeling.

---

## 🎯 Key Improvements

### 1. Log-Space Training

**Problem:** Biomass distributions are highly right-skewed, with values spanning multiple orders of magnitude. Linear models trained in raw space often struggle with extreme values and heteroscedasticity.

**Solution:** Train models in `log1p` space and transform predictions back to original scale.

**Implementation:**
```python
# During training
if USE_LOG_SPACE:
    y_train = np.log1p(y)  # log(1 + y) handles zeros gracefully
else:
    y_train = y

# During prediction
if USE_LOG_SPACE:
    pred = np.expm1(pred)  # exp(pred) - 1
    pred = np.maximum(pred, 0)  # Clip negatives
```

**Benefits:**
- Better handling of skewed distributions
- Reduces impact of extreme outliers
- Often improves R² by 0.01-0.03
- More stable gradients during training

**Trade-offs:**
- Adds slight complexity to pipeline
- Must remember to transform back for evaluation
- Can underestimate very large values

---

### 2. Isotonic Calibration

**Problem:** Even well-trained models can exhibit systematic bias in their predictions (e.g., consistently over/under-predicting in certain ranges).

**Solution:** Apply isotonic regression on out-of-fold predictions to calibrate the model's output distribution.

**Implementation:**
```python
from sklearn.isotonic import IsotonicRegression

# Fit on OOF predictions (ensures no data leakage)
if USE_ISOTONIC_CALIBRATION:
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_pred_raw, y_true)
    oof_pred = iso.predict(oof_pred_raw)
    calibrators[target] = iso
```

**Benefits:**
- Improves calibration without retraining base model
- Reduces systematic bias
- Non-parametric, adapts to actual error distribution
- Typical R² improvement: 0.005-0.02

**Trade-offs:**
- Requires storing additional calibrator objects
- Can overfit if OOF sample size is very small
- Monotonic constraint may limit flexibility

---

### 3. Physics-Based Constraints

**Problem:** Machine learning models don't inherently know physical relationships. They may predict:
- Negative biomass values
- `GDM ≠ Dry_Green + Dry_Clover` (violates definition)
- `Dry_Total < GDM` (impossible)

**Solution:** Post-process predictions to enforce physical consistency.

**Implementation:**
```python
def apply_physical_constraints(preds: pd.DataFrame) -> pd.DataFrame:
    preds = preds.copy()
    
    # 1. Ensure non-negative
    for t in TARGETS:
        preds[t] = np.maximum(preds[t], 0)
    
    # 2. Enforce GDM ≈ Dry_Green + Dry_Clover
    gdm_from_components = preds['Dry_Green_g'] + preds['Dry_Clover_g']
    preds['GDM_g'] = 0.7 * preds['GDM_g'] + 0.3 * gdm_from_components
    
    # 3. Ensure Dry_Total >= GDM
    preds['Dry_Total_g'] = np.maximum(preds['Dry_Total_g'], preds['GDM_g'])
    
    return preds
```

**Benefits:**
- Reduces physically impossible predictions
- Improves ensemble consistency
- Better interpretability for agronomists
- Typically improves weighted R² by 0.005-0.015

**Trade-offs:**
- Can slightly reduce flexibility
- May clip some valid predictions
- Weighted averaging ratio (0.7/0.3) is a hyperparameter

**Design Choices:**
- **70/30 blend** for GDM: Balances model prediction with component sum, avoiding over-constraining
- **Hard constraint** for Dry_Total: This is a strict physical requirement
- **Soft constraint** for GDM components: Allows small deviations to preserve model signal

---

### 4. Enhanced Validation & Cross-Checking

**Problem:** Custom metric implementations can have subtle bugs. It's critical to validate correctness before submission.

**Solution:** Cross-check manual R² against `sklearn.metrics.r2_score`.

**Implementation:**
```python
from sklearn.metrics import r2_score

# In weighted_r2_from_long function
for t in TARGETS:
    sub = merged[merged['target_name'] == t]
    r2 = r2_manual(sub['y_true'].values, sub['y_pred'].values)
    r2_sklearn = r2_score(sub['y_true'].values, sub['y_pred'].values)
    out[t] = float(r2)
    out[f'{t}_sklearn'] = float(r2_sklearn)  # For verification
    final += WEIGHTS[t]*r2
```

**Benefits:**
- Catches implementation bugs early
- Builds confidence in metric calculation
- Helps debug edge cases (zero variance, perfect predictions)

---

## 📊 Expected Performance Impact

Based on typical pasture biomass modeling scenarios:

| Enhancement | Expected Δ R² | Confidence | Effort |
|-------------|---------------|------------|--------|
| Log-space training | +0.01 to +0.03 | High | Low |
| Isotonic calibration | +0.005 to +0.02 | Medium | Low |
| Physics constraints | +0.005 to +0.015 | Medium | Low |
| **Combined** | **+0.02 to +0.06** | **Medium-High** | **Low** |

**Note:** Actual improvements depend heavily on:
- Data distribution and quality
- Base model strength
- Feature availability in test set
- Cross-validation strategy alignment with leaderboard split

---

## 🔧 Configuration Flags

The notebook now includes easy-to-toggle configuration flags:

```python
USE_LOG_SPACE = True               # Train in log1p space
USE_ISOTONIC_CALIBRATION = True    # Calibrate predictions
APPLY_PHYSICS_CONSTRAINTS = True   # Enforce physical rules
```

**Recommendation:** Start with all `True`, then ablate one at a time to measure individual impact.

---

## 🧪 Ablation Study Template

To measure the contribution of each enhancement:

1. **Baseline:** All flags `False`
2. **+Log:** `USE_LOG_SPACE = True`
3. **+Isotonic:** Add `USE_ISOTONIC_CALIBRATION = True`
4. **+Physics:** Add `APPLY_PHYSICS_CONSTRAINTS = True`

Track OOF weighted R² at each step:

| Configuration | Weighted R² | Δ from Previous |
|---------------|-------------|-----------------|
| Baseline | ? | - |
| +Log | ? | ? |
| +Isotonic | ? | ? |
| +Physics | ? | ? |

---

## 🚨 Important Caveats

### 1. Test Set Feature Availability
The competition test set **may not include metadata** (State, Species, Date, Height, NDVI). The notebook gracefully falls back to:
- Per-target mean predictions (simplest)
- Or a vision-only model if you add image features

**Always validate** that your enhancements work in the fallback scenario.

### 2. Calibration Requires Sufficient Data
Isotonic regression needs adequate OOF samples per target. With very small datasets (<100 samples/target), consider:
- Simpler linear calibration
- Skipping calibration entirely
- Using Platt scaling instead

### 3. Physics Constraints Are Heuristic
The 70/30 blend for GDM is not sacred. Consider:
- Tuning via grid search on OOF CV
- Making it target-dependent
- Using constrained optimization instead of post-hoc clipping

### 4. Log-Space Edge Cases
- **Zero values:** `log1p(0) = 0` is fine
- **Negatives:** Should never occur in biomass, but safeguard with `np.maximum(y, 0)` before `log1p`
- **Very large values:** Can be underestimated; monitor residuals

---

## 📈 Next Steps for Further Improvement

### Short-term (Low Effort, High Impact)
1. **Tune Ridge alpha:** Grid search over `[0.1, 1.0, 10.0, 100.0]`
2. **Add more date features:** Day-of-year, sin/cos encoding of seasonality
3. **Try LightGBM/XGBoost:** Often outperforms Ridge on tabular data
4. **Ensemble weights:** Instead of 50/50 Python/R blend, optimize via linear regression on OOF

### Medium-term (Moderate Effort)
5. **Vision features:** Add a DINOv2 or EfficientNet backbone for image embeddings
6. **Multi-modal fusion:** Concatenate image features + tabular metadata
7. **Pseudo-labeling:** Use high-confidence test predictions to augment training
8. **Advanced augmentation:** Geometric + photometric transforms for image data

### Long-term (Research Directions)
9. **Self-supervised pretraining:** Pretrain vision model on unlabeled pasture images
10. **Uncertainty quantification:** Deep ensembles or conformal prediction for prediction intervals
11. **Active learning:** Identify most informative samples for labeling
12. **Multi-task learning:** Joint head that models inter-target correlations

---

## 🐛 Debugging Checklist

If results are worse after enhancements:

- [ ] Check data preprocessing consistency between train/test
- [ ] Verify calibrators are fitted on OOF predictions only
- [ ] Ensure log-space transforms are inverted correctly
- [ ] Validate physics constraints don't over-constrain
- [ ] Cross-check metric calculation with sklearn
- [ ] Look for NaN/inf in intermediate predictions
- [ ] Check GroupKFold is preventing image_id leakage
- [ ] Verify submission format matches `sample_submission.csv`

---

## 📚 References

1. **README.md** - Research textbook with detailed motivation for each enhancement
2. **Competition rules** - https://www.kaggle.com/competitions/csiro-biomass
3. **Dataset paper** - Estimating Pasture Biomass from Top-View Images (arXiv:2510.22916)
4. **Scikit-learn isotonic regression** - https://scikit-learn.org/stable/modules/isotonic.html

---

## 🙏 Acknowledgments

These enhancements are based on:
- Best practices from the Kaggle community
- Agronomic domain knowledge from CSIRO researchers
- Recommendations from the research textbook (README.md)
- Standard techniques in uncertainty quantification and calibration

---

## 📝 Version History

- **v2.0** (Oct 31, 2025): Added log-space training, isotonic calibration, physics constraints, enhanced validation
- **v1.0** (Initial): Basic Python + R hybrid baseline with GroupKFold and weighted R² metric

---

## ⚖️ License

This code is released under the same license as the competition dataset. Use freely for research and competition purposes.
