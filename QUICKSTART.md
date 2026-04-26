# Quick Start Guide - Hybrid IoT IDS

## 5-Minute Setup

### Step 1: Install Dependencies
```bash
cd c:\Users\Dell\Documents\IOT\ML_IOT
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Run Pipeline
```bash
python main.py
```

**First run will:**
- Download dataset (UNSW-NB15 or synthetic fallback)
- Cache it for future runs (~15-30 minutes total)
- Train 2 ML models
- Generate alerts and reports
- Print results to console and log files

### Step 3: Review Results
- **Console Output**: Comprehensive metrics summary
- **Logs**: `logs/hybrid_ids_YYYYMMDD.log`
- **Alerts**: `reports/phase4_alerts.json`
- **Models**: `models/isolation_forest_v1.joblib` and `models/random_forest_v1.joblib`

---

## Common Commands

### Run with cached data (fastest)
```python
# In main.py:
run_pipeline(use_cache=True, tune_models=False)
```
**Time: ~5 minutes**

### Run with hyperparameter tuning
```python
# In main.py:
run_pipeline(use_cache=True, tune_models=True)
```
**Time: ~15-20 minutes**

### Force fresh data download
```python
# In main.py:
run_pipeline(use_cache=False)
```
**Time: ~10-15 minutes** (includes download)

### Skip model saving
```python
# In main.py:
run_pipeline(save_models=False)
```

---

## Directory Quick Reference

| Directory | Purpose |
|-----------|---------|
| `config/` | Configuration and hyperparameters |
| `phase1_data/` | Data loading and preprocessing |
| `phase2_profiling/` | Device behavioral profiling |
| `phase3_models/` | ML models (IF, RF) |
| `phase4_fusion/` | Decision fusion and thresholding |
| `utils/` | Logging and utilities |
| `data/` | Cached datasets (auto-created) |
| `models/` | Trained model files (auto-created) |
| `logs/` | Log files (auto-created) |
| `reports/` | Alert and report outputs (auto-created) |

---

## Key Files to Understand

1. **main.py** - Pipeline orchestrator, entry point
2. **config/config.py** - All parameters and hyperparameters
3. **phase1_data/data_loader.py** - Dataset handling
4. **phase3_models/**.py - ML model implementations
5. **phase4_fusion/adaptive_threshold.py** - Alert generation

---

## Expected Output Format

### Console Output (Final Report)
```
╔══════════════════════════════════════════════════════╗
║              FINAL PIPELINE RESULTS                 ║
╚══════════════════════════════════════════════════════╝

── Phase 1: Data ──────────────────────────────────────
  Source      : Synthetic (or UNSW-NB15)
  Total rows  : 15,000
  Features    : 18

── Phase 2: Profiling ─────────────────────────────────
  Device profiles : 20
  Warmed up       : 18
  Augmented dims  : 20

── Phase 3: Models ────────────────────────────────────
  Isolation Forest  (test): ROC-AUC=0.92 | F1=0.85 | FPR=0.05
  Random Forest     (test): ROC-AUC=0.95 | F1=0.89 | FPR=0.03

── Phase 4: Fusion & Threshold ────────────────────────
  Best fusion method : weighted_linear
  Best alpha         : 0.45
  Calibrated τ_base  : 0.52
  Fused F1-score     : 0.91
  Fused ROC-AUC      : 0.96
  ...
  Total Alerts       : 342
  Critical Alerts    : 89
  Warning Alerts     : 253
```

### Alert JSON Format
```json
{
  "alert_id": "ALERT-000001",
  "device_id": "device_005",
  "timestamp": "2026-04-26T14:30:45.123Z",
  "fused_score": 0.8234,
  "threshold_used": 0.5234,
  "alert_level": "CRITICAL",
  "attack_type": "DDoS",
  "s_if_score": 0.75,
  "p_rf_score": 0.89,
  "context": {
    "network_load": 0.65,
    "device_risk": 0.5,
    "time_factor": 0.72,
    "hour": 14,
    "device_type": "gateway"
  }
}
```

---

## Performance Metrics Explained

| Metric | Meaning | Target |
|--------|---------|--------|
| **ROC-AUC** | Area under ROC curve, 0-1 | >0.90 = excellent |
| **F1-Score** | Harmonic mean of precision & recall | >0.85 = good |
| **Precision** | % of alerts that are true attacks | >0.95 = low FP |
| **Recall** | % of attacks detected | >0.80 = comprehensive |
| **FPR** | False positive rate | <0.05 = acceptable |

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'sklearn'`
```bash
pip install scikit-learn
```

### Issue: Dataset download timeout
The framework falls back to synthetic data. To retry:
```python
run_pipeline(use_cache=False)
```

### Issue: Memory error with large dataset
Reduce dataset size in `phase1_data/data_loader.py`:
```python
generator = SyntheticIoTDataset(n_samples=5000)  # Reduce from 15000
```

### Issue: No logs appearing
Check `logs/` directory exists:
```bash
mkdir logs
```

---

## Next Steps

1. **Customize hyperparameters**: Edit `config/config.py`
2. **Add your own dataset**: Implement in `phase1_data/data_loader.py`
3. **Extend fusion methods**: Add to `phase4_fusion/decision_fusion.py`
4. **Deploy models**: Load from `models/` directory and call predict()

---

For detailed documentation, see **README.md**
