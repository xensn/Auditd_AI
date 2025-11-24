# Quick Start Guide - Improved Auditd ML Model

## 🚀 Run the Improved Model (3 Steps)

### Step 1: Check Prerequisites

```bash
cd /home/ubuntu/Auditd_AI

# Check that cleaned data exists
ls -lh data/cleaned_data.csv

# Should see something like:
# -rw-r--r-- 1 ubuntu ubuntu 1.2M Nov 24 10:30 data/cleaned_data.csv
```

### Step 2: Create Required Directories

```bash
# Create directories for outputs
mkdir -p models results

# Verify
ls -d models results
```

### Step 3: Run the Improved Script

```bash
# Activate virtual environment (if you have one)
source .venv/bin/activate

# Run the improved model
python scripts/improved_auditd_ml.py
```

**Expected Runtime:** ~30-45 minutes (5 folds × ~6-9 minutes each)

---

## 📊 What Will Happen

### Phase 1: Data Loading & Preparation
```
============================================================
IMPROVED AUDITD ML MODEL - OVERFITTING FIXES
============================================================

1. Loading preprocessed data...
   Loaded 4797 consolidated events

2. Creating sequences (non-overlapping windows)...
100%|██████████| 96/96
   Total sequences: 96
   Label distribution:
     PRIVILEGE ESCALATION:   31 (32.3%)
     RECONNAISSANCE      :   29 (30.2%)
     BENIGN              :   19 (19.8%)
     DATA EXFILTRATION   :    9 ( 9.4%)
     PERSISTENCE         :    8 ( 8.3%)
```

### Phase 2: K-Fold Cross-Validation (5 Folds)
```
======================================================================
FOLD 1/5
======================================================================
Original training size: 76
After augmentation: 228 (+152)

Class weights for this fold:
  BENIGN               | Count:   15 | Weight: 0.76
  DATA EXFILTRATION    | Count:    5 | Weight: 2.28
  PERSISTENCE          | Count:    4 | Weight: 2.85
  PRIVILEGE ESCALATION | Count:   25 | Weight: 0.61
  RECONNAISSANCE       | Count:   23 | Weight: 0.66

Model parameters: 66,955,269
Trainable parameters: 66,955,269

Epoch 1/50 - Training: 100%|██| 57/57 [03:45<00:00, loss=1.4523, acc=0.4123]
Validation: 100%|██| 5/5 [00:18<00:00]

Epoch 1 Results:
  Train: Loss=1.4523, Acc=0.4123 (41.23%)
  Val:   Loss=1.4102, Acc=0.3947 (39.47%)
  Overfitting Gap: Acc=0.0176 (1.8%), Loss=0.0421
  ✓ New best model saved (val_loss: 1.4102)

...

Epoch 14 Results:
  Train: Loss=0.7234, Acc=0.7193 (71.93%)
  Val:   Loss=0.6891, Acc=0.7368 (73.68%)
  Overfitting Gap: Acc=-0.0175 (-1.8%), Loss=-0.0343
  ✓ New best model saved (val_loss: 0.6891)

...

Epoch 19 Results:
  Train: Loss=0.6523, Acc=0.7456 (74.56%)
  Val:   Loss=0.6712, Acc=0.7632 (76.32%)
  No improvement for 3 epoch(s)

  Early stopping triggered after 19 epochs

Fold 1 Best Results:
  Best Val Accuracy: 0.7632 (76.32%)
  Best Val Loss: 0.6712
  Overfitting Gap: 0.0124 (1.2%)
```

*This repeats for Folds 2-5...*

### Phase 3: Final Results
```
======================================================================
CROSS-VALIDATION SUMMARY
======================================================================
Average Validation Accuracy: 0.7234 ± 0.0312 (72.34%)
Average Overfitting Gap: 0.0543 ± 0.0189 (5.4%)

Per-Fold Results:
  Fold 1: Val Acc=0.7632, Gap=0.0124
  Fold 2: Val Acc=0.7105, Gap=0.0621
  Fold 3: Val Acc=0.6842, Gap=0.0589
  Fold 4: Val Acc=0.7368, Gap=0.0432
  Fold 5: Val Acc=0.7368, Gap=0.0549

6. Plotting training curves...
   Saved: /home/ubuntu/Auditd_AI/results/cross_validation_curves.png

======================================================================
TRAINING COMPLETE!
======================================================================

IMPROVEMENTS APPLIED:
  ✓ DistilBERT (40% fewer parameters)
  ✓ Strong dropout (0.3-0.4)
  ✓ Data augmentation (3x training data)
  ✓ Label smoothing (reduces overconfidence)
  ✓ Non-overlapping windows (no data leakage)
  ✓ L2 regularization (weight decay)
  ✓ Learning rate scheduling
  ✓ K-Fold cross-validation

Expected improvement: 19% overfitting gap → <8% gap
```

---

## 📁 Output Files

After running, you'll have:

### 1. Model Checkpoints
```bash
ls -lh models/
# best_model_fold1.pt  (best model from fold 1)
# best_model_fold2.pt  (best model from fold 2)
# best_model_fold3.pt  (best model from fold 3)
# best_model_fold4.pt  (best model from fold 4)
# best_model_fold5.pt  (best model from fold 5)
```

Each file contains:
- Model weights (state_dict)
- Optimizer state
- Training history
- Best validation metrics

### 2. Training Curves
```bash
ls -lh results/
# cross_validation_curves.png (loss & accuracy plots for all folds)
```

---

## 🔍 Interpreting Results

### ✅ Good Results (Overfitting Fixed):
```
Average Validation Accuracy: 0.72 ± 0.03 (72%)
Average Overfitting Gap: 0.05 ± 0.02 (5%)
```
- Validation accuracy **>70%** ✓
- Overfitting gap **<8%** ✓
- Standard deviation **<5%** ✓ (consistent across folds)

### ⚠️ Moderate Results (Some Overfitting):
```
Average Validation Accuracy: 0.68 ± 0.05 (68%)
Average Overfitting Gap: 0.12 ± 0.04 (12%)
```
- Validation accuracy 65-70% (okay)
- Overfitting gap 10-15% (still needs work)
- Try increasing augmentation or dropout

### ❌ Poor Results (Severe Overfitting):
```
Average Validation Accuracy: 0.55 ± 0.08 (55%)
Average Overfitting Gap: 0.22 ± 0.06 (22%)
```
- Validation accuracy <60% (not good)
- Overfitting gap >15% (severe)
- See troubleshooting section

---

## 🐛 Troubleshooting

### Error: "FileNotFoundError: data/cleaned_data.csv"

**Solution:**
```bash
# Check if file exists
ls data/

# If it's named differently, update the script:
# Edit line ~445 in improved_auditd_ml.py
df_consolidated = pd.read_csv('/home/ubuntu/Auditd_AI/data/YOUR_ACTUAL_FILENAME.csv')
```

---

### Error: "CUDA out of memory"

**Solution 1** - Use CPU instead:
```python
# Edit line ~30 in improved_auditd_ml.py
device = torch.device("cpu")  # Force CPU
```

**Solution 2** - Reduce batch size:
```python
# Edit get_optimized_config() function
'batch_size': 2,  # Reduce from 4 to 2
```

---

### Warning: "Validation accuracy not improving"

**Possible Causes:**
1. **Dataset too small** (<100 sequences)
   - Increase augmentation to 5x: `num_augmentations=5`
   - Use smaller model: Try `prajjwal1/bert-tiny`

2. **Classes severely imbalanced**
   ```bash
   # Check class distribution
   python -c "import pandas as pd; print(pd.read_csv('data/cleaned_data.csv')['label'].value_counts())"
   ```
   - If any class has <5 samples, consider removing it or collecting more data

3. **Log templates too simple**
   - Check if log_template column has enough variety
   - Might need to enhance feature engineering

---

### Training is very slow

**Speed optimizations:**

1. **Reduce sequence length:**
   ```python
   'max_token_length': 256,  # Instead of 384
   ```

2. **Use smaller batch size:**
   ```python
   'batch_size': 2,  # Fewer samples per batch = faster
   ```

3. **Reduce number of folds:**
   ```python
   # In main() function, line ~480
   fold_results, histories = cross_validation_training(
       sequences, labels, tokenizer, config, n_folds=3  # Instead of 5
   )
   ```

4. **Use GPU** (if available):
   ```bash
   # Check if GPU is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## 🧪 Testing on New Data

Once training is complete, test on new data:

```python
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load best model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=5
)
checkpoint = torch.load('models/best_model_fold1.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Test on new sequence
new_sequence = "SYSCALL_execve [SEP] UID_root [SEP] TIME_GAP_SHORT [SEP] ..."
encoding = tokenizer(
    new_sequence,
    padding='max_length',
    max_length=384,
    truncation=True,
    return_tensors='pt'
)
encoding = {k: v.to(device) for k, v in encoding.items()}

# Predict
with torch.no_grad():
    outputs = model(**encoding)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    probabilities = torch.softmax(outputs.logits, dim=1)[0]

# Decode prediction
label_names = ['BENIGN', 'DATA EXFILTRATION', 'PERSISTENCE',
               'PRIVILEGE ESCALATION', 'RECONNAISSANCE']
print(f"Prediction: {label_names[prediction]}")
print(f"Confidence: {probabilities[prediction]:.2%}")
print("\nAll probabilities:")
for label, prob in zip(label_names, probabilities):
    print(f"  {label:20s}: {prob:.2%}")
```

---

## 📈 Monitoring Training in Real-Time

If you want to see live progress:

```bash
# Run in background and log to file
nohup python scripts/improved_auditd_ml.py > training.log 2>&1 &

# Watch progress in another terminal
tail -f training.log

# Or use watch
watch -n 5 "tail -20 training.log"
```

---

## 💡 Tips for Best Results

### 1. Data Quality
- Ensure `data/cleaned_data.csv` has good log_template values
- Check for missing values in critical columns
- Verify timestamp ordering (should be sorted)

### 2. Class Balance
- Ideally, each class should have at least 10 samples
- If severely imbalanced, consider SMOTE oversampling
- Or focus on binary classification (attack vs benign)

### 3. Hyperparameter Tuning
After getting baseline results, try tuning:
```python
# More aggressive dropout
'dropout': 0.5  # Instead of 0.3

# More augmentation
num_augmentations=5  # Instead of 2

# Longer training
'num_epochs': 100  # Instead of 50
'patience': 8       # Instead of 5
```

### 4. Model Selection
If results are still poor, try:
```python
# Even smaller model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
model = AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny')
# Only 4.4M parameters!
```

---

## ✅ Success Checklist

- [ ] Script runs without errors
- [ ] All 5 folds complete training
- [ ] Average validation accuracy >70%
- [ ] Overfitting gap <8%
- [ ] Standard deviation across folds <5%
- [ ] Training curves saved in results/
- [ ] Model checkpoints saved in models/
- [ ] Can load and test saved models

**If all checked, congratulations! Your model is ready for deployment!**

---

## 🎯 Next Steps After Training

1. **Evaluate on test set:**
   - Load best model from fold with highest val accuracy
   - Test on completely unseen data
   - Calculate precision, recall, F1 per class

2. **Error analysis:**
   - Find misclassified examples
   - Understand failure modes
   - Collect more data for weak classes

3. **Model deployment:**
   - Export model to ONNX for production
   - Create inference API
   - Set up monitoring for drift detection

4. **Documentation:**
   - Document model performance
   - Create user guide for predictions
   - Set up retraining pipeline

---

## 📞 Need Help?

Common questions:

**Q: Training stuck at same accuracy?**
A: Check learning rate (might be too low), try 1e-5 instead of 5e-6

**Q: Validation accuracy lower than original?**
A: This is expected if original had data leakage. New results are more realistic.

**Q: Out of memory errors?**
A: Reduce batch_size to 2 or max_token_length to 256

**Q: Model predicts same class for everything?**
A: Class weights might be too extreme. Try without class weights or balance dataset differently.

**Q: How do I know which fold's model to use?**
A: Use the fold with highest validation accuracy. Check `fold_results` output.

---

**Ready to go? Run the script!**

```bash
cd /home/ubuntu/Auditd_AI/scripts
python improved_auditd_ml.py
```

Good luck! 🚀
