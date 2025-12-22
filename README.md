# Auditd_AI

A DistilBERT-based machine learning system for detecting security threats in Linux audit logs. This project uses transformer models to classify audit log sequences into attack types, enabling automated threat detection and security monitoring with improved efficiency.

## Overview

Auditd_AI analyzes Linux audit daemon (auditd) logs to detect malicious activities through deep learning. The system processes raw audit logs, engineers meaningful features, and uses DistilBERT (a distilled version of BERT - Bidirectional Encoder Representations from Transformers) to classify sequences of events into specific attack categories or benign activity. DistilBERT provides faster inference and lower memory footprint while maintaining strong performance.

### Supported Attack Types

- **Reconnaissance**: Network scanning, system enumeration, information gathering
- **Privilege Escalation**: Attempts to gain elevated system privileges
- **Persistence**: Establishing persistent access mechanisms (cron jobs, startup scripts)
- **Data Exfiltration**: Unauthorized data extraction and transfer
- **Benign**: Normal, legitimate system activities

## Key Features

- **Automated Attack Simulation**: Generate labeled training data with realistic attack patterns
- **Advanced Feature Engineering**: Extract meaningful patterns from raw audit logs using enhanced log templates
- **Sliding Window Approach**: Contextual sequence analysis with configurable window sizes and attack thresholds
- **DistilBERT-based Classification**: Leverage efficient transformer architecture for superior sequence understanding with 40% fewer parameters
- **Class Imbalance Handling**: Multiple strategies including weighted loss, Focal Loss, and sqrt-transformed class weights
- **Hyperparameter Optimization**: Integrated Optuna framework for automated hyperparameter tuning
- **Model Persistence**: Automated saving of models and training results with joblib
- **Comprehensive Data Pipeline**: End-to-end processing from raw logs to predictions

## Project Structure

```
Auditd_AI/
├── data/                          # Dataset directory
│   ├── training_data.txt         # Labeled training data (raw audit logs)
│   ├── training_data.csv         # Training data in CSV format
│   ├── standardised_data.csv     # Standardized field extraction (15,017 records)
│   └── bp_training_data.txt      # Baseline pattern training data (6,277 lines)
│
├── scripts/                       # Processing and training scripts
│   ├── attack_sim.sh             # Attack simulation and data generation
│   ├── txt_to_csv.py             # Raw log to CSV converter
│   ├── data_format.py            # Audit log field standardization
│   └── auditd_ml.ipynb           # Main ML pipeline (training & evaluation)
│
├── models/                        # Saved models (created during training)
│   └── baseline_results.pkl      # Baseline training results and configuration
│
├── results/                       # Training results (created during training)
│
└── README.md                      # This file
```

## Requirements

### System Requirements
- Linux-based operating system (for audit log generation)
- Python 3.8+
- CUDA-compatible GPU (optional, recommended for training)
- Minimum 8GB RAM (16GB+ recommended)

### Python Dependencies

```bash
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
transformers>=4.20.0
torch>=1.10.0
tqdm>=4.62.0
optuna>=3.0.0          # For hyperparameter tuning
joblib>=1.1.0          # For model persistence
```

### System Tools
- `auditd` - Linux audit daemon
- `ausearch` - Audit log search utility
- `sudo` privileges (for audit log access)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Auditd_AI
   ```

2. **Install Python dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn transformers torch tqdm optuna joblib
   ```

3. **Verify auditd installation**
   ```bash
   sudo systemctl status auditd
   sudo ausearch --version
   ```

## Usage

### 1. Generate Training Data

Use the attack simulation script to generate labeled audit logs:

```bash
cd scripts
chmod +x attack_sim.sh

# Generate data: 10 attack sessions and 5 benign sessions
sudo ./attack_sim.sh 10 5
```

**Output**: `training_data.txt` with labeled audit logs

### 2. Convert to CSV Format

```bash
python3 txt_to_csv.py training_data
```

**Input**: `data/training_data.txt`
**Output**: `data/training_data.csv`

### 3. Standardize Audit Fields

```bash
python3 data_format.py training_data
```

**Input**: `data/training_data.csv`
**Output**: `data/standardised_data.csv` (91 normalized columns)

### 4. Train the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook auditd_ml.ipynb
```

**Key Steps in Notebook**:
1. Load standardized data
2. Data cleaning and exploratory analysis
3. Feature engineering (log templates, sliding windows)
4. Train/validation/test split (60/20/20)
5. BERT model training with class weights
6. Model evaluation and metrics

### 5. Model Training Configuration

```python
CONFIG = {
    'num_epochs': 25,
    'batch_size': 32,
    'learning_rate': 1e-5,
    'max_token_length': 512,
    'model_name': 'distilbert-base-uncased',
    'dropout': 0.1,
    'attention_dropout': 0.1
}
```

## Data Pipeline

### Stage 1: Raw Log Collection
The `attack_sim.sh` script executes commands and captures audit logs using `ausearch`:

```bash
# Example: Capture logs for a command
start_time=$(date +%H:%M:%S)
command_to_monitor
end_time=$(date +%H:%M:%S)
sudo ausearch -ts $start_date $start_time -te $end_date $end_time -i
```

### Stage 2: CSV Conversion
`txt_to_csv.py` parses raw audit logs:
- Removes duplicate entries
- Handles multi-value fields
- Combines command arguments
- Labels each event

### Stage 3: Field Standardization
`data_format.py` creates uniform field structure:
- Extracts 91 unique audit fields
- Handles complex field values (e.g., `exit=ENOENT(No,such,file,or,directory)`)
- Creates consistent schema across all events

### Stage 4: Feature Engineering
The ML notebook performs:

**Event Consolidation**: Groups related audit events by `audit_id`
```
4,797 consolidated events from 15,017 raw records
```

**Enhanced Log Template Creation**: Extracts rich semantic features
```
SYSCALL_execve SYSCALL_CAT_PROCESS USER_ROOT UID_ROOT
EFFECTIVE_USER_ROOT USER_SWITCH SYSTEM_BINARY FILES_2
PRIVILEGE_CHANGE ESCALATE_TO_ROOT ACCESS_PASSWD_FILES
SUSPICIOUS_BINARY TIMING_RAPID RAPID_REPEAT_SYSCALL
```

**Sliding Window Sequences**: Creates context-aware samples
```
Window size: 50 events
Stride: 25 events
Attack threshold: 30% (for labeling)
Result: 190 sequences
```

### Stage 5: Model Training
- **Architecture**: DistilBERT-base-uncased (~66M parameters, 40% smaller than BERT)
- **Tokenization**: WordPiece with 512 max tokens
- **Loss Functions**:
  - Weighted CrossEntropyLoss with sqrt-transformed class weights
  - Focal Loss (alternative implementation for severe class imbalance)
- **Optimizer**: AdamW with learning rate 1e-5 and weight decay 0.01
- **Early Stopping**: Patience of 5 epochs
- **Regularization**: Dropout (0.1) and Attention Dropout (0.1)

## Model Architecture

```
Input: Log Template Sequence (50 events)
  ↓
DistilBERT Tokenizer (max_length=512)
  ↓
DistilBERT Encoder (6 layers, 768 hidden, 12 attention heads, ~66M params)
  ↓
Pre-classifier Layer (768 → 768) with Dropout (0.1)
  ↓
Classification Head (768 → 5 classes)
  ↓
Output: [BENIGN, DATA EXFILTRATION, PERSISTENCE,
         PRIVILEGE ESCALATION, RECONNAISSANCE]
```

**DistilBERT vs BERT Comparison**:
| Feature | BERT-base | DistilBERT |
|---------|-----------|------------|
| Layers | 12 | 6 |
| Hidden Size | 768 | 768 |
| Attention Heads | 12 | 12 |
| Parameters | ~110M | ~66M |
| Speed | Baseline | ~60% faster |
| Model Size | 440MB | 260MB |

## Results

### Dataset Statistics
- **Total sequences**: 190
- **Sequences with attacks**: 146 (76.8%)

### Label Distribution
| Attack Type | Count | Percentage |
|-------------|-------|------------|
| BENIGN | 57 | 30.0% |
| PRIVILEGE ESCALATION | 53 | 27.9% |
| RECONNAISSANCE | 41 | 21.6% |
| DATA EXFILTRATION | 20 | 10.5% |
| PERSISTENCE | 19 | 10.0% |

### Model Performance
- **Final Training Accuracy**: 57.02%
- **Final Validation Accuracy**: 52.63%
- **Training/Validation Split**: 60/20/20 (stratified)
- **Total Epochs Completed**: 25
- **Best Validation Loss**: 1.0759

### Training Configuration
- **Epochs**: 25
- **Batch Size**: 32
- **Learning Rate**: 1e-5
- **Weight Decay**: 0.01
- **Dropout**: 0.1
- **Attention Dropout**: 0.1

### Class Weights (sqrt-transformed balanced weights)
| Class | Count | Weight |
|-------|-------|--------|
| BENIGN | 34 | 0.67 |
| DATA EXFILTRATION | 12 | 1.90 |
| PERSISTENCE | 11 | 2.07 |
| PRIVILEGE ESCALATION | 32 | 0.71 |
| RECONNAISSANCE | 25 | 0.91 |

## File Descriptions

### Scripts

#### `attack_sim.sh`
Bash script for generating labeled training data through controlled attack simulations.

**Features**:
- Simulates 4 attack types with realistic command sequences
- Generates benign activity patterns
- Randomly interleaves attack and benign sessions
- Captures audit logs with precise timestamps
- Outputs labeled data ready for processing

**Usage**: `sudo ./attack_sim.sh <attack_sessions> <benign_sessions>`

#### `txt_to_csv.py`
Converts raw audit log text files to CSV format.

**Features**:
- Parses audit log syntax (type=value pairs)
- Combines multi-part commands (PROCTITLE)
- Handles special characters and formatting
- Filters duplicate entries
- Preserves label information

**Usage**: `python3 txt_to_csv.py <filename_without_extension>`

#### `data_format.py`
Standardizes audit log fields into consistent schema.

**Features**:
- Auto-discovers 91+ unique audit fields
- Handles complex field values with embedded commas
- Creates uniform column structure
- Validates parsing quality
- Exports to standardized CSV

**Usage**: `python3 data_format.py <input_csv_name_without_extension>`

#### `auditd_ml.ipynb`
Main machine learning pipeline for training and evaluation.

**Sections**:
1. Data loading and preprocessing
2. Exploratory Data Analysis (EDA)
3. Enhanced feature engineering (advanced log templates)
4. Sliding window sequence generation with attack thresholds
5. DistilBERT model training with class balancing
6. Model evaluation and metrics
7. Hyperparameter tuning with Optuna
8. Model persistence and result saving

### Data Files

| File | Records | Description |
|------|---------|-------------|
| `training_data.txt` | Variable | Labeled logs from attack simulation |
| `training_data.csv` | Variable | CSV-converted training data |
| `standardised_data.csv` | 15,017 | Normalized 91-column format |
| `bp_training_data.txt` | 6,277 | Baseline pattern data |

### Model Files

| File | Description |
|------|-------------|
| `models/baseline_results.pkl` | Saved baseline training results, configuration, and history |

## Log Template Features

The system extracts rich semantic features from audit events through an enhanced template system:

### Syscall Categories
- `SYSCALL_CAT_FILE_OPS`: File operations (open, read, write, unlink, rename)
- `SYSCALL_CAT_PROCESS`: Process management (fork, execve, exit, kill, ptrace)
- `SYSCALL_CAT_NETWORK`: Network operations (socket, connect, bind, listen)
- `SYSCALL_CAT_PRIVILEGE`: Privilege changes (setuid, setgid, capset)
- `SYSCALL_CAT_RECON`: Reconnaissance (stat, access, getdents, readlink)

### User & Privilege Patterns
- `USER_ROOT`, `USER_SYSTEM`, `USER_NORMAL`: User context
- `PRIVILEGE_CHANGE`: UID/EUID mismatch detected
- `ESCALATE_TO_ROOT`: Escalation to root privileges
- `USER_SWITCH`: AUID differs from UID

### File Access Patterns
- `ACCESS_PASSWD_FILES`: /etc/passwd or /etc/shadow access
- `ACCESS_SSH_CONFIG`: SSH configuration access
- `ACCESS_LOG_FILES`: Log file access
- `MANY_FILES_ACCESSED`: Bulk file operations (>10 files)

### Behavioral Patterns
- `SUSPICIOUS_BINARY`: Known attack tools (nmap, nc, netcat, metasploit, john, hydra, sqlmap)
- `TEMP_EXECUTION`: Execution from /tmp or /dev/shm (potential malicious activity)
- `SYSTEM_BINARY`: Execution from /usr/bin or /bin (legitimate system paths)
- `ADMIN_TOOL`: Administrative utilities (sudo, ssh, chmod, passwd, useradd, chown)
- `RAPID_REPEAT_SYSCALL`: Identical syscalls in rapid succession (potential scanning behavior)

### Temporal Features
- `TIMING_INSTANT`: <0.1s between events
- `TIMING_RAPID`: <1s between events
- `TIMING_NORMAL`: <10s between events
- `TIMING_SLOW`: >10s between events

## Advanced Training Features

### Focal Loss Implementation

Focal Loss is implemented as an alternative to standard CrossEntropyLoss for handling severe class imbalance. It down-weights easy examples and focuses training on hard-to-classify cases.

**Mathematical Definition**:
```
FL(pt) = -α(1-pt)^γ * log(pt)
```

Where:
- `α` (alpha): Class weight (uses sqrt-transformed balanced weights)
- `γ` (gamma): Focusing parameter (typically 2.0-3.0)
- `pt`: Model's estimated probability for the correct class

**Benefits**:
- Reduces the relative loss for well-classified examples
- Focuses training effort on challenging samples
- Particularly effective for minority attack classes (Persistence, Data Exfiltration)

**Usage in Code**:
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha      # Class weights
        self.gamma = gamma      # Focusing parameter
        self.reduction = reduction

criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
```

### Hyperparameter Tuning with Optuna

The project integrates Optuna for automated hyperparameter optimization to reduce overfitting and improve generalization.

**Tunable Parameters**:
- `learning_rate`: 5e-6 to 5e-5 (log scale)
- `weight_decay`: 0.01 to 0.1 (log scale)
- `dropout`: 0.2 to 0.5
- `attention_dropout`: 0.1 to 0.4
- `batch_size`: [8, 16, 32, 64]
- `gamma` (Focal Loss): 1.5 to 3.0

**Optimization Objective**:
```python
objective = max(0, loss_gap) * 2.0 + (1 - val_acc) * 1.0
```
Minimizes overfitting (validation-training loss gap) while maximizing validation accuracy.

**Features**:
- Early trial pruning for inefficient hyperparameter combinations
- 8-epoch trials for faster exploration
- Automated GPU memory management
- Results visualization with optimization history plots

**Usage**:
```python
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
```

### Model Persistence

Training results and configurations are automatically saved for reproducibility:

**Baseline Results (.pkl)**:
```python
baseline_results = {
    'train_acc': final_training_accuracy,
    'val_acc': final_validation_accuracy,
    'best_val_loss': best_validation_loss,
    'loss_gap': overfitting_metric,
    'config': training_configuration,
    'history': epoch_by_epoch_metrics
}
```

Saved to: `Auditd_AI/models/baseline_results.pkl`

## Limitations & Future Work

### Current Limitations
- Limited to 4 specific attack types
- Requires labeled training data from simulations
- Performance dependent on data quality and volume
- May not generalize to novel attack patterns
- Small dataset size (190 sequences) limits model generalization
- Moderate accuracy (~52-57%) indicates need for more training data

### Implemented Improvements
- **Efficient Architecture**: Migrated from BERT to DistilBERT (40% fewer parameters, 60% faster)
- **Advanced Class Balancing**: Implemented Focal Loss and sqrt-transformed class weights
- **Hyperparameter Optimization**: Integrated Optuna for automated tuning
- **Enhanced Feature Engineering**: Sophisticated log template system with 15+ feature categories
- **Model Persistence**: Automated saving of models and training configurations
- **Regularization**: Added dropout and attention dropout to reduce overfitting

### Future Improvements
- **Data Expansion**: Increase dataset size and diversity for better generalization
- **Attack Type Coverage**: Add lateral movement, denial of service, credential access
- **Online Learning**: Implement adaptive detection for evolving attack patterns
- **Explainability**: Add attention visualization and SHAP values for interpretability
- **Real-time Inference**: Optimize for production deployment with model quantization
- **SIEM Integration**: Build connectors for Splunk, ELK, and other security platforms
- **Anomaly Detection**: Complement supervised learning with unsupervised methods for zero-day attacks
- **Ensemble Methods**: Combine DistilBERT with other models (LSTM, CNN) for improved accuracy
- **Active Learning**: Automatically identify and prioritize samples for labeling

## Contributing

Contributions are welcome! Priority areas for improvement:
- **Dataset Expansion**: Generate more diverse attack scenarios and benign activity
- **Model Optimization**: Experiment with different architectures and training strategies
- **Feature Engineering**: Develop additional log template features and pattern detection
- **Real-world Testing**: Validate on production audit logs from actual systems
- **Performance Tuning**: Optimize inference speed for real-time detection
- **Documentation**: Improve setup guides and add tutorial notebooks

## Security Notice

This project is intended for:
- Educational purposes
- Authorized security testing
- Research and development
- Defensive security applications

**Do not use for unauthorized access or malicious activities.**

## License

[Specify your license here]

## Acknowledgments

- **BERT & DistilBERT**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2018); Sanh et al., "DistilBERT: a distilled version of BERT" (2019)
- **Hugging Face Transformers**: For providing pre-trained models and tools
- **Linux Audit Framework**: For comprehensive system event logging
- **Optuna**: For efficient hyperparameter optimization framework
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)

## References

- **BERT Paper**: https://arxiv.org/abs/1810.04805
- **DistilBERT Paper**: https://arxiv.org/abs/1910.01108
- **Focal Loss Paper**: https://arxiv.org/abs/1708.02002
- **Hugging Face Documentation**: https://huggingface.co/docs/transformers
- **Optuna Documentation**: https://optuna.readthedocs.io/
- **Linux Audit Documentation**: https://linux.die.net/man/8/auditd
- **MITRE ATT&CK Framework**: https://attack.mitre.org/

---

**Project Status**: Active Development
**Last Updated**: 2024-12-05
**Current Version**: 2.0 (DistilBERT with Hyperparameter Tuning)
