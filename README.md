# Auditd_AI

A BERT-based machine learning system for detecting security threats in Linux audit logs. This project uses transformer models to classify audit log sequences into attack types, enabling automated threat detection and security monitoring.

## Overview

Auditd_AI analyzes Linux audit daemon (auditd) logs to detect malicious activities through deep learning. The system processes raw audit logs, engineers meaningful features, and uses BERT (Bidirectional Encoder Representations from Transformers) to classify sequences of events into specific attack categories or benign activity.

### Supported Attack Types

- **Reconnaissance**: Network scanning, system enumeration, information gathering
- **Privilege Escalation**: Attempts to gain elevated system privileges
- **Persistence**: Establishing persistent access mechanisms (cron jobs, startup scripts)
- **Data Exfiltration**: Unauthorized data extraction and transfer
- **Benign**: Normal, legitimate system activities

## Key Features

- **Automated Attack Simulation**: Generate labeled training data with realistic attack patterns
- **Advanced Feature Engineering**: Extract meaningful patterns from raw audit logs using log templates
- **Sliding Window Approach**: Contextual sequence analysis with configurable window sizes
- **BERT-based Classification**: Leverage transformer architecture for superior sequence understanding
- **Class Imbalance Handling**: Weighted loss functions and stratified sampling
- **Comprehensive Data Pipeline**: End-to-end processing from raw logs to predictions

## Project Structure

```
Auditd_AI/
├── data/                          # Dataset directory
│   ├── actual_data.txt           # Raw audit logs (31,517 lines)
│   ├── actual_data.csv           # Converted CSV format (15,575 records)
│   ├── training_data.txt         # Labeled training data (31,517 lines)
│   ├── training_data.csv         # Training data in CSV (15,575 records)
│   ├── standardised_data.csv     # Standardized field extraction (15,576 records)
│   ├── cleaned_data.csv          # Final processed dataset (4,797 events)
│   └── bp_training_data.txt      # Baseline pattern training data (6,277 lines)
│
├── scripts/                       # Processing and training scripts
│   ├── attack_sim.sh             # Attack simulation and data generation
│   ├── txt_to_csv.py             # Raw log to CSV converter
│   ├── data_format.py            # Audit log field standardization
│   └── auditd_ml.ipynb           # Main ML pipeline (training & evaluation)
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
   pip install pandas numpy matplotlib seaborn scikit-learn transformers torch tqdm
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
    'num_epochs': 20,
    'batch_size': 8,
    'learning_rate': 2e-5,
    'max_token_length': 512,
    'model_name': 'bert-base-uncased'
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

**Log Template Creation**: Extracts semantic features
```
SYSCALL_execve SYSCALL_CAT_PROCESS USER_ROOT UID_ROOT
EFFECTIVE_USER_ROOT USER_SWITCH SYSTEM_BINARY FILES_2
```

**Sliding Window Sequences**: Creates context-aware samples
```
Window size: 50 events
Stride: 25 events
Attack threshold: 30% (for labeling)
Result: 190 sequences
```

### Stage 5: Model Training
- **Architecture**: BERT-base-uncased (110M parameters)
- **Tokenization**: WordPiece with 512 max tokens
- **Loss Function**: Weighted CrossEntropyLoss (handles class imbalance)
- **Optimizer**: AdamW with learning rate 2e-5
- **Early Stopping**: Patience of 3 epochs

## Model Architecture

```
Input: Log Template Sequence (50 events)
  ↓
BERT Tokenizer (max_length=512)
  ↓
BERT Encoder (12 layers, 768 hidden, 12 attention heads)
  ↓
Classification Head (768 → 5 classes)
  ↓
Output: [BENIGN, DATA EXFILTRATION, PERSISTENCE,
         PRIVILEGE ESCALATION, RECONNAISSANCE]
```

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
- **Final Training Accuracy**: 72.81%
- **Final Validation Accuracy**: 68.42%
- **Training/Validation Split**: 60/20/20 (stratified)

### Class Weights (Balanced)
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
3. Feature engineering (log templates)
4. Sliding window sequence generation
5. BERT model training
6. Model evaluation and metrics

### Data Files

| File | Records | Description |
|------|---------|-------------|
| `actual_data.txt` | 31,517 | Raw audit logs from actual system |
| `training_data.txt` | 31,517 | Labeled logs from simulation |
| `training_data.csv` | 15,575 | CSV-converted training data |
| `standardised_data.csv` | 15,576 | Normalized 91-column format |
| `cleaned_data.csv` | 4,797 | Consolidated events with templates |
| `bp_training_data.txt` | 6,277 | Baseline pattern data |

## Log Template Features

The system extracts rich semantic features from audit events:

### Syscall Categories
- `SYSCALL_CAT_FILE_OPS`: File operations (open, read, write, unlink)
- `SYSCALL_CAT_PROCESS`: Process management (fork, execve, exit)
- `SYSCALL_CAT_NETWORK`: Network operations (socket, connect, bind)
- `SYSCALL_CAT_PRIVILEGE`: Privilege changes (setuid, setgid)
- `SYSCALL_CAT_RECON`: Reconnaissance (stat, access, getdents)

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
- `SUSPICIOUS_BINARY`: Known attack tools (nmap, nc, metasploit)
- `TEMP_EXECUTION`: Execution from /tmp or /dev/shm
- `ADMIN_TOOL`: Administrative utilities (sudo, ssh, chmod)
- `RAPID_REPEAT_SYSCALL`: Potential scanning behavior

### Temporal Features
- `TIMING_INSTANT`: <0.1s between events
- `TIMING_RAPID`: <1s between events
- `TIMING_NORMAL`: <10s between events
- `TIMING_SLOW`: >10s between events

## Limitations & Future Work

### Current Limitations
- Limited to 4 specific attack types
- Requires labeled training data from simulations
- Performance dependent on data quality and volume
- May not generalize to novel attack patterns
- Quadratic complexity of transformer attention (O(n²))

### Future Improvements
- Expand attack type coverage (lateral movement, denial of service)
- Implement online learning for adaptive detection
- Add explainability features (attention visualization)
- Optimize for real-time inference
- Integrate with SIEM systems
- Explore efficient architectures (Mamba, sparse transformers)
- Implement anomaly detection for zero-day attacks

## Contributing

Contributions are welcome! Areas for improvement:
- Additional attack type simulations
- Feature engineering enhancements
- Model architecture experiments
- Real-world dataset integration
- Performance optimization

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

- **BERT**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)
- **Hugging Face Transformers**: For providing pre-trained models and tools
- **Linux Audit Framework**: For comprehensive system event logging

## References

- Original BERT Paper: https://arxiv.org/abs/1810.04805
- Hugging Face Documentation: https://huggingface.co/docs/transformers
- Linux Audit Documentation: https://linux.die.net/man/8/auditd
- Attack Simulation Techniques: MITRE ATT&CK Framework

---

**Author**: [Your Name]
**Last Updated**: 2025-11-19
**Version**: 1.0
