# Auditd_AI

## Overview
Auditd_AI is a machine learning-based security monitoring tool that analyzes Linux audit logs (auditd) to detect malicious activity. The system can identify and classify four types of security attacks:
- **Reconnaissance**: Network and system enumeration activities
- **Privilege Escalation**: Attempts to gain elevated privileges
- **Persistence**: Activities to maintain long-term access
- **Data Exfiltration**: Unauthorized data transfer attempts

## What Can This Tool Do?

### 1. Attack Simulation
The `attack_sim1.sh` script can:
- Simulate realistic attack scenarios across multiple attack types
- Generate labeled training data from audit logs
- Mix attack and benign activities to create realistic datasets
- Randomize command execution order and timing
- Collect auditd events for each simulated activity

**Attack Types Simulated:**
- **Reconnaissance**: Port scanning, service enumeration, network discovery
- **Privilege Escalation**: SUID binary enumeration, credential hunting, permission checks
- **Persistence**: Cron job enumeration, startup script inspection, configuration file access
- **Data Exfiltration**: Data compression, file transfer simulation

**Benign Activities Include:**
- Standard development commands (git, python, pip)
- System monitoring (top, df, free, uptime)
- File operations (ls, pwd)
- Service checks (systemctl)

### 2. Data Processing
The `data_format.py` script can:
- Parse raw auditd log files
- Clean and normalize log entries
- Remove system noise (ausearch commands)
- Convert logs into CSV format for ML training
- Handle PROCTITLE entries by combining command arguments
- Process message fields and remove formatting characters

### 3. Machine Learning Analysis
The `auditd_ml.ipynb` Jupyter notebook provides:
- Framework for training ML models on audit log data
- Data analysis and visualization capabilities
- Classification of security events
- Pattern recognition in system behavior

## Usage

### Running Attack Simulation
```bash
./attack_sim1.sh <ATTACK_SESSIONS> <BENIGN_SESSIONS>
```

Example:
```bash
./attack_sim1.sh 10 15
# Runs 10 attack sessions and 15 benign sessions in random order
```

**Note:** Requires sudo privileges for ausearch command.

### Processing Audit Data
```bash
python3 data_format.py
```

This will:
- Read from `training_data.txt`
- Process and clean the data
- Output to `training_data.csv`

### Machine Learning Analysis
Open and run the Jupyter notebook:
```bash
jupyter notebook auditd_ml.ipynb
```

## Requirements
- Linux system with auditd enabled
- Python 3.x
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
- Jupyter Notebook (for ML analysis)
- sudo privileges (for ausearch)

## Data Flow
1. **Generate Data**: Run `attack_sim1.sh` to create `training_data.txt`
2. **Process Data**: Run `data_format.py` to convert to `training_data.csv`
3. **Train Model**: Use `auditd_ml.ipynb` to analyze and train ML models
4. **Detect Threats**: Apply trained models to real-time audit logs

## Output Format
The tool generates labeled audit log data in the format:
```
[LABEL, timestamp, event_type, details...]
```

Where LABEL is one of:
- BENIGN
- RECONNAISSANCE
- PRIVILEGE ESCALATION
- PERSISTENCE
- DATA EXFILTRATION

## Security Use Cases
- **Training Security Analysts**: Generate realistic attack scenarios
- **Testing Detection Systems**: Validate security monitoring tools
- **ML Model Development**: Build and train intrusion detection models
- **Behavioral Analysis**: Study attack patterns and system behavior
- **Baseline Creation**: Establish normal vs. malicious activity patterns

## Limitations
- Requires auditd to be properly configured and running
- Simulation commands may trigger real security alerts
- Should only be used in controlled test environments
- Some commands require appropriate system permissions

## Contributing
Contributions are welcome! Please ensure any simulated attacks are safe and ethical.
