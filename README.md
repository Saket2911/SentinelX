# SentinelX
## AI-Driven Automated SOC Analyst and Explainable Threat Triage Engine

SentinelX is a predictive cyber-defence prototype that uses an LSTM-based World Model to learn temporal patterns in network activity and forecast future network states. It also provides residual-based analyst review, explainability, and historical streaming replay.

## 1. Problem Statement

Traditional security monitoring often depends on identifying suspicious activity after it has occurred. SentinelX explores predictive network-state modelling to help analysts understand how network behaviour evolves and identify unusual forecasting errors that may deserve further investigation.

## 2. Proposed Architecture

NETWORK DATA (CIC-IDS-2018)
        |
        v
FEATURE EXTRACTION
        |
        v
TIMESTAMPED NETWORK STATE REPRESENTATION
        |
        v
TIME-WINDOW SEQUENCES
        |
        v
LSTM WORLD MODEL
        |
        v
NEXT NETWORK STATE PREDICTION
        |
        v
RESIDUAL ANALYSIS AND ANALYST REVIEW

The model learns:

P(S(t+1) | S(t), S(t-1), ..., S(t-9))

The model predicts future network-state features. It does not directly predict attack stages or attack probabilities.

## 3. Dataset

Primary dataset: CSE-CIC-IDS2018.

The project uses multiple processed CIC-IDS-2018 network traffic captures from the available dataset archive.

Dataset processing includes:
- Data loading and validation
- Chronological splitting
- Missing and infinite value handling
- Feature preprocessing and scaling
- Timestamp-based network-state aggregation
- Sequence generation

The available archive contains ten capture-day CSV files. The dataset is stored locally and is not included in the source repository.

## 4. Technology Stack

- Python 3.12
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Joblib
- SHAP
- Streamlit

## 5. Machine Learning Model

SentinelX uses an LSTM World Model.

V3 configuration:
- Input features: 209
- Hidden size: 128
- LSTM layers: 1
- Dropout: 0.2
- Sequence length: 10
- Prediction horizon: One-step-ahead

The model predicts the next network state using the previous ten network-state observations.

## 6. Model Evaluation

Recorded V3 one-step test-partition results:

| Metric | LSTM | Persistence baseline |
|---|---:|---:|
| MAE | 0.320627 | 0.356227 |
| RMSE | 0.531328 | 0.679033 |
| Huber loss | 0.120682 | 0.177071 |

These metrics are measured in the preprocessed model feature space.

The test partition has also been used in exploratory analysis. It should not be described as an untouched independent test set.

## 7. Main Implemented Components

- Dataset loading and preprocessing
- Timestamped network-state construction
- Sequence generation
- LSTM World Model training
- One-step and multi-step forecasting evaluation
- Residual analysis
- Feature attribution using SHAP
- Residual-based analyst triage
- Contextual MITRE ATT&CK mapping
- Streaming prediction interface
- Historical replay and integration validation

## 8. Streaming and Historical Replay

The streaming predictor accepts already-aggregated, timestamped network-state rows containing 209 model features.

It maintains a sequence of ten observations and generates a next-state prediction when sufficient consecutive observations are available.

Historical replay was evaluated using 31,406 observations from the existing V3 test partition.

The streaming implementation is a prototype. It does not directly ingest raw packets or provide live packet capture.

## 9. Explainability and Threat Triage

SentinelX uses prediction residuals to identify observations that may require analyst review.

SHAP is used to study input-feature contributions to selected predicted state features.

MITRE ATT&CK mappings provide contextual references for dataset labels. They are not definitive attack classifications produced by the LSTM.

## 10. Project Structure

The repository has been reorganized into a stable layout that keeps the preserved research pipeline and generated artifacts intact while separating active code from historical analysis scripts.

- `src/sentinelx/` - Active model, feature, training, inference, evaluation, and explainability code
- `data/raw/`, `data/interim/`, `data/processed/`, `data/sequences/` - Current data lifecycle directories
- `scripts/pipeline/` - Primary dataset and feature construction scripts
- `scripts/training/` - Model training and benchmarking scripts
- `scripts/evaluation/` - Forecasting, residual, and explainability evaluation scripts
- `scripts/diagnostics/` - Dataset integrity and capture validation tools
- `scripts/analysis/` - Research analysis scripts
- `scripts/legacy/` - Retained V3 analysis and source-partition tooling
- `.quarantine/v1-v2-retirement-20261117/` - Reversible quarantine of obsolete V1/V2 scripts and generated artifacts; nothing has been permanently deleted
- `Datasets/` - Local datasets, raw archives, generated arrays, and metadata
- `Models/` - Trained models and checkpoints, including the preserved V3 run
- `reports/` - Evaluation and analyst-facing reports
- `tests/` - Automated validation tests
- `sentinelx_dashboard.py` - Streamlit dashboard prototype

A zero-byte extract from the original CIC-IDS archive was repaired by restoring the exact bytes from `Datasets/archive.zip` into `Datasets/archive/03-02-2018.csv` without modifying the source archive.

## 11. Installation

Create and activate a Python virtual environment.

Install the dependencies:

    python -m pip install -r requirements.txt

The recorded development environment used Python 3.12.16 and a CPU PyTorch build. GPU execution requires a compatible PyTorch installation and CUDA environment.

The dataset and trained model artifacts must be available locally before running the complete pipeline.

## 12. Testing

Run the automated streaming tests:

    # POSIX shells
    PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v

    # PowerShell
    $env:PYTHONPATH='src'; python -m unittest discover -s tests -p 'test_*.py' -v

Compile the source package:

    python -m compileall -q src/sentinelx

## 13. Limitations

- The system is a research and demonstration prototype, not a production SOC deployment.
- Forecasting errors are not proof of malicious activity.
- Low forecasting errors do not guarantee benign activity.
- Residual-based triage is not a validated standalone intrusion detector.
- MITRE ATT&CK mappings are contextual and depend on supporting evidence.
- Streaming inference expects preprocessed network-state rows rather than raw packet captures.
- Historical replay does not establish live-network performance.
- Additional independent compatible capture evaluation is unavailable with the current dataset collection.

## 14. Future Work

- Integrate raw network capture feature extraction with streaming inference.
- Evaluate additional compatible network datasets.
- Improve residual-based analyst review using independent validation data.
- Complete and validate the interactive SOC dashboard.
- Evaluate deployment performance in a controlled environment.

## 15. Project Status

SentinelX has completed its core V3 model development, forecasting evaluation, streaming prototype, historical replay, explainability analysis, and integrated triage prototype.

The dashboard and final end-to-end presentation workflow require final verification.
