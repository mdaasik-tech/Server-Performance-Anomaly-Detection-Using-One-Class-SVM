# Server Performance Anomaly Detection Using One-Class SVM

## Overview

Server infrastructure continuously generates operational telemetry such as CPU utilization, memory utilization, disk I/O, network latency, active thread count, and application error rates. Identifying abnormal behavior across these metrics is important for detecting performance degradation and potential operational issues.

This project implements an unsupervised machine learning pipeline for detecting anomalous server-performance patterns using **One-Class Support Vector Machine (One-Class SVM)**.

The project covers the complete workflow from data preprocessing and exploratory analysis to model development, hyperparameter tuning, anomaly scoring, evaluation, and visualization.

---

## Objectives

The primary objectives of this project are to:

- Analyze server-performance telemetry data.
- Identify abnormal observations without requiring manually labeled training data.
- Apply One-Class SVM for unsupervised anomaly detection.
- Preprocess and standardize heterogeneous numerical telemetry features.
- Tune the model's `nu` and `gamma` hyperparameters.
- Evaluate detected anomalies against a rule-based reference label.
- Analyze anomaly scores and visualize detected outliers.
- Establish a foundation for extending the solution into a production monitoring system.

---

## Problem Statement

Traditional server monitoring systems often depend on individual threshold rules. However, abnormal behavior may result from a combination of several metrics rather than a single metric exceeding a threshold.

For example, a server may exhibit a combination of:

- High CPU utilization
- Increased memory utilization
- Elevated disk activity
- Increased network latency
- High active thread count
- Increased application error rate

A machine learning-based anomaly detection approach can learn the underlying distribution of server telemetry and identify observations that deviate from normal behavior.

This project addresses that problem using One-Class SVM.

---

## Machine Learning Approach

### One-Class SVM

One-Class SVM is an unsupervised learning algorithm designed for novelty and anomaly detection. Instead of learning separate classes, the algorithm learns a boundary around the normal data distribution and identifies observations outside that boundary as anomalies.

The implementation uses an RBF kernel.

Model predictions are interpreted as:

```text
 1  -> Normal
-1  -> Anomaly
```

The notebook converts these predictions into the project's binary representation:

```text
0  -> Normal
1  -> Anomaly
```

---

## Dataset

The project uses the following dataset:

```text
server_performance_telemetry.csv
```

The supplied dataset contains:

- 600 records
- 7 columns

### Features

| Feature | Description |
|---|---|
| `ServerID` | Server identifier |
| `CPU_Utilization_Pct` | CPU utilization percentage |
| `Memory_Utilization_Pct` | Memory utilization percentage |
| `Disk_IOPS` | Disk input/output operations per second |
| `Network_Latency_MS` | Network latency in milliseconds |
| `Active_Thread_Count` | Number of active threads |
| `Error_Log_Rate_per_Min` | Error logs generated per minute |

The following six numerical telemetry features are used by the machine learning model:

```text
CPU_Utilization_Pct
Memory_Utilization_Pct
Disk_IOPS
Network_Latency_MS
Active_Thread_Count
Error_Log_Rate_per_Min
```

`ServerID` is treated as an identifier and is not used as a model feature.

---

## Project Workflow

```text
Raw Server Telemetry
        |
        v
Data Loading
        |
        v
Data Inspection
        |
        v
Data Cleaning
        |
        v
Exploratory Data Analysis
        |
        v
Feature Selection
        |
        v
Feature Scaling
        |
        v
Reference Label Generation
        |
        v
Baseline One-Class SVM
        |
        v
Hyperparameter Tuning
        |
        v
Final One-Class SVM
        |
        v
Anomaly Prediction
        |
        v
Anomaly Score Analysis
        |
        v
Model Evaluation
        |
        v
Visualization
```

---

## Data Preprocessing

The preprocessing pipeline includes the following stages.

### Column Name Cleaning

Whitespace is removed from column names to ensure consistent feature access.

```python
df.columns = df.columns.str.strip()
```

### Missing Value Analysis

Missing values are identified before model training.

```python
df.isnull().sum()
```

### Duplicate Analysis

Duplicate records are checked as part of the data-quality process.

```python
df.duplicated().sum()
```

### Infinite Value Handling

Infinite values are converted to missing values.

```python
df = df.replace([np.inf, -np.inf], np.nan)
```

### Missing Record Removal

Records containing missing values are removed.

```python
df = df.dropna().reset_index(drop=True)
```

### Feature Scaling

Since the telemetry features operate on different numerical scales, `StandardScaler` is used before training the One-Class SVM model.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Exploratory Data Analysis

The project performs exploratory analysis to understand the distribution and relationships among the server telemetry features.

The analysis includes:

- Individual feature distributions
- Correlation analysis
- CPU and memory utilization relationship
- Anomaly score distribution
- Anomaly visualization

The generated visualizations include:

```text
Distribution for CPU_Utilization_Pct.png
Distribution for Memory_Utilization_Pct.png
Distribution for Disk_IOPS.png
Distribution for Network_Latency_MS.png
Distribution for Active_Thread_Count.png
Distribution for Error_Log_Rate_per_Min.png
Correlation Heatmap.png
CPU vs Memory Utilization.png
Anomaly Score Distribution.png
```

The time-based anomaly visualization is conditional and requires a `Timestamp` column. The supplied dataset does not currently contain a `Timestamp` column.

---

## Reference Anomaly Label

The dataset does not contain a dedicated ground-truth anomaly label. Therefore, a rule-based reference label is generated for evaluation purposes.

An observation is classified as a reference anomaly when either of the following conditions is satisfied:

```text
CPU Utilization > 90%
OR
Error Log Rate > 50 per minute
```

The implementation is:

```python
df["reference_label"] = np.where(
    (df["CPU_Utilization_Pct"] > 90) |
    (df["Error_Log_Rate_per_Min"] > 50),
    1,
    0
)
```

For the supplied dataset:

```text
Reference normal records   : 580
Reference anomaly records  : 20
```

This reference label is a project-defined evaluation rule. It should not be interpreted as operational ground truth from a production monitoring environment.

---

## Model Configuration

The One-Class SVM implementation uses an RBF kernel.

```text
Kernel : RBF
nu     : Tuned
gamma  : Tuned
```

The model is trained using the standardized telemetry features.

---

## Hyperparameter Tuning

The project evaluates multiple combinations of `nu` and `gamma`.

### `nu`

```python
[
    0.01,
    0.03,
    0.05,
    0.10,
    0.20
]
```

### `gamma`

```python
[
    "scale",
    0.001,
    0.01,
    0.1,
    1
]
```

Each configuration is evaluated using:

- Precision
- Recall
- F1 Score
- Detected anomaly count
- Detected normal count
- Anomaly percentage

The final configuration is selected based on the highest F1 score against the project-defined reference label.

---

## Results

Based on the supplied dataset and notebook implementation, the selected configuration and evaluation results are:

| Metric | Result |
|---|---:|
| Total records | 600 |
| Reference anomalies | 20 |
| Reference normal records | 580 |
| Selected `nu` | 0.05 |
| Selected `gamma` | `scale` |
| Detected anomalies | 32 |
| Detected normal records | 568 |
| Detected anomaly percentage | 5.33% |
| Precision | 0.469 |
| Recall | 0.750 |
| F1 Score | 0.577 |

These results are specific to the supplied dataset and the rule-based reference-label definition described above.

---

## Model Evaluation

The model is evaluated using three primary classification metrics.

### Precision

Precision measures the proportion of model-detected anomalies that correspond to reference anomalies.

### Recall

Recall measures the proportion of reference anomalies detected by the model.

### F1 Score

F1 Score provides a combined measure of precision and recall.

Because the reference labels are generated from predefined thresholds rather than verified incident records, these metrics measure agreement with the reference rule rather than real-world incident-detection accuracy.

---

## Anomaly Scoring

The One-Class SVM decision function is stored in:

```python
df["Final_Score"]
```

The score can be used to examine how observations relate to the learned decision boundary.

Final anomaly predictions are extracted using:

```python
final_anomalies = df[
    df["Final_prediction"] == 1
]
```

This allows the detected anomalous records to be inspected independently.

---

## Project Structure

```text
server-performance-anomaly-detection/
|
├── data/
│   └── server_performance_telemetry.csv
|
├── notebooks/
│   └── Server_Performance_Anomaly_Detection_using_One_Class_SVM.ipynb
|
├── outputs/
│   ├── Distribution for CPU_Utilization_Pct.png
│   ├── Distribution for Memory_Utilization_Pct.png
│   ├── Distribution for Disk_IOPS.png
│   ├── Distribution for Network_Latency_MS.png
│   ├── Distribution for Active_Thread_Count.png
│   ├── Distribution for Error_Log_Rate_per_Min.png
│   ├── Correlation Heatmap.png
│   ├── CPU vs Memory Utilization.png
│   └── Anomaly Score Distribution.png
|
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies

| Category | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Scaling | StandardScaler |
| Anomaly Detection | One-Class SVM |
| Development Environment | Jupyter Notebook |

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd server-performance-anomaly-detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

For Linux or macOS:

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

A basic `requirements.txt` can contain:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## Running the Project

### Using Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/Server_Performance_Anomaly_Detection_using_One_Class_SVM.ipynb
```

Ensure that the dataset path matches the repository structure.

For example:

```python
df = pd.read_csv(
    "../data/server_performance_telemetry.csv"
)
```

Run the notebook cells sequentially to reproduce the analysis.

### Using Google Colab

Upload the dataset and notebook to Google Colab and update the dataset path if required:

```python
df = pd.read_csv(
    "/content/server_performance_telemetry.csv"
)
```

---

## Limitations

### Lack of Ground-Truth Labels

The dataset does not contain verified anomaly or incident labels. The evaluation therefore relies on a manually defined reference rule.

### Reference Rule Dependency

Precision, Recall, and F1 Score depend on the selected CPU and error-rate thresholds. Different thresholds will produce different evaluation results.

### Static Batch Processing

The current implementation processes a CSV dataset as a batch. It does not currently process live telemetry streams.

### Timestamp Availability

The supplied dataset does not contain a `Timestamp` field, limiting time-series analysis and temporal anomaly visualization.

### Model Scope

The current implementation focuses on One-Class SVM. Additional anomaly-detection algorithms have not been incorporated into the current notebook.

---

## Future Enhancements

The project can be extended into a production-oriented monitoring platform with the following capabilities:

- Real-time telemetry ingestion
- Streaming anomaly detection
- REST API using FastAPI
- Interactive monitoring dashboard
- Real-time anomaly alerts
- Server-level anomaly history
- Configurable detection thresholds
- Model persistence using Joblib
- Automated model retraining
- Model comparison with Isolation Forest
- Local Outlier Factor based detection
- Time-series anomaly analysis
- Alert integrations
- Docker-based deployment
- Cloud deployment
- Model and data monitoring

---

## Potential Production Architecture

```text
Server Monitoring Agents
          |
          v
   Telemetry Collector
          |
          v
      Data Pipeline
          |
          v
 Feature Preprocessing
          |
          v
  Anomaly Detection Model
          |
          v
   Anomaly Score / Label
          |
      +---+---+
      |       |
      v       v
 Dashboard  Alerting System
```

A production implementation could additionally introduce a feature store, model registry, persistent telemetry database, authentication, logging, monitoring, and automated retraining.

---

## Learning Outcomes

This project demonstrates practical experience with:

- Python-based data analysis
- Data cleaning and preprocessing
- Exploratory Data Analysis
- Feature selection
- Feature scaling
- Unsupervised machine learning
- Anomaly detection
- One-Class SVM
- Hyperparameter tuning
- Model evaluation
- Precision, Recall, and F1 Score
- Data visualization
- Machine learning workflow design
- Translating infrastructure telemetry into anomaly signals

---

## Reproducibility

To reproduce the results:

1. Clone the repository.
2. Install the dependencies listed in `requirements.txt`.
3. Place the dataset in the `data/` directory.
4. Open the supplied Jupyter Notebook.
5. Verify the dataset path.
6. Run the notebook from beginning to end.
7. Review the generated visualizations, anomaly predictions, scores, and evaluation metrics.

---

## Author

**Your Name**

Machine Learning / Python Project

Replace this section with your professional details before publishing:

```text
GitHub   : <your-github-profile>
LinkedIn : <your-linkedin-profile>
Email    : <your-email>
```

---

## License

This project can be distributed under the MIT License or another license selected by the project author.

---

## Project Summary

Server Performance Anomaly Detection Using One-Class SVM is an unsupervised machine learning project that analyzes server telemetry and identifies observations that deviate from learned normal behavior.

The project demonstrates an end-to-end anomaly-detection workflow covering data preprocessing, exploratory analysis, feature scaling, One-Class SVM modeling, hyperparameter tuning, anomaly scoring, evaluation, and visualization.

The current implementation provides a foundation that can be extended into a real-time server monitoring and anomaly-alerting system.
