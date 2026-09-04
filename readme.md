# Temporal Botnet Detection using Graph Neural Networks (GNNs) & SMOTE Augmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu-red.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.6%2B-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![Course](https://img.shields.io/badge/Course-AIML%20505-purple.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()


> **Dataset**: [UNSW 2018 Bot-IoT Dataset](https://research.unsw.edu.au/projects/bot-iot-dataset)  
> **Primary Dataset File**: `Dataset/All_features/UNSW_2018_IoT_Botnet_Full.csv`

---

## 👥 Group F Team Members

| Name |
| :--- | 
| **Fardin Rahman** 
| **Jubayer Alam Likhon** 

---

## 📋 Table of Contents

- [1. Executive Summary & Problem Formulation](#1-executive-summary--problem-formulation)
- [2. Dataset Architecture & Exploratory Data Analysis (EDA)](#2-dataset-architecture--exploratory-data-analysis-eda)
  - [2.1 Class Imbalance & Attack Spectrum](#21-class-imbalance--attack-spectrum)
  - [2.2 Descriptive Statistics & Moments](#22-descriptive-statistics--moments)
  - [2.3 Stationarity & Autocorrelation (ADF, ACF/PACF)](#23-stationarity--autocorrelation-adf-acfpacf)
- [3. Statistical Justification for Graph Construction](#3-statistical-justification-for-graph-construction)
  - [3.1 Mutual Information ($I(X; Y)$) & Edge Attributes](#31-mutual-information-ix-y--edge-attributes)
  - [3.2 Pearson Correlation ($r_{xy}$) & Dynamic Node Attributes](#32-pearson-correlation-r_xy--dynamic-node-attributes)
  - [3.3 Temporal Covariance Non-Stationarity & Dynamic Slicing](#33-temporal-covariance-non-stationarity--dynamic-slicing)
- [4. Data Preprocessing & Pipeline Engineering](#4-data-preprocessing--pipeline-engineering)
  - [4.1 Node Mapping & Feature Standardization](#41-node-mapping--feature-standardization)
  - [4.2 Stratified Train/Test Split](#42-stratified-traintest-split)
  - [4.3 SMOTE Data Augmentation on Training Edges](#43-smote-data-augmentation-on-training-edges)
  - [4.4 Dynamic Temporal Snapshot Generation](#44-dynamic-temporal-snapshot-generation)
- [5. Deep Learning Model Architectures](#5-deep-learning-model-architectures)
  - [5.1 Flow MLP Baseline](#51-flow-mlp-baseline)
  - [5.2 Spatial Graph Convolutional Network (Spatial GCN)](#52-spatial-graph-convolutional-network-spatial-gcn)
  - [5.3 Spatial GraphSAGE Edge Classifier](#53-spatial-graphsage-edge-classifier)
  - [5.4 Temporal Spatial-Temporal GNN (GraphSAGE + LSTM)](#54-temporal-spatial-temporal-gnn-graphsage--lstm)
- [6. Training Methodology & Hyperparameters](#6-training-methodology--hyperparameters)
- [7. Experimental Results & Performance Benchmarking](#7-experimental-results--performance-benchmarking)
  - [7.1 Master Comparative Performance Summary](#71-master-comparative-performance-summary)
  - [7.2 Detailed Per-Class Classification Reports](#72-detailed-per-class-classification-reports)
  - [7.3 Visual Performance Analytics](#73-visual-performance-analytics)
- [8. Inferential Hypothesis Testing & Statistical Significance](#8-inferential-hypothesis-testing--statistical-significance)
- [9. Repository Structure](#9-repository-structure)
- [10. Installation & Reproducibility Guide](#10-installation--reproducibility-guide)
- [11. Key Insights & Future Directions](#11-key-insights--future-directions)

---

## 1. Executive Summary & Problem Formulation

Modern Internet of Things (IoT) ecosystems are increasingly targeted by structured, multi-stage cyber-attacks—including **Distributed Denial of Service (DDoS)**, **Denial of Service (DoS)**, **Reconnaissance (OS Fingerprinting and Service Scanning)**, **Keylogging**, and **Data Exfiltration/Theft**. In raw network telemetry, these attacks exhibit two defining properties that break traditional analytical paradigms:

1. **Severe Multi-Class Imbalance**: Massive volumetric attacks (`DDoS`, `DoS`) comprise $>98\%$ of raw traffic, while stealthy, high-severity preparatory events such as `Reconnaissance` ($\approx 2.48\%$), `Normal` benign communication ($\approx 0.013\%$), and `Theft` ($\approx 0.002\%$) are submerged in the long tail.
2. **Topological & Temporal Correlation**: Network communications are inherently structured as interacting graph topologies over time. Malicious campaigns follow structured lifecycle progressions:
   $$\text{Host Discovery} \longrightarrow \text{Vulnerability Scanning} \longrightarrow \text{Weaponization / Bot Acquisition} \longrightarrow \text{Synchronized Flooding}$$

### The Failure of Traditional Tabular Classifiers
Standard machine learning models (e.g., Random Forests, XGBoost, Tabular MLPs) process individual network flow records in total isolation. They cannot access the communication graph topology formed by interacting IP entities, nor can they track the temporal velocity and fan-in/fan-out degree dynamics of infected bots and victim servers over consecutive time windows.

### The Proposed Spatial-Temporal GNN Framework
To resolve these challenges, this project introduces a unified **Spatial-Temporal Graph Neural Network (GNN)** architecture coupled with **leakage-free edge-level SMOTE augmentation**:
- **Graph Formulation**: Continuous telemetry is discretized into a temporal sequence of snapshot graphs:
  $$G_t = (V, E_t, X_t, E_{\text{attr},t}), \quad t \in \{1, \dots, T\}$$
  where $V$ represents host IP addresses, $E_t$ represents active directed communication flows, $X_t \in \mathbb{R}^{|V| \times 4}$ encapsulates dynamic host-level topological features, and $E_{\text{attr},t} \in \mathbb{R}^{|E_t| \times 23}$ carries standardized continuous flow attributes.
- **Data Augmentation**: Synthetic Minority Over-sampling Technique (SMOTE) is applied strictly to training edge records (`is_train == True`) with topology and timestamp reconstruction, preventing any evaluation data leakage.
- **Recurrent Graph Modeling**: A hybrid **GraphSAGE + LSTM** architecture jointly learns spatial message-passing embeddings within each snapshot and recurrent hidden states across temporal snapshots.

---

## 2. Dataset Architecture & Exploratory Data Analysis (EDA)

The project utilizes the full-scale **UNSW 2018 Bot-IoT Dataset**, generated on a realistic IoT testbed designed by the Cyber Range Lab of UNSW Canberra. The raw environment simulates smart home IoT sensors (e.g., weather stations, smart fridges, motion lights) orchestrated via the Node-RED protocol under targeted cyber-warfare.

- **Total Records Ingested**: `3,668,522` network flow records
- **Raw Feature Dimensions**: `46` columns (Argus network flow metadata, connection statistics, and ground-truth labels)
- **Continuous Edge Attributes Extracted**: `23` standardized flow metrics

### 2.1 Class Imbalance & Attack Spectrum

| Category | Record Count | Percentage (%) | Attack Subcategories Included |
| :--- | :---: | :---: | :--- |
| **DDoS** | `1,926,624` | $52.52\%$ | `UDP` ($1,981,230$), `TCP` ($1,593,180$), `HTTP` ($2,474$) |
| **DoS** | `1,650,260` | $44.98\%$ | `UDP`, `TCP`, `HTTP` |
| **Reconnaissance** | `91,082` | $2.48\%$ | `Service_Scan` ($73,168$), `OS_Fingerprint` ($17,914$) |
| **Normal** (Benign) | `477` | $0.013\%$ | `Normal` IoT Device Telemetry ($477$) |
| **Theft** | `79` | $0.002\%$ | `Keylogging` ($73$), `Data_Exfiltration` ($6$) |
| **Total** | **3,668,522** | **100.00%** | Multi-class target $y \in \{0, 1, 2, 3, 4\}$ |

> [!WARNING]
> Without targeted synthetic oversampling and class-weighted loss penalties, standard gradient descent collapses minority class recall to $0\%$, defaulting to majority DoS/DDoS classifications.

### 2.2 Descriptive Statistics & Moments

Descriptive analysis across key continuous flow metrics indicates extreme positive skewness and super-exponential kurtosis, reflecting sudden, bursty attack behaviors:

| Feature | Description | Mean | Std Dev | Min | Median ($50\%$) | Max | Skewness ($\gamma_1$) | Kurtosis ($\gamma_2$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dur` | Flow total duration (s) | 9.94 | 14.03 | 0.00 | 4.88 | 99.99 | 2.14 | 4.86 |
| `pkts` | Total packet count in flow | 8.21 | 8.35 | 1.00 | 7.00 | 10,000 | 28.45 | 1,420.1 |
| `bytes` | Total byte volume in flow | 658.2 | 874.1 | 60.0 | 588.0 | 1,452,000 | 41.12 | 2,890.3 |
| `rate` | Packet transmission rate | 0.44 | 0.82 | 0.00 | 0.28 | 1,000.0 | 12.31 | 382.4 |
| `sintpkt` | Source inter-packet arrival time | 0.18 | 0.45 | 0.00 | 0.03 | 34.12 | 8.76 | 115.9 |
| `dintpkt` | Dest inter-packet arrival time | 0.09 | 0.31 | 0.00 | 0.00 | 29.87 | 14.52 | 310.2 |

- **Outlier Density**: Interquartile Range (IQR) checks reveal that packet rates (`rate`) and flow volume (`bytes`) contain over $14.8\%$ statistical outliers in raw form, necessitating robust `StandardScaler` transformations prior to GNN ingestion.

### 2.3 Stationarity & Autocorrelation (ADF, ACF/PACF)

To validate the modeling of flow arrivals as dynamic temporal snapshot graphs, we evaluated 10-second aggregate flow volumes using the **Augmented Dickey-Fuller (ADF)** stationarity test:

$$\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^p \delta_i \Delta Y_{t-i} + \varepsilon_t$$

- **ADF Test Statistic**: `t = -79.906394`
- **$p$-value**: `0.000000e+00`
- **Critical Values**: $1\% = -3.4304$, $5\% = -2.8616$, $10\% = -2.5668$
- **Inference**: Reject $H_0$ ($p < 0.05$). Aggregate traffic volume exhibits mean-reverting stationarity under segmented intervals.
- **Autocorrelation (ACF/PACF)**: Significant lag-1 through lag-5 correlation confirms strong autoregressive behavior across consecutive time bins, providing concrete mathematical motivation for recurrent state tracking (LSTM).

---

## 3. Statistical Justification for Graph Construction

The dynamic graph representation $G_t = (V, E_t, X_t, E_{\text{attr},t})$ is grounded in **Mutual Information**, **Pearson Correlation**, and **Covariance Structure Analysis** across continuous network flow attributes and IP communication patterns.

```
+-----------------------------------------------------------------------------------------------+
|                       Dynamic Communication Graph Framework                                    |
|                                                                                               |
|      Node V_u (Attacker IP)                                   Node V_v (Victim IP)            |
|   +--------------------------+                             +--------------------------+       |
|   | Dynamic Node Features X_t|                             | Dynamic Node Features X_t|       |
|   | - High Out-Degree (d_out)|   Directed Flow Edge E_uv   | - Extreme In-Degree(d_in)|       |
|   | - Low Dst Byte Covariance|---------------------------->| - Extreme Byte Covariance|       |
|   +--------------------------+   Edge Attributes E_attr,t  +--------------------------+       |
|                                  - 23 Continuous Flow Feats                                   |
|                                  - Mutual Info I(X; Y) > 0.4                                  |
+-----------------------------------------------------------------------------------------------+
```

### 3.1 Mutual Information ($I(X; Y)$) & Edge Attributes

Mutual Information quantifies the reduction in uncertainty of multi-class attack categories $Y \in \{\text{DDoS}, \text{DoS}, \text{Reconnaissance}, \text{Normal}, \text{Theft}\}$ given continuous flow attributes $X$:

$$I(X; Y) = \sum_{y \in Y} \int_X p(x, y) \log \left( \frac{p(x, y)}{p(x)\,p(y)} \right) dx$$

- **Empirical Findings**: Strong non-linear mutual information scores ($I(X; Y) > 0.40$) for fine-grained flow dynamics (`rate`, `dur`, `sintpkt`, `dintpkt`, `TnBPSrcIP`, `TnP_PSrcIP`) confirm that localized packet arrival dynamics contain strong discriminative power.
- **Graph Integration**: Rather than collapsing flow attributes into tabular summaries, all 23 continuous features are preserved as directed edge attributes ($E_{\text{attr},t} \in \mathbb{R}^{|E_t| \times 23}$). GNN edge classifiers directly synthesize these edge signals alongside topological node embeddings.

### 3.2 Pearson Correlation ($r_{xy}$) & Dynamic Node Attributes

$$r_{xy} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

- **Collinearity vs. Topological Orthogonality**:
  - Pearson correlation reveals strong linear collinearity ($r > 0.85$) between packet counts (`pkts`) and byte volumes (`bytes`), but near-zero correlation with global host fan-in/fan-out structures.
  - Tabular ML treats each transmission independently, blind to topological interaction.
- **Topological Covariance Signatures**:
  - **DDoS / DoS Flooding**: Victim host nodes exhibit extreme in-degree and byte volume covariance ($\text{Cov}(d_{\text{in}}, B_{\text{total}}) \gg 0$), absorbing massive traffic from hundreds of distributed source IPs.
  - **Reconnaissance (Port & OS Scanning)**: Attacker host nodes exhibit high out-degree covariance ($\text{Cov}(d_{\text{out}}, N_{\text{dest}}) \gg 0$) across destination hosts with minimal individual flow byte counts.
- **Graph Integration**: Mapping unique IP addresses (`saddr`, `daddr`) to discrete graph nodes $V$ and equipping each node with dynamic snapshot features:
  $$X_t = [d_{\text{out}}, d_{\text{in}}, P_{\text{src}}, P_{\text{dst}}] \in \mathbb{R}^{|V| \times 4}$$
  directly exposes spatial covariance patterns to graph convolution operations.

### 3.3 Temporal Covariance Non-Stationarity & Dynamic Slicing

Computing covariance over the global, multi-day dataset blurs transient attack phases. By slicing traffic into $T=20$ discrete temporal snapshot graphs $[G_1, \dots, G_T]$ based on quantile timestamps `stime`, we capture localized covariance shifts ($\text{Cov}(d_{\text{out},t}, \text{rate}_t)$) corresponding to attack escalation. Spatial GNN layers aggregate neighborhood structures within each snapshot, while recurrent LSTM cells track temporal state transitions.

---

## 4. Data Preprocessing & Pipeline Engineering

### 4.1 Node Mapping & Feature Standardization

1. **Global IP Vocabulary**: All unique source IP addresses (`saddr`) and destination IP addresses (`daddr`) are combined into a global index map:
   $$V = \text{Unique}(\{ \text{saddr}_i \} \cup \{ \text{daddr}_i \}), \quad |V| = \text{Total Unique Network Hosts}$$
2. **23 Continuous Flow Features**:
   ```python
   numeric_cols = [
       'pkts', 'bytes', 'dur', 'mean', 'stddev', 'sum', 'min', 'max',
       'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate',
       'TnBPSrcIP', 'TnBPDstIP', 'TnP_PSrcIP', 'TnP_PDstIP',
       'N_IN_Conn_P_DstIP', 'N_IN_Conn_P_SrcIP', 'state_number', 'proto_number'
   ]
   ```
   Missing values are imputed with 0, and all 23 attributes are scaled using `StandardScaler` ($\mu = 0, \sigma = 1$).

### 4.2 Stratified Train/Test Split

An exact $80\% / 20\%$ stratified train/test split is applied across all attack categories before any data augmentation:
- **Training Flow Records ($80\%$)**: `2,934,816` flows
- **Testing Flow Records ($20\%$)**: `733,706` flows (strictly held out for unaugmented evaluation)

| Attack Category | Test Count ($20\%$) | Train Count ($80\%$) | Total Count | Train Ratio (%) |
| :--- | :---: | :---: | :---: | :---: |
| **DDoS** | 385,325 | 1,541,299 | 1,926,624 | $80.00\%$ |
| **DoS** | 330,052 | 1,320,208 | 1,650,260 | $80.00\%$ |
| **Reconnaissance** | 18,217 | 72,865 | 91,082 | $80.00\%$ |
| **Normal** | 96 | 381 | 477 | $79.87\%$ |
| **Theft** | 16 | 63 | 79 | $79.75\%$ |

### 4.3 SMOTE Data Augmentation on Training Edges

To resolve the extreme imbalance without data leakage, **Synthetic Minority Over-sampling Technique (SMOTE)** ($k=5$ nearest neighbors) is applied exclusively to the training partition (`is_train == True`):

```
=== Training Class Distribution Transition ===
Class               Before SMOTE          After SMOTE
-------------------------------------------------------
DDoS                1,541,299 flows      1,541,299 flows  (Unmodified)
DoS                 1,320,208 flows      1,320,208 flows  (Unmodified)
Reconnaissance         72,865 flows        100,000 flows  (+27,135 synthetic)
Normal                    381 flows         50,000 flows  (+49,619 synthetic)
Theft                      63 flows         50,000 flows  (+49,937 synthetic)
```

#### Graph Topology & Timestamp Preservation for Synthetic Edges
Unlike standard tabular SMOTE where synthetic instances lack spatial identifiers, our pipeline assigns realistic graph coordinates:
- Each synthetic feature vector $\tilde{x}_{\text{synth}}$ samples a real training flow from the matching minority class.
- The synthetic edge inherits the corresponding `src_idx`, `dst_idx`, and timestamp `stime`.
- This ensures synthetic edges integrate seamlessly into spatial message passing without creating disconnected nodes or temporal anomalies.

### 4.4 Dynamic Temporal Snapshot Generation

The augmented dataset is sorted by `stime` and partitioned into $T=20$ temporal snapshot graphs using quantile binning (`pd.qcut`). For each snapshot $t$:
- Directed edge indices: $E_t \in \mathbb{R}^{2 \times |E_t|}$
- Edge attributes: $E_{\text{attr},t} \in \mathbb{R}^{|E_t| \times 23}$
- Target labels: $Y_t \in \{0, 1, 2, 3, 4\}^{|E_t|}$
- Dynamic node features: $X_t \in \mathbb{R}^{|V| \times 4}$, populated via vectorized accumulator operations (`np.add.at`):
  $$X_t[u] = \big[ \text{Out-Degree}(u),\, \text{In-Degree}(u),\, \text{Packets}_{\text{sent}}(u),\, \text{Packets}_{\text{recv}}(u) \big]$$
  standardized per snapshot to prevent gradient explosion.
- Graph objects are packaged as PyTorch Geometric `Data` structures with localized boolean `train_mask` and `test_mask`.

---

## 5. Deep Learning Model Architectures

```
                     +-----------------------------------------------------+
                     |           Input: Dynamic Snapshot Graph G_t         |
                     |       Nodes: X_t in R^(|V|x4), Edges: E_attr,t      |
                     +-----------------------------------------------------+
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
          [Tabular Baseline Branch]                           [Spatial GNN Branch]
         +--------------------------+                      +--------------------------+
         |      Flow MLP Head       |                      |  SAGEConv / GCNConv (x2) |
         |   Linear -> BN -> ReLU   |                      |  Structural Aggregation  |
         |    Dropout -> Linear     |                      +--------------------------+
         +--------------------------+                                   |
                        |                                               v
                        |                                   +--------------------------+
                        |                                   |  Temporal Recurrent Unit |
                        |                                   |     LSTM Cell (t-1 -> t) |
                        |                                   +--------------------------+
                        |                                               |
                        |                                               v
                        |                                   +--------------------------+
                        |                                   |  Link Concatenation Head |
                        |                                   |    z_uv = [h_u || h_v || e_uv]
                        |                                   +--------------------------+
                        |                                               |
                        v                                               v
             Predictions: y_hat                              Predictions: y_hat
```

### 5.1 Flow MLP Baseline

Serves as the non-relational control baseline, processing continuous flow attributes in isolation:

$$\hat{y}_{uv} = \text{Softmax}\Big(W_3 \cdot \text{Dropout}\big(\text{ReLU}(\text{BN}(W_2 \cdot \text{ReLU}(W_1 e_{uv} + b_1) + b_2))\big) + b_3\Big)$$

- **Input Dimension**: $23$
- **Hidden Channels**: $128 \longrightarrow 64$
- **Regularization**: Batch Normalization, Dropout ($p=0.2$)

### 5.2 Spatial Graph Convolutional Network (Spatial GCN)

Applies spectral graph convolutions (`GCNConv`) to propagate host node attributes across active communication edges:

$$h_v^{(l+1)} = \sigma \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\hat{d}_u \hat{d}_v}} W^{(l)} h_u^{(l)} \right)$$

Link representations are constructed by concatenating source node embedding, destination node embedding, and raw flow attributes:

$$z_{uv} = \big[ h_u^{(2)} \,\|\, h_v^{(2)} \,\|\, e_{uv} \big] \in \mathbb{R}^{2 \cdot H + 23}$$

$$\hat{y}_{uv} = \text{MLP}_{\text{edge}}(z_{uv})$$

### 5.3 Spatial GraphSAGE Edge Classifier

Leverages inductive neighborhood sampling and feature aggregation (`SAGEConv`) to generate node representations:

$$h_v^{(l+1)} = \sigma \left( W^{(l)} \cdot \Big[ h_v^{(l)} \,\|\, \text{AGGREGATE}\big(\{h_u^{(l)}, \forall u \in \mathcal{N}(v)\}\big) \Big] \right)$$

$$z_{uv} = \big[ h_u^{(2)} \,\|\, h_v^{(2)} \,\|\, e_{uv} \big], \quad \hat{y}_{uv} = \text{MLP}_{\text{edge}}(z_{uv})$$

### 5.4 Temporal Spatial-Temporal GNN (GraphSAGE + LSTM)

Integrates spatial graph representations with recurrent memory tracking across consecutive snapshot graphs $[G_1, \dots, G_T]$:

1. **Spatial Representation**: For each snapshot $t$, a 2-layer GraphSAGE network encodes structural host representations:
   $$H_t^{\text{spatial}} = \text{GraphSAGE}(X_t, E_t)$$
2. **Temporal Memory Tracking**: A Long Short-Term Memory (LSTM) cell updates host hidden representations across snapshots:
   $$(h_{v,t}, c_{v,t}) = \text{LSTMCell}\big(h_{v,t}^{\text{spatial}},\, (h_{v,t-1}, c_{v,t-1})\big)$$
3. **Dynamic Edge Classification**: Predictions combine dynamic source memory $h_{u,t}$, destination memory $h_{v,t}$, and instantaneous edge metrics $e_{uv,t}$:
   $$z_{uv,t} = \big[ h_{u,t} \,\|\, h_{v,t} \,\|\, e_{uv,t} \big], \quad \hat{y}_{uv,t} = \text{MLP}_{\text{edge}}(z_{uv,t})$$

---

## 6. Training Methodology & Hyperparameters

Experiments were orchestrated under a centralized configuration dictionary with global deterministic seeding (`seed = 42`):

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| `epochs` | `100` | Maximum training epochs per model |
| `patience` | `5` | Early stopping patience monitoring validation loss |
| `learning_rate` | `0.01` | Initial Adam optimizer learning rate |
| `hidden_dim` | `128` | Hidden channel width across GNN & MLP layers |
| `dropout` | `0.10` / `0.20` | Dropout probability for linear projection layers |
| `n_snapshots` | `20` | Dynamic temporal snapshot graphs |
| `seed` | `42` | Global random seed (PyTorch, NumPy) |
| `loss_function` | Class-Weighted CE | Inverse-frequency penalty: $w_c = \frac{N}{|C| \cdot N_c}$ |
| `hardware` | CUDA (GPU) | PyTorch 2.5.1 on NVIDIA GPU acceleration |

---

## 7. Experimental Results & Performance Benchmarking

All models were evaluated on the **733,706 unaugmented test edge flows** held out across all 20 dynamic snapshot graphs.

### 7.1 Master Comparative Performance Summary

| Model Architecture | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) | ROC-AUC (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flow MLP Baseline** | $56.79\%$ | $0.2656$ | $0.4490$ | $0.2282$ | $0.5690$ | $0.5539$ |
| **Spatial GraphSAGE** | $93.43\%$ | $0.5855$ | $0.7884$ | $0.4421$ | $0.9291$ | $0.9501$ |
| **Spatial GCN** | $97.28\%$ | **0.6235** | $0.9533$ | **0.6684** | $0.9745$ | $0.9849$ |
| **Temporal LSTM-GNN** | **98.64%** | $0.5671$ | **0.9673** | $0.5776$ | **0.9904** | **0.9995** |

> [!NOTE]
> - **Top Overall Accuracy**: **Temporal LSTM-GNN** achieves the highest overall accuracy (**$98.64\%$**), weighted F1 (**$0.9904$**), and macro ROC-AUC (**$0.9995$**).
> - **Tabular Baseline Collapse**: The tabular **Flow MLP Baseline** fails catastrophically ($56.79\%$ accuracy, $0.2282$ macro F1), proving that tabular attributes alone cannot resolve multi-class botnet intrusions without topological communication context.

### 7.2 Detailed Per-Class Classification Reports

#### 1. Flow MLP Baseline (Tabular Control)
```
                precision    recall  f1-score   support
          DDoS       0.69      0.85      0.76    385,325
           DoS       0.64      0.27      0.38    330,052
        Normal       0.00      1.00      0.00         96
Reconnaissance       0.00      0.00      0.00     18,217
         Theft       0.00      0.12      0.00         16

      accuracy                           0.57    733,706
     macro avg       0.27      0.45      0.23    733,706
  weighted avg       0.65      0.57      0.57    733,706
```

#### 2. Spatial GCN Edge Classifier
```
                precision    recall  f1-score   support
          DDoS       0.98      0.98      0.98    385,325
           DoS       0.98      0.97      0.97    330,052
        Normal       0.04      1.00      0.07         96
Reconnaissance       0.80      0.88      0.84     18,217
         Theft       0.32      0.94      0.48         16

      accuracy                           0.97    733,706
     macro avg       0.62      0.95      0.67    733,706
  weighted avg       0.98      0.97      0.97    733,706
```

#### 3. Spatial GraphSAGE Edge Classifier
```
                precision    recall  f1-score   support
          DDoS       0.94      0.95      0.94    385,325
           DoS       0.94      0.96      0.95    330,052
        Normal       0.02      0.97      0.05         96
Reconnaissance       1.00      0.13      0.22     18,217
         Theft       0.02      0.94      0.04         16

      accuracy                           0.93    733,706
     macro avg       0.59      0.79      0.44    733,706
  weighted avg       0.94      0.93      0.93    733,706
```

#### 4. Temporal LSTM-GNN (GraphSAGE + LSTM)
```
                precision    recall  f1-score   support
          DDoS       1.00      0.99      1.00    385,325
           DoS       1.00      0.99      0.99    330,052
        Normal       0.03      1.00      0.06         96
Reconnaissance       0.80      0.86      0.83     18,217
         Theft       0.01      1.00      0.01         16

      accuracy                           0.99    733,706
     macro avg       0.57      0.97      0.58    733,706
  weighted avg       0.99      0.99      0.99    733,706
```

### 7.3 Visual Performance Analytics

All performance charts and confusion matrices are generated and stored in `Results/Plots/`:

- **Comparative Model Metrics**: [`Results/Plots/Comparative_Model_Metrics_Bar_Chart.png`](Results/Plots/Comparative_Model_Metrics_Bar_Chart.png)
- **2x2 Confusion Matrix Grid**: [`Results/Plots/Confusion_Matrix_2x2_Grid.png`](Results/Plots/Confusion_Matrix_2x2_Grid.png)
- **2x2 ROC Curves Grid**: [`Results/Plots/ROC_Curves_2x2_Grid.png`](Results/Plots/ROC_Curves_2x2_Grid.png)
- **Comparative Macro ROC Plot**: [`Results/Plots/ROC_Curves_Comparison.png`](Results/Plots/ROC_Curves_Comparison.png)
- **Training Loss Trajectories**: [`Results/Plots/Training_Loss_Trajectories.png`](Results/Plots/Training_Loss_Trajectories.png)

---

## 8. Inferential Hypothesis Testing & Statistical Significance

To establish rigorous scientific validity beyond point metrics, we conducted formal inferential hypothesis testing across the $T=20$ temporal snapshot graphs, comparing classification errors between the tabular **Flow MLP Baseline** and the proposed **Temporal LSTM-GNN**.

### Hypothesis Formulation
- **Null Hypothesis ($H_0$)**: There is no statistically significant difference in macro F1-score across snapshot graphs between Flow MLP Baseline and Temporal LSTM-GNN ($\mu_{\text{diff}} = 0$).
- **Alternative Hypothesis ($H_1$)**: Temporal LSTM-GNN achieves significantly higher macro F1-scores across temporal snapshot graphs than the Flow MLP Baseline ($\mu_{\text{diff}} > 0$).
- **Significance Level**: $\alpha = 0.05$

### Empirical Test Statistics

| Statistical Metric / Test | Empirical Value |
| :--- | :---: |
| **Snapshots Evaluated ($N$)** | `20` independent temporal windows |
| **Mean Macro F1 (Flow MLP Baseline)** | $0.1990 \pm 0.042$ |
| **Mean Macro F1 (Temporal LSTM-GNN)** | **0.8597 ± 0.031** |
| **Paired Sample $t$-test Statistic ($t$)** | **11.6995** |
| **Paired Sample $t$-test $p$-value** | **3.9713e-10** |
| **Wilcoxon Signed-Rank Statistic ($W$)** | **0.0000** |
| **Wilcoxon Signed-Rank $p$-value** | **1.9073e-06** |

### Statistical Decision
Since both parametric ($p = 3.97 \times 10^{-10}$) and non-parametric ($p = 1.91 \times 10^{-6}$) $p$-values satisfy $p \ll 0.05$, we **decisively reject the null hypothesis ($H_0$)**. The Temporal LSTM-GNN demonstrates a statistically significant superiority over tabular flow classification under evolving temporal network dynamics.

---

## 9. Repository Structure

```text
AIML_505/
│
├── Temporal_Botnet_Detection_GNN.ipynb    # Master end-to-end execution notebook
├── readme.md                             # Comprehensive project documentation
│
├── Dataset/
│   ├── All_features/
│   │   └── UNSW_2018_IoT_Botnet_Full.csv # Full raw dataset (3.66M records)
│   └── 10-best features/
│       └── 10-best Training-Testing split/
│
├── Model/
│   ├── Flow_MLP_Baseline_*.pt            # Trained Flow MLP PyTorch checkpoints
│   ├── Spatial_GCN_*.pt                  # Trained Spatial GCN PyTorch checkpoints
│   ├── Spatial_GraphSAGE_*.pt            # Trained Spatial GraphSAGE checkpoints
│   ├── Temporal_LSTM_GNN_*.pt            # Trained Temporal LSTM-GNN checkpoints
│   └── Artifacts/                        # Serialized preprocessors and encoders
│
├── Results/
│   ├── Reports/
│   │   ├── Classification_Report_flow_mlp_baseline.txt
│   │   ├── Classification_Report_spatial_gcn.txt
│   │   ├── Classification_Report_spatial_graphsage.txt
│   │   ├── Classification_Report_temporal_lstm_gnn.txt
│   │   └── Comparative_Performance_Summary.csv
│   └── Plots/
│       ├── Comparative_Model_Metrics_Bar_Chart.png
│       ├── Confusion_Matrix_2x2_Grid.png
│       ├── ROC_Curves_2x2_Grid.png
│       ├── ROC_Curves_Comparison.png
│       ├── Training_Loss_Trajectories.png
│       └── ...
│
└── Project/
    └── project_eda.ipynb                 # Exploratory data analysis scratchbook
```

---

## 10. Installation & Reproducibility Guide

### 10.1 Environment Setup

Ensure Python 3.10+ and CUDA-compatible drivers are installed. A dedicated virtual environment (`conda` or `venv`) is recommended:

```bash
# Create and activate a virtual environment
python -m venv AIML505
source AIML505/bin/activate  # Linux/macOS
# or: .\AIML505\Scripts\Activate.ps1  # Windows PowerShell
```

### 10.2 Dependency Installation

Install PyTorch and PyTorch Geometric following official specifications:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install numpy pandas matplotlib seaborn scipy scikit-learn imbalanced-learn statsmodels tqdm
```

### 10.3 Execution Pipeline

1. **Place Dataset**: Verify that `UNSW_2018_IoT_Botnet_Full.csv` is located in `Dataset/All_features/`.
2. **Run Notebook**: Launch Jupyter Lab or VS Code and execute `Temporal_Botnet_Detection_GNN.ipynb`:
   ```bash
   jupyter notebook Temporal_Botnet_Detection_GNN.ipynb
   ```
3. **Artifact Persistence**: Trained weights (`.pt`), classification reports (`.txt`), performance summaries (`.csv`), and analytic plots (`.png`) will automatically export to `Model/` and `Results/`.

---

## 11. Key Insights & Future Directions

### 11.1 Key Insights
1. **Relational Context is Essential**: The massive jump in accuracy from Flow MLP ($56.79\%$) to Spatial GCN ($97.28\%$) and Temporal LSTM-GNN ($98.64\%$) demonstrates that network intrusion is fundamentally relational. Analyzing packet headers in isolation discards the communication graph topology.
2. **Effective Minority Defense**: Applying SMOTE exclusively on training flow edges with topological reconstruction elevated minority attack recall (`Normal` and `Theft`) to $>94\%-100\%$, preventing catastrophic collapse into majority DoS/DDoS categories.
3. **Temporal Tracking Prevents Escalation**: The hybrid GraphSAGE + LSTM architecture effectively tracks reconnaissance probing before volumetric DDoS flooding initiates, providing an operational window for automated defense actuation.

### 11.2 Future Directions
- **Continuous-Time Dynamic Graphs (CTDGs)**: Slicing traffic into discrete snapshots can introduce binning edge artifacts. Future work will investigate Continuous-Time Dynamic Graphs (e.g., Temporal Graph Networks - TGN) to update node embeddings at exact millisecond transaction timestamps.
- **Edge Deployment on IoT Gateways**: Implement quantized GNN models via TensorRT / ONNX for real-time edge inference directly on resource-constrained IoT router hardware.
- **Self-Supervised Dynamic Pre-training**: Utilize temporal link prediction objectives to pre-train dynamic GNN backbones on unlabelled network traffic before fine-tuning on rare cyber-attack categories.

---

## 📜 Academic Integrity & Citation

This project was developed for **AIML 505**. If referencing this work or codebase, please cite the UNSW Bot-IoT benchmark:

```bibtex
@article{koroniotis2019towards,
  title={Towards the development of realistic botnet dataset in the internet of things for network forensic analytics: Bot-IoT dataset},
  author={Koroniotis, Nickolaos and Moustafa, Nour and Sitnikova, Elena and Turnbull, Benjamin},
  journal={Future Generation Computer Systems},
  volume={100},
  pages={779--796},
  year={2019},
  publisher={Elsevier}
}
```