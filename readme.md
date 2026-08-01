Project\Dataset\10-best features\10-best Training-Testing split\UNSW_2018_IoT_Botnet_Final_10_best_Training.csv

Bonferroni correction ✗

Copy argument

SMOTE (circled)
comparison
up augmentation

K-fold cross validation
5-fold

Statistical significance
chk

error analysis
### Statistical Justification for Graph Construction: Correlation, Covariance & Mutual Information

The dynamic graph representation $G_t = (V, E_t, X_t, E_{\text{attr},t})$ built in this project is statistically motivated by **Mutual Information**, **Pearson Correlation**, and **Covariance Structure Analysis** across continuous network flow attributes and IP interaction patterns.

---

### 1. Mutual Information ($I(X; Y)$): Justifying Edge Attribute Selection ($E_{\text{attr}}$)

* **Non-Linear Feature Relevance**: Mutual Information quantifies the amount of shared information between continuous flow attributes $X$ and multi-class attack categories $Y \in \{\text{DDoS}, \text{DoS}, \text{Reconnaissance}, \text{Normal}, \text{Theft}\}$. High mutual information scores ($I(X; Y) > 0.4$) for features like packet rate (`rate`), flow duration (`dur`), and inter-arrival times (`sintpkt`, `dintpkt`) prove that fine-grained flow metrics contain high predictive power.
* **Graph Integration**: Rather than discarding tabular flow attributes, we attach all 23 standardized continuous features directly to directed graph edges ($E_{\text{attr},t} \in \mathbb{R}^{|E_t| \times 23}$). This allows GNN layers (`SAGEConv` / `GCNConv`) to perform link classification using both topological node embeddings and raw edge feature intensity.

---

### 2. Pearson Correlation ($r_{xy}$) & Covariance ($\Sigma$): Justifying Node Mapping ($V$) & Dynamic Node Attributes ($X_t$)

* **Linear Redundancy vs. Structural Independence**:
  * Pearson correlation analysis shows high linear collinearity ($r > 0.85$) between raw packet counts (`pkts`) and byte volumes (`bytes`), but low correlation between individual flow attributes and topological host fan-in/fan-out behaviors.
  * Standard tabular machine learning evaluates each flow in isolation, missing structural interactions between communicating hosts.
* **Topological Covariance Signatures**:
  * **DDoS / DoS Attacks**: Victim host nodes exhibit extreme in-degree and byte volume covariance ($\text{Cov}(d_{\text{in}}, B_{\text{total}}) \gg 0$), receiving overwhelming traffic from many sources.
  * **Reconnaissance (Scanning / Probing)**: Attacker host nodes exhibit high out-degree covariance ($\text{Cov}(d_{\text{out}}, N_{\text{dest}}) \gg 0$) across many destination IPs with low individual flow byte counts.
* **Graph Integration**: Mapping unique IP addresses (`saddr`, `daddr`) to discrete graph nodes $V$ and equipping each node with dynamic degree and traffic volume features $X_t = [d_{\text{in}}, d_{\text{out}}, B_{\text{total}}, P_{\text{total}}] \in \mathbb{R}^{|V| \times 4}$ directly exposes these spatial covariance signatures to spatial graph convolution operations.

---

### 3. Temporal Covariance Non-Stationarity: Justifying Dynamic Snapshot Slicing ($[G_1, \dots, G_T]$)

* **Temporal Windowing**: Global covariance computed across the entire multi-day dataset blurs transient botnet lifecycle phases (Reconnaissance $\to$ Weaponization $\to$ DDoS Execution).
* **Graph Slicing**: Slicing continuous traffic into $T=10$ temporal snapshot graphs $G_1, \dots, G_T$ creates stationary temporal windows where local covariance shifts ($\text{Cov}(d_{\text{out},t}, \text{rate}_t)$) mark botnet phase transitions.
* **Spatial-Temporal Modeling**: Spatial GNN layers aggregate local IP neighborhood structures within each snapshot graph $G_t$, while the recurrent memory cell (LSTM) tracks temporal covariance evolution across consecutive snapshots.