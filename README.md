# End-to-End Predictive Maintenance Pipeline (C-MAPSS)

An end-to-end Big Data and Machine Learning pipeline for turbofan engine predictive maintenance using Apache Spark (PySpark), SparkSQL, Spark MLlib, and Structured Streaming.

This project predicts the **Remaining Useful Life (RUL)** of turbofan engines using NASA's benchmark **C-MAPSS** dataset (subset FD001: single operating condition, high-pressure compressor degradation).

---

## Architecture Overview (Medallion Architecture)

```
Raw Space-Delimited .txt
       │
       ▼
 [ Part 1: Bronze Layer ]  ──► Typed, structured Parquet storage
       │
       ▼
 [ Part 2: Silver Layer ]  ──► Cleaned sensors & piece-wise linear RUL labels (capped at 125)
       │
       ▼
 [ Part 3: Gold Layer ]    ──► Feature Engineering (43 features: raw + rolling means + rolling stds)
       │
       ├────────────────────────────────────────┬────────────────────────────────────────┐
       ▼                                        ▼                                        ▼
[ Part 4: MLlib Regression ]          [ Part 5: Structured Streaming ]          [ Part 6: Lab 6 Hand-off ]
- Linear Regression                   - Simulated JSON micro-batches             - Min-Max scaled features [0, 1]
- Random Forest (Best: RMSE 18.17)    - ForeachBatch incremental scoring        - Export normalization.json &
- Gradient-Boosted Trees (GBT)        - Real-time alerts (RUL < 30 cycles)        baseline_metrics.json for Deep Learning
```

---

## Pipeline Stages

### Part 1 — Ingestion & Bronze Layer
- **Source:** Raw space-delimited text files (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`).
- **Processing:** Handled trailing whitespace using regex tokenization (`\s+`), cast 26 columns (`unit_id`, `cycle`, 3 operational settings, and 21 sensors) to explicit types (`int`, `double`), and added tracking metadata (`dataset`, `split`, `ingested_at`).
- **RUL Ground Truth:** Utilized Spark RDD `zipWithIndex()` to map 1-indexed engine IDs to line numbers.
- **Storage:** Persisted as compressed Parquet files (`train`: 20,631 rows, `test`: 13,096 rows, `rul_truth`: 100 rows).

### Part 2 — Exploration & Silver Layer
- **Fleet Lifetime Analysis:** Explored engine lifespans with SparkSQL subqueries (fleet of 100 engines ranging from 128 to 362 cycles, average life ~206 cycles).
- **Sensor Pruning:** Computed standard deviation across all settings and sensors in a single distributed aggregation pass. Dropped 7 flat/constant columns (`s1`, `s5`, `s10`, `s16`, `s18`, `s19`, `setting_3`) with $\sigma < 0.01$, preserving 14 informative sensors.
- **RUL Labelling:** Constructed piece-wise linear degradation targets capped at 125 cycles using PySpark window functions (`Window.partitionBy("unit_id")`).

### Part 3 — Feature Engineering & Gold Layer
- **Rolling Temporal Features:** Extracted temporal trends using a rows-based sliding window over preceding 10 cycles (`rowsBetween(-9, 0)`):
  - Rolling mean: `F.avg(s).over(w)`
  - Rolling standard deviation: `F.coalesce(F.stddev(s).over(w), 0.0)`
  - Final feature vector: 43 columns (cycle + 14 raw sensors + 14 rolling means + 14 rolling stds).
- **Physical Plan Optimization:** Benchmarked physical execution plans (`SortMergeJoin` vs. `BroadcastHashJoin`). Broadcasting the 100-row `rul_truth` eliminated 2 network shuffle stages (`Exchange hashpartitioning`).
- **Official Evaluation Points:** Extracted the final flight cycle for each test engine (`gold_test_last`: exactly 100 rows).

### Part 4 — Machine Learning with Spark MLlib
Built an MLlib `Pipeline` with `VectorAssembler` $\to$ `StandardScaler(withMean=True, withStd=True)` $\to$ `Regressor`.

Evaluated models using an 80/20 engine-level split (preventing data leakage across consecutive engine cycles):

| Model | Val RMSE | Val MAE | Fit Time (s) |
|---|:---:|:---:|:---:|
| **Random Forest** (Best) | **17.17** | **11.54** | 21.82 |
| **Gradient-Boosted Trees (GBT)** | 18.47 | 12.70 | 26.38 |
| **Linear Regression** | 19.87 | 16.11 | 3.35 |

- **Final Evaluation (Test Set - 100 engines):**
  - **RMSE:** 18.17 cycles
  - **MAE:** 13.36 cycles
  - **NASA Asymmetric Scoring Function:** 607.95 (penalizing late predictions more heavily than early predictions).
- **Top Predictive Features:** `cycle`, `s11_mean_10`, `s9_mean_10`, `s4_mean_10`.

### Part 5 — Structured Streaming & Real-Time Scoring
- Simulated an incoming streaming workload with JSON micro-batches representing aircraft landing and streaming sensor telemetry.
- Implemented `foreachBatch` pattern:
  - Incremental history accumulation.
  - On-the-fly rolling feature generation across batch boundaries.
  - Scored latest flight cycles using the exported `PipelineModel`.
  - Triggered automatic maintenance alerts when predicted RUL fell below 30 cycles (`ALERT_THRESHOLD = 30`).

### Part 6 — Deep Learning Preparation (Lab 6 Hand-Off)
- Min-Max normalized sensor columns to the $[0, 1]$ interval based strictly on training set statistics.
- Exported:
  - `dl_ready/train/` & `dl_ready/test/` (Parquet)
  - `normalization.json` (Scaling parameters)
  - `baseline_metrics.json` (Benchmark results)

---

## Tech Stack & Prerequisites

- **Compute & Storage Engine:** Apache Spark 3.4+ / 4.x (PySpark)
- **Programming Language:** Python 3.10+
- **Data & ML Libraries:** PySpark SQL & MLlib, NumPy, Pandas, PyArrow, Matplotlib
- **Data Format:** Apache Parquet, JSON lines

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/aminebouaouda02/Lab-5-Practical-assignment---End-to-End-Predictive-Maintenance.git
   cd Lab-5-Practical-assignment---End-to-End-Predictive-Maintenance
   ```

2. Install dependencies:
   ```bash
   pip install pyspark pandas pyarrow numpy matplotlib
   ```

3. Ensure Java 17+ is installed and configured in your environment (`JAVA_HOME`).

4. Open and run the notebook:
   ```bash
   jupyter notebook Assignment1_PredictiveMaintenance.ipynb
   ```
