# Student Performance Prediction using a Genetic Algorithm

**Goal:** predict whether a student passes (final grade G3 >= 10) from the UCI Student Performance dataset.
**Model:** Random Forest (default) or SVM. **Bio-inspired algorithm:** Genetic Algorithm that optimises
**feature selection + hyper-parameters** at the same time.

## Run
```
pip install -r requirements.txt
python ga_student_performance.py                    # downloads dataset automatically
python ga_student_performance.py --no-prev-grades   # recommended 2nd experiment (drops G1, G2)
python ga_student_performance.py --model svm
python ga_student_performance.py --dataset por --pop 40 --gens 40
```
If the download is blocked, get the zip from
https://archive.ics.uci.edu/dataset/320/student+performance and place `student-mat.csv`
(and/or `student-por.csv`) in a `data/` folder next to the script.

## Outputs (`results/`)
| File | Content |
|---|---|
| `metrics.csv` | Accuracy, F1, #features for baseline vs GA variants |
| `ga_history.csv` | Best/mean fitness, CV-F1, #features per generation |
| `fitness_over_generations.png` | GA convergence curve |
| `features_over_generations.png` | Feature count of best individual |
| `baseline_comparison.png` | Accuracy / F1 / #features bars |
| `selected_feature_importance.png` | Importance of selected features (RF) |
| `selected_features.txt` | Chosen features + hyper-parameters |

## How the GA works
* **Chromosome:** one bit per (one-hot encoded) feature + one integer gene per hyper-parameter
  (RF: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features;
  SVM: C, gamma, kernel).
* **Fitness:** `5-fold CV F1 (train set only) - alpha * (#selected / #total)`, so smaller feature sets are rewarded.
* **Operators:** tournament selection (k=3), uniform crossover (p=0.8), bit-flip + gene-resample mutation, elitism (2).
* **No leakage:** the test split (25%) is only used once at the end.
* **Comparison:** (1) baseline: all features + default model, (2) GA features only, (3) GA features + tuned hyper-parameters.

## Notes for your report
* G1 and G2 (previous period grades) predict G3 very strongly, so accuracy is already ~90%+ and the GA gains are small.
  Run with and without `--no-prev-grades` and discuss both.
* The dataset is small (395 / 649 rows), so test metrics vary by seed; run a few `--seed` values and report mean +/- std.
* The pass class is imbalanced (~67% pass in math), which is why F1 is reported next to accuracy.
* Tune `--alpha` to trade accuracy against number of features.
