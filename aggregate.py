import pandas as pd, glob

# results = seed 42 (first run), plus seed1, seed2, seed3
folders = ["results", "results_seed1", "results_seed2", "results_seed3"]
dfs = []
for f in folders:
    try:
        d = pd.read_csv(f"{f}/metrics.csv"); d["run"] = f; dfs.append(d)
    except FileNotFoundError:
        print("missing:", f)

all_runs = pd.concat(dfs)
print(all_runs[["run", "method", "accuracy", "f1", "n_features"]].to_string(index=False))

summary = all_runs.groupby("method")[["accuracy", "f1", "n_features"]].agg(["mean", "std"]).round(3)
print("\n=== MEAN and STD over runs ===")
print(summary.to_string())
summary.to_csv("summary_over_seeds.csv")