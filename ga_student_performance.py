"""
Student Performance Prediction using a Genetic Algorithm
=========================================================
Task      : Predict whether a student passes (final grade G3 >= 10).
Dataset   : UCI Student Performance (student-mat.csv / student-por.csv).
Model     : Random Forest  (or SVM with --model svm).
Bio-algo  : Genetic Algorithm that jointly optimises
              (1) feature subset  (binary genes)
              (2) model hyper-parameters (integer / categorical genes)

Usage
-----
    python ga_student_performance.py                      # math dataset, G1/G2 included
    python ga_student_performance.py --no-prev-grades     # harder, more interesting task
    python ga_student_performance.py --dataset por --model svm
    python ga_student_performance.py --pop 40 --gens 30

Outputs (in ./results): metrics.csv, ga_history.csv, selected_features.txt, *.png
"""
import argparse, io, os, random, zipfile, urllib.request, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")
UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def synthetic_students(n=395, seed=0):
    """Fallback ONLY if the real dataset cannot be loaded. Mimics UCI columns."""
    r = np.random.RandomState(seed)
    df = pd.DataFrame({
        "school": r.choice(["GP", "MS"], n, p=[.88, .12]),
        "sex": r.choice(["F", "M"], n),
        "age": r.randint(15, 22, n),
        "address": r.choice(["U", "R"], n, p=[.78, .22]),
        "famsize": r.choice(["GT3", "LE3"], n, p=[.7, .3]),
        "Pstatus": r.choice(["T", "A"], n, p=[.9, .1]),
        "Medu": r.randint(0, 5, n), "Fedu": r.randint(0, 5, n),
        "Mjob": r.choice(["teacher", "health", "services", "at_home", "other"], n),
        "Fjob": r.choice(["teacher", "health", "services", "at_home", "other"], n),
        "reason": r.choice(["home", "reputation", "course", "other"], n),
        "guardian": r.choice(["mother", "father", "other"], n),
        "traveltime": r.randint(1, 5, n), "studytime": r.randint(1, 5, n),
        "failures": r.choice([0, 1, 2, 3], n, p=[.79, .13, .05, .03]),
        "schoolsup": r.choice(["yes", "no"], n, p=[.13, .87]),
        "famsup": r.choice(["yes", "no"], n),
        "paid": r.choice(["yes", "no"], n),
        "activities": r.choice(["yes", "no"], n),
        "nursery": r.choice(["yes", "no"], n, p=[.8, .2]),
        "higher": r.choice(["yes", "no"], n, p=[.95, .05]),
        "internet": r.choice(["yes", "no"], n, p=[.83, .17]),
        "romantic": r.choice(["yes", "no"], n),
        "famrel": r.randint(1, 6, n), "freetime": r.randint(1, 6, n),
        "goout": r.randint(1, 6, n), "Dalc": r.randint(1, 6, n),
        "Walc": r.randint(1, 6, n), "health": r.randint(1, 6, n),
        "absences": r.poisson(5, n),
    })
    ability = (0.9 * df.studytime - 1.6 * df.failures - 0.08 * df.absences
               + 0.3 * df.Medu + r.randn(n) * 1.5)
    g1 = np.clip(np.round(10 + ability + r.randn(n)), 0, 20)
    g2 = np.clip(np.round(g1 + r.randn(n)), 0, 20)
    g3 = np.clip(np.round(g2 + r.randn(n) * 1.2), 0, 20)
    df["G1"], df["G2"], df["G3"] = g1.astype(int), g2.astype(int), g3.astype(int)
    return df


def load_data(dataset="mat"):
    fname = f"student-{dataset}.csv"
    for p in (fname, os.path.join("data", fname)):
        if os.path.exists(p):
            print(f"[data] loaded local file {p}")
            return pd.read_csv(p, sep=";"), True
    try:
        print("[data] downloading from UCI ...")
        raw = urllib.request.urlopen(UCI_URL, timeout=30).read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        os.makedirs("data", exist_ok=True)
        z.extractall("data")
        print("[data] downloaded and saved in ./data")
        return pd.read_csv(os.path.join("data", fname), sep=";"), True
    except Exception as e:
        print(f"[data] !! could not get UCI data ({e}).")
        print("[data] !! Using SYNTHETIC stand-in data. Put student-mat.csv in ./data for real results.")
        return synthetic_students(), False


def preprocess(df, use_prev_grades=True):
    df = df.copy()
    y = (df["G3"] >= 10).astype(int)          # 1 = pass
    X = df.drop(columns=["G3"])
    if not use_prev_grades:
        X = X.drop(columns=["G1", "G2"])
    X = pd.get_dummies(X, drop_first=True).astype(float)
    return X, y


# ----------------------------------------------------------------------------
# 2. MODEL FACTORY  +  CHROMOSOME DEFINITION
# ----------------------------------------------------------------------------
# Hyper-parameter genes: (name, list of allowed values). Gene value = index.
HP_SPACE = {
    "rf": [
        ("n_estimators",      [20, 50, 100, 150, 200, 300]),
        ("max_depth",         [3, 5, 8, 12, 20, None]),
        ("min_samples_split", [2, 3, 5, 8, 12]),
        ("min_samples_leaf",  [1, 2, 3, 5]),
        ("max_features",      ["sqrt", "log2", 0.5, 0.8]),
    ],
    "svm": [
        ("C",      [0.1, 0.5, 1, 2, 5, 10, 50, 100]),
        ("gamma",  ["scale", "auto", 0.001, 0.01, 0.05, 0.1]),
        ("kernel", ["rbf", "linear", "poly"]),
    ],
}


def build_model(model_type, hp, seed):
    if model_type == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **hp)
    return Pipeline([("sc", StandardScaler()), ("svm", SVC(random_state=seed, **hp))])


def default_model(model_type, seed):
    if model_type == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=-1)
    return Pipeline([("sc", StandardScaler()), ("svm", SVC(random_state=seed))])


class GA:
    """Chromosome = [ n_features binary genes | hyper-parameter index genes ]"""

    def __init__(self, X, y, model_type="rf", pop=30, gens=25, cx=0.8,
                 mut_feat=None, mut_hp=0.2, elite=2, tournament=3,
                 alpha=0.02, cv=5, seed=42):
        self.X, self.y = X.values, y.values
        self.cols = list(X.columns)
        self.nf = X.shape[1]
        self.mt = model_type
        self.space = HP_SPACE[model_type]
        self.pop_size, self.gens = pop, gens
        self.cx, self.elite, self.k = cx, elite, tournament
        self.mut_feat = mut_feat if mut_feat is not None else 1.0 / self.nf
        self.mut_hp = mut_hp
        self.alpha = alpha                 # penalty weight on fraction of features used
        self.cv = StratifiedKFold(cv, shuffle=True, random_state=seed)
        self.seed = seed
        self.rng = random.Random(seed)
        self.cache = {}
        self.history = []

    # --- chromosome helpers
    def random_ind(self):
        feats = [1 if self.rng.random() < 0.5 else 0 for _ in range(self.nf)]
        hps = [self.rng.randrange(len(v)) for _, v in self.space]
        return feats + hps

    def decode(self, ind):
        mask = np.array(ind[:self.nf], dtype=bool)
        hp = {n: v[g] for (n, v), g in zip(self.space, ind[self.nf:])}
        return mask, hp

    # --- fitness = CV-F1  -  alpha * (selected / total)
    def fitness(self, ind):
        key = tuple(ind)
        if key in self.cache:
            return self.cache[key]
        mask, hp = self.decode(ind)
        if mask.sum() == 0:
            self.cache[key] = (0.0, 0.0, 0.0)
            return self.cache[key]
        model = build_model(self.mt, hp, self.seed)
        f1 = cross_val_score(model, self.X[:, mask], self.y, cv=self.cv,
                             scoring="f1").mean()
        fit = f1 - self.alpha * mask.sum() / self.nf
        self.cache[key] = (fit, f1, mask.sum())
        return self.cache[key]

    # --- GA operators
    def select(self, pop, fits):
        cand = self.rng.sample(range(len(pop)), self.k)
        return pop[max(cand, key=lambda i: fits[i])][:]

    def crossover(self, a, b):
        if self.rng.random() > self.cx:
            return a[:], b[:]
        c1, c2 = a[:], b[:]
        for i in range(len(a)):                      # uniform crossover
            if self.rng.random() < 0.5:
                c1[i], c2[i] = c2[i], c1[i]
        return c1, c2

    def mutate(self, ind):
        for i in range(self.nf):                     # bit flip
            if self.rng.random() < self.mut_feat:
                ind[i] = 1 - ind[i]
        for j, (_, vals) in enumerate(self.space):   # resample hyper-parameter
            if self.rng.random() < self.mut_hp:
                ind[self.nf + j] = self.rng.randrange(len(vals))
        return ind

    def run(self):
        pop = [self.random_ind() for _ in range(self.pop_size)]
        best_ever, best_fit = None, -1
        for g in range(self.gens + 1):
            scored = [self.fitness(i) for i in pop]
            fits = [s[0] for s in scored]
            bi = int(np.argmax(fits))
            if fits[bi] > best_fit:
                best_fit, best_ever = fits[bi], pop[bi][:]
            bf = self.fitness(best_ever)
            self.history.append(dict(
                generation=g, best_fitness=bf[0], best_cv_f1=bf[1],
                n_features=int(bf[2]), mean_fitness=float(np.mean(fits)),
                gen_best_fitness=fits[bi]))
            print(f"gen {g:02d} | best fit {bf[0]:.4f} | cv-F1 {bf[1]:.4f} | "
                  f"#feat {int(bf[2]):2d} | mean {np.mean(fits):.4f}")
            if g == self.gens:
                break
            order = np.argsort(fits)[::-1]
            nxt = [pop[i][:] for i in order[:self.elite]]          # elitism
            while len(nxt) < self.pop_size:
                c1, c2 = self.crossover(self.select(pop, fits), self.select(pop, fits))
                nxt += [self.mutate(c1), self.mutate(c2)]
            pop = nxt[:self.pop_size]
        return best_ever, pd.DataFrame(self.history)


# ----------------------------------------------------------------------------
# 3. EVALUATION + PLOTS
# ----------------------------------------------------------------------------
def evaluate(name, model, Xtr, Xte, ytr, yte, n_feat):
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    return dict(method=name, accuracy=accuracy_score(yte, p),
                f1=f1_score(yte, p), n_features=n_feat), confusion_matrix(yte, p)


def make_plots(hist, res, feat_names, importances, outdir):
    # fitness over generations
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(hist.generation, hist.best_fitness, "o-", label="Best fitness")
    ax.plot(hist.generation, hist.mean_fitness, "s--", label="Population mean")
    ax.plot(hist.generation, hist.best_cv_f1, "^:", label="Best CV F1")
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness"); ax.grid(alpha=.3)
    ax.set_title("GA convergence"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{outdir}/fitness_over_generations.png", dpi=150); plt.close(fig)

    # features over generations
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist.generation, hist.n_features, "o-", color="tab:green")
    ax.set_xlabel("Generation"); ax.set_ylabel("# selected features (best individual)")
    ax.grid(alpha=.3); ax.set_title("Feature count of best individual"); fig.tight_layout()
    fig.savefig(f"{outdir}/features_over_generations.png", dpi=150); plt.close(fig)

    # baseline comparison
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#999999", "#4c8bf5", "#2ca02c"][:len(res)]
    for ax, col, t in zip(axs, ["accuracy", "f1", "n_features"],
                          ["Test accuracy", "Test F1-score", "# features used"]):
        bars = ax.bar(res.method, res[col], color=colors)
        for b, v in zip(bars, res[col]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}" if col != "n_features" else int(v),
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(t); ax.tick_params(axis="x", labelrotation=15)
        if col != "n_features":
            ax.set_ylim(max(0, res[col].min() - .1), min(1.05, res[col].max() + .06))
    fig.tight_layout(); fig.savefig(f"{outdir}/baseline_comparison.png", dpi=150); plt.close(fig)

    # importance of selected features
    if importances is not None:
        s = pd.Series(importances, index=feat_names).sort_values()
        fig, ax = plt.subplots(figsize=(7, max(3, .3 * len(s))))
        s.plot.barh(ax=ax, color="tab:orange"); ax.set_title("Importance of GA-selected features")
        fig.tight_layout(); fig.savefig(f"{outdir}/selected_feature_importance.png", dpi=150)
        plt.close(fig)


# ----------------------------------------------------------------------------
# 4. MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mat", choices=["mat", "por"])
    ap.add_argument("--model", default="rf", choices=["rf", "svm"])
    ap.add_argument("--no-prev-grades", action="store_true", help="drop G1 and G2")
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.02, help="feature-count penalty")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed)

    df, real = load_data(a.dataset)
    X, y = preprocess(df, not a.no_prev_grades)
    print(f"[data] {X.shape[0]} students, {X.shape[1]} features after encoding, "
          f"pass rate {y.mean():.2%}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=a.seed)

    # ---- GA runs on TRAIN only (CV inside); test set is untouched until the end
    ga = GA(Xtr, ytr, a.model, a.pop, a.gens, alpha=a.alpha, seed=a.seed)
    best, hist = ga.run()
    mask, hp = ga.decode(best)
    sel = [c for c, m in zip(X.columns, mask) if m]
    print("\nBest hyper-parameters:", hp)
    print(f"Selected {len(sel)}/{X.shape[1]} features:", sel)

    # ---- Final comparison on held-out test set
    rows, cms = [], {}
    r, cms["Baseline (all feats, default)"] = evaluate(
        "Baseline (all feats, default)", default_model(a.model, a.seed),
        Xtr, Xte, ytr, yte, X.shape[1]); rows.append(r)
    r, cms["GA features only"] = evaluate(
        "GA features only (default HP)", default_model(a.model, a.seed),
        Xtr[sel], Xte[sel], ytr, yte, len(sel)); rows.append(r)
    final = build_model(a.model, hp, a.seed)
    r, cms["GA features + HP"] = evaluate(
        "GA features + HP", final, Xtr[sel], Xte[sel], ytr, yte, len(sel)); rows.append(r)
    res = pd.DataFrame(rows)

    # CV on train for all three (more stable than one split)
    cv = StratifiedKFold(5, shuffle=True, random_state=999)
    res["cv_f1_train"] = [
        cross_val_score(default_model(a.model, a.seed), Xtr, ytr, cv=cv, scoring="f1").mean(),
        cross_val_score(default_model(a.model, a.seed), Xtr[sel], ytr, cv=cv, scoring="f1").mean(),
        cross_val_score(build_model(a.model, hp, a.seed), Xtr[sel], ytr, cv=cv, scoring="f1").mean()]

    print("\n=== RESULTS (held-out test set) ===")
    print(res.round(4).to_string(index=False))
    print("\nConfusion matrix, GA features + HP (rows=true, cols=pred; 0=fail 1=pass):")
    print(cms["GA features + HP"])

    imp = getattr(final, "feature_importances_", None)
    make_plots(hist, res, sel, imp, a.out)
    res.round(4).to_csv(f"{a.out}/metrics.csv", index=False)
    hist.round(4).to_csv(f"{a.out}/ga_history.csv", index=False)
    with open(f"{a.out}/selected_features.txt", "w") as f:
        f.write(f"data: {'REAL UCI' if real else 'SYNTHETIC'}\nmodel: {a.model}\n")
        f.write(f"hyperparameters: {hp}\nselected ({len(sel)}/{X.shape[1]}):\n")
        f.write("\n".join(sel))
    print(f"\nSaved tables and figures in {a.out}/")


if __name__ == "__main__":
    main()
