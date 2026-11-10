from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr, t

# =========================
# 1. Load and match responses
# =========================
base_dir = Path(__file__).resolve().parent
human_cols = [f"NEO{group}_{item}" for group in range(1, 7) for item in range(1, 11)]
llm_cols = [str(i) for i in range(1, 61)]
llm_rename = dict(zip(llm_cols, human_cols))

human = pd.read_csv(base_dir / "humanresponse.csv", dtype={"CVDID": str})
llm = pd.read_csv(base_dir / "llm_answers.csv", dtype={"id": str})

human = human[["CVDID"] + human_cols].rename(columns={"CVDID": "id"})
llm = llm[["id"] + llm_cols].rename(columns=llm_rename)
llm[human_cols] = llm[human_cols].apply(pd.to_numeric, errors="coerce")

merged = pd.merge(human, llm, on="id", how="inner", suffixes=("_human", "_llm"))
if merged.empty:
    raise ValueError("No matching IDs between humanresponse.csv and llm_answers.csv")

# =========================
# 2. Helper functions
# =========================
def fisher_z_pool(r_values):
    r = np.asarray(r_values, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan
    return np.tanh(np.arctanh(np.clip(r, -0.999999, 0.999999)).mean())


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return spearmanr(x, y).statistic


# =========================
# 3. Accuracy
# =========================
RANGE = 4  # Response scale: 1 to 5
accuracy_values = [
    1 - (merged[f"{col}_human"] - merged[f"{col}_llm"]).abs() / RANGE
    for col in human_cols
]
overall_accuracy = pd.concat(accuracy_values, ignore_index=True).mean()

# =========================
# 4. Item-level correlation
# =========================
item_correlations = []

for col in human_cols:
    x = merged[f"{col}_human"]
    y = merged[f"{col}_llm"]
    item_correlations.append(safe_spearman(x, y))

overall_item_corr = fisher_z_pool(item_correlations)

# =========================
# 5. Profile correlation
# =========================
profile_correlations = []

for _, row in merged.iterrows():
    human_profile = np.array([row[f"{col}_human"] for col in human_cols], dtype=float)
    llm_profile = np.array([row[f"{col}_llm"] for col in human_cols], dtype=float)
    valid = np.isfinite(human_profile) & np.isfinite(llm_profile)
    if valid.sum() >= 3:
        profile_correlations.append(safe_spearman(human_profile[valid], llm_profile[valid]))

overall_profile_corr = fisher_z_pool(profile_correlations)

# =========================
# 6. Print results
# =========================
summary = pd.DataFrame({
    "metric": [
        "overall_accuracy",
        "overall_item_level_correlation",
        "overall_profile_correlation"
    ],
    "value": [overall_accuracy, overall_item_corr, overall_profile_corr]
})

print("Summary")
print(summary.to_string(index=False))

# =========================
# 7. Error slopes and 95% confidence intervals
# =========================
dimension_items = {
    "Overall": range(1, 61),
    "Openness": [13, 23, 43, 48, 53, 58, 3, 8, 18, 38, 28, 33],
    "Conscientiousness": [5, 10, 15, 30, 55, 25, 35, 60, 20, 40, 45, 50],
    "Extraversion": [7, 12, 37, 42, 2, 17, 27, 57, 22, 32, 47, 52],
    "Agreeableness": [9, 14, 19, 24, 29, 44, 54, 59, 4, 34, 39, 49],
    "Neuroticism": [1, 11, 16, 31, 46, 6, 21, 26, 36, 41, 51, 56]
}

slope_rows = []
for dimension, items in dimension_items.items():
    cols = [human_cols[i - 1] for i in items]
    human_mean = merged[[f"{col}_human" for col in cols]].mean(axis=1)
    llm_mean = merged[[f"{col}_llm" for col in cols]].mean(axis=1)
    valid = human_mean.notna() & llm_mean.notna()
    x = human_mean[valid]
    error = llm_mean[valid] - x
    n = len(x)

    if n < 3 or x.nunique() < 2:
        slope, ci_low, ci_high = np.nan, np.nan, np.nan
    else:
        fit = linregress(x, error)
        margin = t.ppf(0.975, n - 2) * fit.stderr
        slope, ci_low, ci_high = fit.slope, fit.slope - margin, fit.slope + margin

    slope_rows.append({
        "dimension": dimension,
        "slope": slope,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n
    })

print("\nError slope (LLM - human) and 95% CI")
print(pd.DataFrame(slope_rows).to_string(index=False))
