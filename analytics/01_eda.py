"""
Zepto Capstone - Analytics Module
Part A: profiling, cleaning, EDA, and data story.

Run from the repository root:
    python analytics/01_eda.py

The only call to sns.load_dataset("titanic") in this module is here.
The modeling script reads the resulting analytics/titanic.csv.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
PLOTS = BASE / "plots"
OUTPUTS = BASE / "outputs"
PLOTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


def savefig(name):
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=160, bbox_inches="tight")
    plt.close()


def iqr_outliers(series):
    s = series.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((s < lower) | (s > upper)).sum())
    return count, q1, q3, lower, upper


def pct(x):
    return f"{x:.2f}%"


def main():
    # ------------------------------------------------------------
    # 1. Load raw dataset ONCE and immediately save the offline copy.
    # ------------------------------------------------------------
    print("Loading Titanic from Seaborn (the only network/cache load)...")
    df = sns.load_dataset("titanic")
    raw_df = df.copy()
    raw_path = BASE / "titanic_raw_download.csv"
    df.to_csv(BASE / "titanic.csv", index=False)
    print(f"Raw dataset saved immediately to: {BASE / 'titanic.csv'}")
    print(f"Shape: {df.shape}")

    # Profile
    info_lines = []
    import io
    info_buf = io.StringIO()
    df.info(buf=info_buf)
    info_lines.append("DATAFRAME INFO")
    info_lines.append(info_buf.getvalue())
    info_lines.append("\nDESCRIBE")
    info_lines.append(df.describe(include="all").to_string())
    info_lines.append(f"\nSHAPE: {df.shape}")

    missing = df.isna().mean().mul(100)
    missing = missing[missing > 0].sort_values(ascending=False)
    info_lines.append("\nMISSING VALUES (%)")
    info_lines.append(missing.to_string())

    # ------------------------------------------------------------
    # 2. Cleaning using the required threshold rule.
    # ------------------------------------------------------------
    decisions = []
    for col, rate in missing.items():
        if rate < 5:
            decisions.append(f"{col}: {rate:.2f}% missing -> DROP affected rows (<5%).")
        elif rate <= 30:
            decisions.append(f"{col}: {rate:.2f}% missing -> IMPUTE (5%-30%).")
        else:
            decisions.append(
                f"{col}: {rate:.2f}% missing -> DROP COLUMN (>30%); "
                "imputation would be unreliable."
            )

    # Apply decisions:
    # age (~20%) -> median imputation.
    # embarked / embark_town (<5%) -> drop affected rows.
    # deck (>30%) -> drop column.
    clean = df.copy()
    if "deck" in clean.columns:
        clean = clean.drop(columns=["deck"])
    for col in ["embarked", "embark_town"]:
        if col in clean.columns:
            clean = clean.dropna(subset=[col])
    if "age" in clean.columns:
        clean["age"] = clean["age"].fillna(clean["age"].median())

    # Keep the cleaned dataset as the single committed offline fallback.
    clean.to_csv(BASE / "titanic.csv", index=False)

    info_lines.append("\nCLEANING DECISIONS")
    info_lines.extend(decisions)
    info_lines.append(
        "\nJustification: columns below 5% missingness lose only a small number "
        "of rows, columns between 5% and 30% are imputed, and columns above 30% "
        "are dropped rather than relying on highly uncertain imputation."
    )
    info_lines.append(f"\nCLEANED SHAPE: {clean.shape}")

    # ------------------------------------------------------------
    # 3. Univariate analysis
    # ------------------------------------------------------------
    age_out = iqr_outliers(clean["age"])
    fare_out = iqr_outliers(clean["fare"])

    fare_mean = clean["fare"].mean()
    fare_median = clean["fare"].median()
    fare_mode = clean["fare"].mode().iloc[0]

    if fare_mean > fare_median > fare_mode:
        skew_text = "right-skewed"
    elif fare_mean < fare_median < fare_mode:
        skew_text = "left-skewed"
    else:
        skew_text = "not perfectly ordered by mean/median/mode; inspect the histogram"

    sns.histplot(clean["age"], kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    savefig("age_histogram.png")

    sns.boxplot(x=clean["age"])
    plt.title("Age Box Plot")
    plt.xlabel("Age")
    savefig("age_boxplot.png")

    sns.histplot(clean["fare"], kde=True)
    plt.title("Fare Distribution")
    plt.xlabel("Fare")
    plt.ylabel("Count")
    savefig("fare_histogram.png")

    sns.boxplot(x=clean["fare"])
    plt.title("Fare Box Plot")
    plt.xlabel("Fare")
    savefig("fare_boxplot.png")

    info_lines += [
        "\nUNIVARIATE ANALYSIS",
        f"Age IQR outliers: {age_out[0]}",
        f"Age IQR bounds: [{age_out[3]:.3f}, {age_out[4]:.3f}]",
        f"Fare IQR outliers: {fare_out[0]}",
        f"Fare IQR bounds: [{fare_out[3]:.3f}, {fare_out[4]:.3f}]",
        f"Fare mean: {fare_mean:.4f}",
        f"Fare median: {fare_median:.4f}",
        f"Fare mode: {fare_mode:.4f}",
        f"Fare distribution conclusion: {skew_text}.",
    ]

    # ------------------------------------------------------------
    # 4. Bivariate analysis and exact 6-column correlation matrix.
    # ------------------------------------------------------------
    sex_survival = clean.groupby("sex", observed=True)["survived"].mean()
    class_survival = clean.groupby("pclass")["survived"].mean()
    sex_class_survival = (
        clean.groupby(["sex", "pclass"], observed=True)["survived"].mean()
    )

    # Boolean masking explicitly used for a useful bivariate check.
    female_first = clean[(clean["sex"] == "female") & (clean["pclass"] == 1)]
    male_third = clean[(clean["sex"] == "male") & (clean["pclass"] == 3)]
    female_first_rate = female_first["survived"].mean()
    male_third_rate = male_third["survived"].mean()

    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = clean[corr_cols].corr()

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Titanic Correlation Matrix")
    savefig("correlation_heatmap.png")

    pairs = []
    for i in range(len(corr_cols)):
        for j in range(i + 1, len(corr_cols)):
            pairs.append(
                (corr_cols[i], corr_cols[j], corr.iloc[i, j], abs(corr.iloc[i, j]))
            )
    pairs.sort(key=lambda x: x[3], reverse=True)
    strongest = pairs[:2]

    info_lines += [
        "\nBIVARIATE SURVIVAL RATES",
        "By sex:",
        sex_survival.to_string(),
        "\nBy pclass:",
        class_survival.to_string(),
        "\nBy sex and pclass:",
        sex_class_survival.to_string(),
        f"\nBoolean-mask example: female & first class survival = {female_first_rate:.4f}",
        f"Boolean-mask example: male & third class survival = {male_third_rate:.4f}",
        "\nEXACT 6-COLUMN CORRELATION MATRIX",
        corr.to_string(),
        "\nTWO STRONGEST ABSOLUTE OFF-DIAGONAL CORRELATIONS:",
    ]
    for a, b, value, absolute in strongest:
        info_lines.append(f"{a} vs {b}: r={value:.4f}, |r|={absolute:.4f}")

    # ------------------------------------------------------------
    # 5. Multivariate data story - at least 4 distinct charts.
    # ------------------------------------------------------------
    sns.barplot(x=sex_survival.index, y=sex_survival.values)
    plt.ylim(0, 1)
    plt.title("Survival Rate by Sex")
    plt.ylabel("Survival Rate")
    savefig("survival_by_sex.png")

    sns.barplot(x=class_survival.index.astype(str), y=class_survival.values)
    plt.ylim(0, 1)
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    savefig("survival_by_class.png")

    pivot = sex_class_survival.unstack()
    pivot.plot(kind="bar")
    plt.ylim(0, 1)
    plt.title("Survival Rate by Sex and Passenger Class")
    plt.xlabel("Sex")
    plt.ylabel("Survival Rate")
    plt.legend(title="Pclass")
    savefig("survival_sex_class.png")

    sns.scatterplot(data=clean, x="age", y="fare", hue="survived", alpha=0.65)
    plt.title("Age vs Fare by Survival")
    savefig("age_fare_survival.png")

    # An additional multivariate chart that connects class, fare and survival.
    sns.boxplot(data=clean, x="pclass", y="fare", hue="survived")
    plt.title("Fare by Passenger Class and Survival")
    savefig("fare_class_survival.png")

    # Interpretations are deliberately written to the results file and README.
    interpretations = f"""
MULTIVARIATE DATA STORY - WRITTEN INTERPRETATIONS

1. Survival by sex:
The survival-rate chart compares the outcome across male and female passengers.
The observed rates show a clear gender difference in survival, which is consistent
with sex being an important predictor in the later classification task.

2. Survival by passenger class:
The class chart shows that survival differs materially across passenger classes.
This indicates that socioeconomic/location-related class information contains useful
signal about survival and should be retained as a modeling feature.

3. Survival by sex and passenger class:
The combined chart demonstrates that sex alone does not tell the whole story because
survival also changes across passenger class within each sex. The interaction is
especially useful for understanding why a multivariate model can outperform a
single-variable explanation.

4. Age versus fare by survival:
The scatter plot combines two continuous variables with survival status.
It helps show that passenger characteristics occupy different regions of the
feature space and that survival is not explained by age or fare in isolation.

5. Fare by class and survival:
This chart connects fare, passenger class, and survival in one view. Fare is strongly
related to class structure, while the survival split within classes shows why the
model benefits from considering multiple features together rather than interpreting
fare as an isolated causal factor.
"""
    info_lines.append(interpretations.strip())

    # ------------------------------------------------------------
    # 6. EDA-stage standardization sanity check.
    # ------------------------------------------------------------
    scaler = StandardScaler()
    standardized = scaler.fit_transform(clean[["age", "fare"]])
    zdf = pd.DataFrame(standardized, columns=["age_z", "fare_z"])

    info_lines += [
        "\nEDA-STAGE STANDARDIZATION CHECK",
        "Before standardization:",
        clean[["age", "fare"]].agg(["mean", "std"]).to_string(),
        "\nAfter standardization:",
        zdf.agg(["mean", "std"]).to_string(),
        "\nThe standardized columns should have means approximately 0 and standard "
        "deviations approximately 1. This EDA-only transformation is not used by "
        "the modeling pipeline.",
    ]

    (OUTPUTS / "eda_results.txt").write_text("\n".join(info_lines), encoding="utf-8")

    # A compact README-ready markdown with actual run-specific results.
    strongest_md = "\n".join(
        f"- **{a} vs {b}**: r = {value:.4f} (absolute r = {absolute:.4f})"
        for a, b, value, absolute in strongest
    )
    report_md = f"""# Run-specific EDA findings

Generated by `01_eda.py`.

- Raw shape: `{raw_df.shape}`
- Cleaned shape: `{clean.shape}`
- Age IQR outliers: **{age_out[0]}**
- Fare IQR outliers: **{fare_out[0]}**
- Fare mean / median / mode: **{fare_mean:.4f} / {fare_median:.4f} / {fare_mode:.4f}**
- Fare conclusion: **{skew_text}**
- Female survival rate: **{sex_survival.get('female', np.nan):.4f}**
- Male survival rate: **{sex_survival.get('male', np.nan):.4f}**

## Strongest correlations

{strongest_md}

## Chart interpretations

{interpretations.strip()}
"""
    (OUTPUTS / "eda_findings.md").write_text(report_md, encoding="utf-8")

    print(f"Cleaned Titanic saved to: {BASE / 'titanic.csv'}")
    print(f"EDA results saved to: {OUTPUTS / 'eda_results.txt'}")
    print("EDA completed successfully.")


if __name__ == "__main__":
    main()
