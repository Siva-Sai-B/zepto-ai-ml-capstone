"""
Zepto Capstone - Analytics Module
Part B: predictive modeling, imbalance comparison, tuning, regression.

Run after 01_eda.py:
    python analytics/02_modeling.py

This script reads analytics/titanic.csv and never calls sns.load_dataset().
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
PLOTS = BASE / "plots"
OUTPUTS = BASE / "outputs"
MODELS = BASE / "models"
PLOTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

RANDOM_STATE = 42


def make_preprocessor(numeric_features, categorical_features):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])


def evaluate_classifier(name, pipeline, X_test, y_test):
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "AUC": roc_auc_score(y_test, proba),
    }
    return metrics, pred, proba


def adjusted_r2(r2, n, p):
    if n <= p + 1:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def main():
    csv_path = BASE / "titanic.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            "analytics/titanic.csv does not exist. Run `python analytics/01_eda.py` first."
        )

    df = pd.read_csv(csv_path)
    print(f"Loaded cleaned Titanic CSV: {df.shape}")

    # ------------------------------------------------------------
    # 7. Stratified train/test split FIRST, before preprocessing.
    # ------------------------------------------------------------
    target = "survived"
    feature_cols = [
        "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"
    ]
    X = df[feature_cols].copy()
    y = df[target].astype(int).copy()

    class_balance = y.value_counts(normalize=True).sort_index()
    print("\nClass balance:")
    print(class_balance)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    split_report = [
        "STRATIFIED TRAIN/TEST SPLIT",
        f"Train rows: {len(X_train)}",
        f"Test rows: {len(X_test)}",
        f"Overall positive class rate: {y.mean():.4f}",
        f"Train positive class rate: {y_train.mean():.4f}",
        f"Test positive class rate: {y_test.mean():.4f}",
        (
            "Stratification preserves approximately the same survived/not-survived "
            "class proportions in both subsets, reducing the risk that a random "
            "split produces an unrepresentative test set."
        ),
    ]

    numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical_features = ["sex", "embarked"]
    preprocessor = make_preprocessor(numeric_features, categorical_features)

    # ------------------------------------------------------------
    # 8-10. Three classifiers using the same split and train-only preprocessing.
    # ------------------------------------------------------------
    model_specs = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    fitted = {}
    metric_rows = []
    predictions = {}
    probabilities = {}

    for name, estimator in model_specs.items():
        pipe = Pipeline([
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        metrics, pred, proba = evaluate_classifier(name, pipe, X_test, y_test)
        fitted[name] = pipe
        metric_rows.append(metrics)
        predictions[name] = pred
        probabilities[name] = proba

    classification_df = pd.DataFrame(metric_rows)

    # Confusion matrices in one figure.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, name in zip(axes, fitted):
        cm = confusion_matrix(y_test, predictions[name])
        ax.imshow(cm, interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(PLOTS / "classification_confusion_matrices.png", dpi=160)
    plt.close()

    # ROC curves.
    plt.figure(figsize=(8, 6))
    for name in fitted:
        fpr, tpr, _ = roc_curve(y_test, probabilities[name])
        auc = classification_df.loc[classification_df["Model"] == name, "AUC"].iloc[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "roc_curves.png", dpi=160)
    plt.close()

    # Decision tree visualization.
    tree_pipe = fitted["Decision Tree"]
    tree_model = tree_pipe.named_steps["model"]
    tree_pre = tree_pipe.named_steps["preprocessor"]
    feature_names = tree_pre.get_feature_names_out()
    plt.figure(figsize=(24, 12))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=["Not survived", "Survived"],
        filled=True,
        rounded=True,
        max_depth=4,
        fontsize=7,
    )
    plt.title("Decision Tree")
    plt.tight_layout()
    plt.savefig(PLOTS / "decision_tree.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # 11. Imbalance handling comparison.
    # ------------------------------------------------------------
    imbalance_specs = {
        "Baseline / no handling": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "class_weight='balanced'": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    imbalance_rows = []
    imbalance_pipes = {}

    for label, estimator in imbalance_specs.items():
        pipe = Pipeline([
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        metrics, _, _ = evaluate_classifier(label, pipe, X_test, y_test)
        imbalance_rows.append({
            "Variant": label,
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
        })
        imbalance_pipes[label] = pipe

    # SMOTE must be inside the training-only imblearn pipeline.
    smote_pipe = ImbPipeline([
        ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    smote_pipe.fit(X_train, y_train)
    smote_metrics, _, _ = evaluate_classifier(
        "SMOTE", smote_pipe, X_test, y_test
    )
    imbalance_rows.append({
        "Variant": "SMOTE",
        "Precision": smote_metrics["Precision"],
        "Recall": smote_metrics["Recall"],
        "F1": smote_metrics["F1"],
    })

    imbalance_df = pd.DataFrame(imbalance_rows)
    imbalance_df.to_csv(OUTPUTS / "imbalance_comparison.csv", index=False)

    best_imb = imbalance_df.sort_values("F1", ascending=False).iloc[0]
    imbalance_conclusion = (
        f"The best imbalance variant by F1 was {best_imb['Variant']} "
        f"(F1={best_imb['F1']:.4f}). The comparison shows the trade-off between "
        f"precision and recall: class weighting or SMOTE can improve recognition "
        f"of the minority class, but the preferred strategy should be selected "
        f"using the metric that best reflects the deployment objective."
    )

    # ------------------------------------------------------------
    # 12. GridSearchCV over Random Forest, with OOB enabled.
    # ------------------------------------------------------------
    rf_pipe = Pipeline([
        ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
        ("model", RandomForestClassifier(
            oob_score=True,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )),
    ])

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10],
        "model__max_features": ["sqrt", "log2"],
    }

    grid = GridSearchCV(
        rf_pipe,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    tuned_rf = grid.best_estimator_
    tuned_metrics, tuned_pred, tuned_proba = evaluate_classifier(
        "Tuned Random Forest", tuned_rf, X_test, y_test
    )
    oob_score = tuned_rf.named_steps["model"].oob_score_

    # Add tuned RF as a candidate if it is better than the untuned RF.
    tuned_row = {
        "Model": "Tuned Random Forest",
        **{k: tuned_metrics[k] for k in ["Accuracy", "Precision", "Recall", "F1", "AUC"]}
    }
    classification_with_tuned = pd.concat(
        [classification_df, pd.DataFrame([tuned_row])], ignore_index=True
    )

    # ------------------------------------------------------------
    # 13. Regression: predict fare from other available features.
    # ------------------------------------------------------------
    # Avoid target-derived "alive" and duplicate representations that do not add
    # independent information. Use passenger characteristics available before fare.
    regression_features = [
        "pclass", "sex", "age", "sibsp", "parch", "embarked",
        "who", "adult_male", "alone"
    ]
    Xr = df[regression_features].copy()
    yr = df["fare"].copy()

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.20, random_state=RANDOM_STATE
    )

    reg_numeric = ["pclass", "age", "sibsp", "parch", "adult_male", "alone"]
    reg_categorical = ["sex", "embarked", "who"]

    reg_pipe = Pipeline([
        ("preprocessor", make_preprocessor(reg_numeric, reg_categorical)),
        ("model", LinearRegression()),
    ])
    reg_pipe.fit(Xr_train, yr_train)
    yr_pred = reg_pipe.predict(Xr_test)

    mae = mean_absolute_error(yr_test, yr_pred)
    rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
    r2 = r2_score(yr_test, yr_pred)
    reg_pre = reg_pipe.named_steps["preprocessor"]
    p = len(reg_pre.get_feature_names_out())
    adj_r2 = adjusted_r2(r2, len(yr_test), p)

    residuals = yr_test.to_numpy() - yr_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(yr_pred, residuals, alpha=0.65)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Fare")
    plt.ylabel("Residual")
    plt.title("Fare Regression Residual Plot")
    plt.tight_layout()
    plt.savefig(PLOTS / "regression_residuals.png", dpi=160)
    plt.close()

    # Simple, transparent heteroscedasticity check: compare residual spread
    # between lower and upper fitted-value halves.
    midpoint = np.median(yr_pred)
    low_resid = np.abs(residuals[yr_pred <= midpoint])
    high_resid = np.abs(residuals[yr_pred > midpoint])
    low_mean = low_resid.mean() if len(low_resid) else np.nan
    high_mean = high_resid.mean() if len(high_resid) else np.nan
    ratio = high_mean / low_mean if low_mean else np.inf
    hetero = "suggests heteroscedasticity" if ratio > 1.5 or ratio < (1 / 1.5) else "does not strongly suggest heteroscedasticity"

    # ------------------------------------------------------------
    # 14. Final model comparison table + recommendation.
    # ------------------------------------------------------------
    comparison = classification_with_tuned.copy()
    for col in ["MAE", "RMSE", "R2", "Adjusted_R2"]:
        comparison[col] = np.nan

    regression_row = {
        "Model": "Linear Regression (fare)",
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "AUC": np.nan,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adj_r2,
    }
    comparison = pd.concat([comparison, pd.DataFrame([regression_row])], ignore_index=True)
    comparison.to_csv(OUTPUTS / "model_comparison.csv", index=False)

    classifier_candidates = classification_with_tuned[
        classification_with_tuned["Model"] != "Tuned Random Forest"
    ].copy()
    # Include tuned RF in deployment decision.
    best_classifier_row = classification_with_tuned.sort_values(
        ["F1", "AUC"], ascending=False
    ).iloc[0]
    best_classifier_name = best_classifier_row["Model"]

    # Map tuned name to fitted pipeline; otherwise use original.
    pipeline_lookup = {**fitted, "Tuned Random Forest": tuned_rf}
    best_pipeline = pipeline_lookup[best_classifier_name]
    best_path = MODELS / "best_model_pipeline.joblib"
    joblib.dump(best_pipeline, best_path)

    # Reload and confirm prediction on raw/unprocessed test rows.
    reloaded = joblib.load(best_path)
    reload_pred = reloaded.predict(X_test.head(5))
    reload_check = bool(len(reload_pred) == min(5, len(X_test)))

    recommendation = (
        f"Recommendation: deploy {best_classifier_name}. It achieved the highest "
        f"F1 score among the evaluated classifiers at {best_classifier_row['F1']:.4f}, "
        f"with precision {best_classifier_row['Precision']:.4f}, recall "
        f"{best_classifier_row['Recall']:.4f}, accuracy {best_classifier_row['Accuracy']:.4f}, "
        f"and AUC {best_classifier_row['AUC']:.4f}. F1 is useful here because it balances "
        f"precision and recall rather than optimizing accuracy alone. The final artifact "
        f"contains preprocessing and the estimator together, so raw feature rows can be "
        f"passed directly to the saved pipeline."
    )

    # ------------------------------------------------------------
    # Write detailed results.
    # ------------------------------------------------------------
    lines = []
    lines.extend(split_report)
    lines += [
        "\nCLASSIFIER COMPARISON",
        classification_with_tuned.round(4).to_string(index=False),
        "\nIMBALANCE COMPARISON",
        imbalance_df.round(4).to_string(index=False),
        "\nIMBALANCE CONCLUSION",
        imbalance_conclusion,
        "\nRANDOM FOREST GRID SEARCH",
        f"Best parameters: {grid.best_params_}",
        f"Best CV F1: {grid.best_score_:.4f}",
        f"OOB score: {oob_score:.4f}",
        "\nREGRESSION RESULTS",
        f"MAE: {mae:.4f}",
        f"RMSE: {rmse:.4f}",
        f"R2: {r2:.4f}",
        f"Adjusted R2: {adj_r2:.4f}",
        f"Residual spread lower-half mean absolute residual: {low_mean:.4f}",
        f"Residual spread upper-half mean absolute residual: {high_mean:.4f}",
        f"Heteroscedasticity conclusion: residual plot {hetero}.",
        "\nFINAL RECOMMENDATION",
        recommendation,
        f"\nSaved complete fitted pipeline: {best_path}",
        f"Reload-and-predict check passed: {reload_check}",
        "\nCLASSIFICATION METRICS AND REGRESSION METRICS ARE SEPARATE GROUPS.",
    ]
    (OUTPUTS / "model_results.txt").write_text("\n".join(lines), encoding="utf-8")

    # Markdown results useful for README/reporting.
    md = f"""# Run-specific modeling findings

## Classification

{classification_with_tuned.round(4).to_markdown(index=False)}

## Imbalance comparison

{imbalance_df.round(4).to_markdown(index=False)}

**Conclusion:** {imbalance_conclusion}

## Random Forest tuning

- Best parameters: `{grid.best_params_}`
- Best cross-validation F1: **{grid.best_score_:.4f}**
- OOB score: **{oob_score:.4f}**

## Regression

| Metric | Value |
|---|---:|
| MAE | {mae:.4f} |
| RMSE | {rmse:.4f} |
| R² | {r2:.4f} |
| Adjusted R² | {adj_r2:.4f} |

The residual-spread check {hetero}. The lower-half mean absolute residual was
{low_mean:.4f}, compared with {high_mean:.4f} for the upper half.

## Final recommendation

{recommendation}

## Saved artifact

The complete fitted preprocessing + classifier pipeline was saved to
`models/best_model_pipeline.joblib` and successfully reloaded for prediction on
raw, unpreprocessed test rows.
"""
    (OUTPUTS / "model_findings.md").write_text(md, encoding="utf-8")

    print("\nMODEL COMPARISON")
    print(classification_with_tuned.round(4).to_string(index=False))
    print("\nBest classifier:", best_classifier_name)
    print("Pipeline saved:", best_path)
    print("Reload check:", reload_check)
    print("Modeling completed successfully.")


if __name__ == "__main__":
    main()
