# Run-specific modeling findings

## Classification

| Model               |   Accuracy |   Precision |   Recall |     F1 |    AUC |
|:--------------------|-----------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.809  |      0.7833 |   0.6912 | 0.7344 | 0.861  |
| Decision Tree       |     0.764  |      0.76   |   0.5588 | 0.6441 | 0.8374 |
| Random Forest       |     0.8034 |      0.7619 |   0.7059 | 0.7328 | 0.8237 |
| Tuned Random Forest |     0.8034 |      0.7619 |   0.7059 | 0.7328 | 0.8237 |

## Imbalance comparison

| Variant                 |   Precision |   Recall |     F1 |
|:------------------------|------------:|---------:|-------:|
| Baseline / no handling  |      0.7833 |   0.6912 | 0.7344 |
| class_weight='balanced' |      0.7183 |   0.75   | 0.7338 |
| SMOTE                   |      0.7353 |   0.7353 | 0.7353 |

**Conclusion:** The best imbalance variant by F1 was SMOTE (F1=0.7353). The comparison shows the trade-off between precision and recall: class weighting or SMOTE can improve recognition of the minority class, but the preferred strategy should be selected using the metric that best reflects the deployment objective.

## Random Forest tuning

- Best parameters: `{'model__max_depth': None, 'model__max_features': 'sqrt', 'model__n_estimators': 300}`
- Best cross-validation F1: **0.7449**
- OOB score: **0.8073**

## Regression

| Metric | Value |
|---|---:|
| MAE | 21.2550 |
| RMSE | 42.4939 |
| R² | 0.3232 |
| Adjusted R² | 0.2650 |

The residual-spread check suggests heteroscedasticity. The lower-half mean absolute residual was
9.8025, compared with 32.7076 for the upper half.

## Final recommendation

Recommendation: deploy Logistic Regression. It achieved the highest F1 score among the evaluated classifiers at 0.7344, with precision 0.7833, recall 0.6912, accuracy 0.8090, and AUC 0.8610. F1 is useful here because it balances precision and recall rather than optimizing accuracy alone. The final artifact contains preprocessing and the estimator together, so raw feature rows can be passed directly to the saved pipeline.

## Saved artifact

The complete fitted preprocessing + classifier pipeline was saved to
`models/best_model_pipeline.joblib` and successfully reloaded for prediction on
raw, unpreprocessed test rows.
