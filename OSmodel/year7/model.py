import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, matthews_corrcoef
import warnings
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings('ignore')

train_data = pd.read_csv('trainSet.csv', index_col=0)
test_data = pd.read_csv('testSet.csv', index_col=0)

X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class'].apply(lambda x: 1 if x == 'Pos' else 0)

X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class'].apply(lambda x: 1 if x == 'Pos' else 0)

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LogisticRegression': LogisticRegression(random_state=42)
}

param_grids = {
    'RandomForest': {'n_estimators': [50, 100, 200, 500]},
    'XGBoost': {'max_depth': [3, 5, 7, 10]},
    'LogisticRegression': {'C': [0.1, 1, 5, 10]}
}

best_models = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='recall')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best Model Parameters for {model_name}: {grid_search.best_params_}")
    best_models[model_name] = best_model

results = []
importance_results = []

for i in range(10):
    print(f"\nIteration {i + 1}/10")

    X_test_sample, _, y_test_sample, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=i)

    for model_name, best_model in best_models.items():
        print(f"Evaluating {model_name} in iteration {i + 1}...")

        y_pred = best_model.predict(X_test_sample)
        y_pred_prob = best_model.decision_function(X_test_sample) if model_name == 'LinearSVC' else best_model.predict_proba(X_test_sample)[:, 1]

        accuracy = accuracy_score(y_test_sample, y_pred)
        precision = precision_score(y_test_sample, y_pred)
        recall = recall_score(y_test_sample, y_pred)
        f1 = f1_score(y_test_sample, y_pred)
        roc_auc = roc_auc_score(y_test_sample, y_pred_prob)
        mcc = matthews_corrcoef(y_test_sample, y_pred)

        print(f'{model_name} - AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}')

        results.append({
            'Iteration': i + 1,
            'Model': model_name,
            'AUC': roc_auc,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1Score': f1,
            'MCC': mcc
        })

        perm_importance = permutation_importance(best_model, X_test_sample, y_test_sample, n_repeats=5, random_state=42)
        perm_importance.importances_mean[perm_importance.importances_mean < 0] = 0
        total_importance = perm_importance.importances_mean.sum() + 0.1
        relative_importance = perm_importance.importances_mean / total_importance

        for feature_idx in range(X_test_sample.shape[1]):
            importance_results.append({
                'Iteration': i + 1,
                'Model': model_name,
                'Feature': X_train.columns[feature_idx],
                'RelativeImportance': relative_importance[feature_idx]
            })

        print(f"\n{model_name} Feature Importances (Relative):")
        for feature, importance in zip(X_train.columns, relative_importance):
            print(f"{feature}: {importance:.4f}")

results_df = pd.DataFrame(results)
importance_df = pd.DataFrame(importance_results)

results_df.to_csv('result.csv', index=False)
importance_df.to_csv('feature_importance.csv', index=False)

rf_best_model = best_models['RandomForest']
year = 7
model_filename = f"random_forest_model_year{year}.pkl"
joblib.dump(rf_best_model, model_filename)
print(f"Random Forest model for year {year} has been saved as '{model_filename}'")
