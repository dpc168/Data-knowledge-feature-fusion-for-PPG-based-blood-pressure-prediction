import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from math import sqrt

# 1. Load data
data = pd.read_excel('../1_Feature_extraction/8D_KF.xlsx', sheet_name='Sheet1')

# 2. Define feature columns and target columns
features = ['t1', 't2', 't3', 'h1', 'h2', 'h3', 'Tsys', 'Tdia']
target_sbp = 'Y_S'  # Systolic blood pressure target column
target_dbp = 'Y_D'  # Diastolic blood pressure target column

# 3. Data preprocessing
X = data[features]
y_sbp = data[target_sbp]
y_dbp = data[target_dbp]

# 4. Create random forest regression pipeline
pipe_rf = Pipeline([
    ('scaler', RobustScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])

# 5. Define hyperparameter grid

param_grid_rf = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [8, 10, 15],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],

}

def evaluate_model(X, y, n_runs=10):
    """Evaluate model and return results dictionary"""
    all_mae = []
    all_rmse = []

    for run in range(n_runs):
        # Data splitting
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=np.random.randint(100))

        # Train model
        grid = GridSearchCV(pipe_rf, param_grid_rf, cv=3,
                            scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)

        # Predict
        y_pred = grid.best_estimator_.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = sqrt(mean_squared_error(y_test, y_pred))

        # Collect results
        all_mae.append(mae)
        all_rmse.append(rmse)

    # Calculate mean and standard deviation
    results = {
        'MAE_mean': np.mean(all_mae),
        'RMSE_mean': np.mean(all_rmse),
    }

    return results


# Evaluate systolic blood pressure (SBP) model
print("=" * 50)
print("Systolic Blood Pressure (SBP) Model Evaluation (10 runs)")
print("=" * 50)
sbp_results = evaluate_model(X, y_sbp, n_runs=10)

# Evaluate diastolic blood pressure (DBP) model
print("\n" + "=" * 50)
print("Diastolic Blood Pressure (DBP) Model Evaluation (10 runs)")
print("=" * 50)
dbp_results = evaluate_model(X, y_dbp, n_runs=10)

# Print results
print("\n" + "=" * 50)
print("Final Results Summary (10 runs mean)")
print("=" * 50)
print(f"SBP - MAE: {sbp_results['MAE_mean']:.2f}")
print(f"SBP - RMSE: {sbp_results['RMSE_mean']:.2f}")
print(f"DBP - MAE: {dbp_results['MAE_mean']:.2f}")
print(f"DBP - RMSE: {dbp_results['RMSE_mean']:.2f}")


