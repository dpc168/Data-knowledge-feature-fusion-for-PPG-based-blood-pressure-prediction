import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_excel('../1_Feature_extraction/8D_KF+9D_DF.xlsx')

# Define features and target columns
features = ['a1', 'a2', 'a3', 'c1', 'c2', 'c3', 'b2-b1', 'b3-b2', 'b3-b1']
target_sbp = 'Y_S'  # Systolic Blood Pressure
target_dbp = 'Y_D'  # Diastolic Blood Pressure

X = data[features]
y_sbp = data[target_sbp]
y_dbp = data[target_dbp]

# Create preprocessing and modeling pipeline
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('svr', SVR())  # Support Vector Regression
])

# Grid search parameter settings
param_grid = {
    'svr__C': [0.1, 0.5, 2, 5],  # Regularization parameter
    'svr__epsilon': [6, 8, 10, 12],  # Epsilon-insensitive region
    'svr__kernel': ['linear', 'rbf'],  # Kernel function type
    'svr__gamma': ['scale']
}

def evaluate_model(X, y, n_runs=10, test_size=0.2):
    """Evaluate model and return average results over n_runs"""
    all_mae = []
    all_rmse = []

    for i in tqdm(range(n_runs), desc="Model Running Progress"):
        # Use different random seed for train-test split each time
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=np.random.randint(100)
        )

        # Grid search for best parameters
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )

        # Train model
        grid_search.fit(X_train, y_train)

        # Predict on test set
        y_pred = grid_search.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = sqrt(mean_squared_error(y_test, y_pred))

        # Collect results
        all_mae.append(mae)
        all_rmse.append(rmse)

    # Calculate mean values
    results = {
        'MAE_mean': np.mean(all_mae),
        'RMSE_mean': np.mean(all_rmse)
    }

    return results

# Evaluate systolic blood pressure model
print("=" * 50)
print("Systolic Blood Pressure Model Evaluation (10 runs)")
print("=" * 50)
sbp_results = evaluate_model(X, y_sbp, n_runs=10)

# Evaluate diastolic blood pressure model
print("\n" + "=" * 50)
print("Diastolic Blood Pressure Model Evaluation (10 runs)")
print("=" * 50)
dbp_results = evaluate_model(X, y_dbp, n_runs=10)

# Print results
print("\n" + "=" * 50)
print("Final Results Summary (10-run average)")
print("=" * 50)
print(f"Systolic BP - MAE: {sbp_results['MAE_mean']:.2f}")
print(f"Systolic BP - RMSE: {sbp_results['RMSE_mean']:.2f}")
print(f"Diastolic BP - MAE: {dbp_results['MAE_mean']:.2f}")
print(f"Diastolic BP - RMSE: {dbp_results['RMSE_mean']:.2f}")

