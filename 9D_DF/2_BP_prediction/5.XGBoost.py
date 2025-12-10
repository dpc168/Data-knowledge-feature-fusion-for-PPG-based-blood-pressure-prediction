# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import warnings
from tqdm import tqdm

# Ignore warning messages
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_excel('../1_Feature_extraction/9D_DF.xlsx')

# Define feature columns and target columns
features = ['a1', 'a2', 'a3', 'c1', 'c2', 'c3', 'b2-b1', 'b3-b2', 'b3-b1']
target_sbp = 'Y_S'  # Systolic blood pressure target
target_dbp = 'Y_D'  # Diastolic blood pressure target

# Prepare feature data and target data
X = data[features]
y_sbp = data[target_sbp]
y_dbp = data[target_dbp]

# Create preprocessing and modeling pipeline
# Note: XGBoost typically doesn't require feature scaling, but we keep RobustScaler for consistency
pipeline = Pipeline([
    ('scaler', RobustScaler()),  # Data standardization
    ('xgb', XGBRegressor())  # XGBoost regression model
])

# Set up XGBoost hyperparameter grid
param_grid = {
    'xgb__n_estimators': [50, 100, 200],
    'xgb__max_depth': [3, 4, 5, 6, 7],
    'xgb__learning_rate': [0.01, 0.15, 0.2],
    'xgb__subsample': [0.7, 0.8],
    'xgb__colsample_bytree': [0.7, 0.8]
}


def evaluate_model(X, y, n_runs=10, test_size=0.2):

    all_mae = []
    all_rmse = []

    # Show progress bar for runs
    for i in tqdm(range(n_runs), desc="Model Running Progress"):
        # Split training and test sets with different random seed each time
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=np.random.randint(100)
        )

        # Grid search for best parameters
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,  # 3-fold cross-validation
            scoring='neg_mean_squared_error',  # Use negative mean squared error as scoring
            n_jobs=-1,  # Use all CPU cores
            verbose=0  # No detailed output
        )

        # Train model
        grid_search.fit(X_train, y_train)

        # Predict test set
        y_pred = grid_search.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)  # Mean absolute error
        rmse = sqrt(mean_squared_error(y_test, y_pred))  # Root mean squared error

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

# Print final results
print("\n" + "=" * 50)
print("Final Results Summary (10-run average )")
print("=" * 50)
print(f"SBP - MAE: {sbp_results['MAE_mean']:.2f}")
print(f"SBP - RMSE: {sbp_results['RMSE_mean']:.2f}")
print(f"DBP - MAE: {dbp_results['MAE_mean']:.2f}")
print(f"DBP - RMSE: {dbp_results['RMSE_mean']:.2f}")

