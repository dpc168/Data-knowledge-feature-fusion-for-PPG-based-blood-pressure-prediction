import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler

# Read data
data = pd.read_excel('../1_Feature_extraction/8D_KF.xlsx')
X = data[['t1', 't2', 't3', 'h1', 'h2', 'h3', 'Tsys', 'Tdia']].values
y_sbp = data['Y_S'].values
y_dbp = data['Y_D'].values


def evaluate_model(X, y, n_runs=10):
    """Evaluate model and return results dictionary"""
    all_mae = []
    all_rmse = []

    for i in range(n_runs):
        # Split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=np.random.randint(100))

        # Robust scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Lasso regression
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train_scaled, y_train)
        y_pred = lasso.predict(X_test_scaled)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Collect results
        all_mae.append(mae)
        all_rmse.append(rmse)

    # Calculate mean and standard deviation
    results = {
        'MAE_mean': np.mean(all_mae),
        'RMSE_mean': np.mean(all_rmse),
    }

    return results


# Evaluate SBP model
print("=" * 50)
print("SBP Model Evaluation (10 runs)")
print("=" * 50)
sbp_results = evaluate_model(X, y_sbp, n_runs=10)

# Evaluate DBP model
print("\n" + "=" * 50)
print("DBP Model Evaluation (10 runs)")
print("=" * 50)
dbp_results = evaluate_model(X, y_dbp, n_runs=10)

# Print results
print("\n" + "=" * 50)
print("Final Results Summary (Mean )")
print("=" * 50)
print(f"SBP - MAE: {sbp_results['MAE_mean']:.2f}")
print(f"SBP - RMSE: {sbp_results['RMSE_mean']:.2f}")
print(f"DBP - MAE: {dbp_results['MAE_mean']:.2f}")
print(f"DBP - RMSE: {dbp_results['RMSE_mean']:.2f}")


