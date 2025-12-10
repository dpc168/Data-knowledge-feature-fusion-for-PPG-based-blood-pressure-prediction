# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import warnings
import warnings
import os


file_dir = os.listdir('../9D_DKF/')

# 按数值大小排序（去掉%号后转数字）
sorted_dir = sorted(file_dir, key=lambda x: float(x.replace('%', '')))

# 创建空列表存储所有结果
all_results = []

for current_dir in sorted_dir:

    current_path = '../9D_DKF/' + str(current_dir) + '/' + '1_Feature_extraction/4_Gaussian_features_BP.xlsx'
    print('---------------------------------')
    print(current_path)
    print('---------------------------------')

    warnings.filterwarnings('ignore')

    # Load dataset
    data = pd.read_excel(current_path)

    # Define features and target columns
    features = ['a1', 'a2', 'a3', 'c1', 'c2', 'c3', 'b2-b1', 'b3-b2', 'b3-b1']
    target_sbp = 'Y_S'  # Systolic blood pressure target column
    target_dbp = 'Y_D'  # Diastolic blood pressure target column

    # Directly extract features and target variables
    X = data[features]
    y_sbp = data[target_sbp]
    y_dbp = data[target_dbp]

    # Create linear regression pipeline
    pipe_lr = Pipeline([
        ('scaler', RobustScaler()),
        ('lr', LinearRegression())
    ])

    # Define hyperparameter grid
    param_grid_lr = {
        'lr__fit_intercept': [True, False],
        'lr__copy_X': [True, False],
        'lr__positive': [False]
    }


    def evaluate_model(X, y, n_runs=10):
        """Evaluate model and return results dictionary"""
        all_mae = []
        all_rmse = []

        # Create repeated KFold cross-validator
        rkf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

        for run in range(n_runs):
            # Data splitting
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=np.random.randint(100))

            # Train model with repeated cross-validation
            grid = GridSearchCV(pipe_lr, param_grid_lr, cv=rkf,
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


    # Main program execution

        # Evaluate systolic blood pressure (SBP) model
    print("=" * 50)
    print("Systolic Blood Pressure (SBP) Model Evaluation (10 runs)")
    print("Using 10-fold 3-repeat cross-validation")
    print("=" * 50)
    sbp_results = evaluate_model(X, y_sbp, n_runs=10)

        # Evaluate diastolic blood pressure (DBP) model
    print("\n" + "=" * 50)
    print("Diastolic Blood Pressure (DBP) Model Evaluation (10 runs)")
    print("Using 10-fold 3-repeat cross-validation")
    print("=" * 50)
    dbp_results = evaluate_model(X, y_dbp, n_runs=10)

    # Print results summary
    print("\n" + "=" * 50)
    print("Final Results Summary (10 runs mean ± standard deviation)")
    print("Using 10-fold 3-repeat cross-validation")
    print("=" * 50)
    print(f"SBP - MAE: {sbp_results['MAE_mean']:.2f}")
    print(f"SBP - RMSE: {sbp_results['RMSE_mean']:.2f}")
    print(f"DBP - MAE: {dbp_results['MAE_mean']:.2f}")
    print(f"DBP - RMSE: {dbp_results['RMSE_mean']:.2f}")

    # 保存当前current_dir的结果到列表
    result_row = {
        'current_dir': current_dir,
        'SBP_MAE': sbp_results['MAE_mean'],
        'SBP_RMSE': sbp_results['RMSE_mean'],
        'DBP_MAE': dbp_results['MAE_mean'],
        'DBP_RMSE': dbp_results['RMSE_mean']
    }
    all_results.append(result_row)

# 将结果转换为DataFrame
results_df = pd.DataFrame(all_results)

# 创建新的DataFrame，按照指定的列顺序排列
final_results_df = pd.DataFrame()

# 按照要求的顺序添加列
final_results_df['current_dir'] = results_df['current_dir']  # 第一列

# # 添加空列作为第二到第13列
# for i in range(2, 14):
#     final_results_df[f'Column_{i}'] = ""  # 空列

# 添加所需的指标列
final_results_df['SBP_MAE'] = results_df['SBP_MAE']  # 第十列
final_results_df['SBP_RMSE'] = results_df['SBP_RMSE']  # 第十一列
final_results_df['DBP_MAE'] = results_df['DBP_MAE']  # 第十二列
final_results_df['DBP_RMSE'] = results_df['DBP_RMSE']  # 第十三列

# 保存到result.xlsx文件
final_results_df.to_excel('result4.xlsx', index=False)

print("\n" + "=" * 50)
print("所有结果已保存到 result4.xlsx 文件")
print("=" * 50)
print(f"共处理了 {len(all_results)} 个目录")
print("结果文件包含以下列：")
print(final_results_df.columns.tolist())