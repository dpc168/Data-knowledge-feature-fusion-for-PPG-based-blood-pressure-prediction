
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create results directory
os.makedirs('results2', exist_ok=True)

# 1. Data loading and preprocessing
# Read data from Excel file
data = pd.read_excel('4_Gaussian_features_BP_with_baseline.xlsx')

# 2. Define features and target columns
features = ['a1', 'a2', 'a3', 'c1', 'c2', 'c3', 'b2-b1', 'b3-b2', 'b3-b1', 'Age(year)', 'Height(cm)', 'Weight(kg)',
            'Sex(M/F)']
X = data[features]
y = data[['Y_S', 'Y_D']]  # Predict both systolic and diastolic blood pressure

# Save sample names
sample_names = data['Name'] if 'Name' in data.columns else [f'Sample_{i}' for i in range(len(data))]

# 3. Data standardization
scaler_X = RobustScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = RobustScaler()
y_scaled = scaler_y.fit_transform(y)

# 4. Convert to PyTorch tensors and split dataset
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(device)
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor.cpu(), y_tensor.cpu(), test_size=0.2, random_state=42)

# Move data back to GPU
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)


# 5. Define neural network model
class BPRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(BPRegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Mish(),
            nn.Linear(32, output_dim)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


# 6. Model training setup
model = BPRegressionModel(X_train.shape[1], output_dim=2).to(device)
criterion = nn.HuberLoss(delta=1.5)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 7. Model training
num_epochs = 300
best_mae = float('inf')
best_model_path = 'results2/best_model.pth'

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    # Validation phase
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        # Move data back to CPU for inverse transformation
        test_pred = scaler_y.inverse_transform(test_outputs.cpu().numpy())
        test_true = scaler_y.inverse_transform(y_test.cpu().numpy())

        # Calculate evaluation metrics
        test_mae_sys = mean_absolute_error(test_true[:, 0], test_pred[:, 0])
        test_mae_dia = mean_absolute_error(test_true[:, 1], test_pred[:, 1])
        test_mae = (test_mae_sys + test_mae_dia) / 2

        scheduler.step(test_mae)
        if test_mae < best_mae:
            best_mae = test_mae
            torch.save(model.state_dict(), best_model_path)

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Test MAE: {test_mae:.2f}')

# 8. Load best model and perform final evaluation
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

with torch.no_grad():
    # Predict for all samples
    predictions_scaled = model(X_tensor)
    predictions = scaler_y.inverse_transform(predictions_scaled.cpu().numpy())
    y_true = scaler_y.inverse_transform(y_tensor.cpu().numpy())

    # Calculate evaluation metrics
    mae_sys = mean_absolute_error(y_true[:, 0], predictions[:, 0])
    mae_dia = mean_absolute_error(y_true[:, 1], predictions[:, 1])
    rmse_sys = np.sqrt(mean_squared_error(y_true[:, 0], predictions[:, 0]))
    rmse_dia = np.sqrt(mean_squared_error(y_true[:, 1], predictions[:, 1]))

    # Save prediction results to CSV file
    results_df = pd.DataFrame({
        'Name': sample_names,
        'SBP_Predicted': predictions[:, 0],
        'DBP_Predicted': predictions[:, 1],
        'SBP_Actual': y_true[:, 0],
        'DBP_Actual': y_true[:, 1]
    })

    # Save to file
    results_file = 'results2/bp_predictions.csv'
    results_df.to_csv(results_file, index=False)
    print(f'\nPrediction results have been saved to: {results_file}')

    # Print predictions for first 10 samples as an example
    print('\nPredictions for first 10 samples:')
    print(results_df.head(10).to_string(index=False))

# Print final results
print("\n" + "=" * 50)
print("DNN Model Evaluation Results")
print("=" * 50)
print(f"SBP - MAE: {mae_sys:.2f}")
print(f"SBP - RMSE: {rmse_sys:.2f}")
print(f"DBP - MAE: {mae_dia:.2f}")
print(f"DBP - RMSE: {rmse_dia:.2f}")
print("=" * 50)
