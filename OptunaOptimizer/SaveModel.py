import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from OptunaOptimizer.MLP import DynamicMLP

# =============================================================================
# MODULE: TEST SAVED MODEL / EXPORTED ARTIFACTS
# =============================================================================


def test_saved_model(model_path: str, best_params: dict, input_state: dict, X_test, y_test, scaler_y):
    """
    Loads the trained PyTorch model weights from disk and evaluates it on unseen test data.
    """
    print("\n=== STARTING FINAL MODEL TESTING ===")

    # 1. Recreate the blank model architecture
    input_dim = X_test.shape[1]
    output_dim = input_state["model_spec"]["output_dim"]

    # 2. Setup Device 
    # This automatically uses "cuda" (GPU) if available, otherwise falls back to "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = DynamicMLP(input_dim, output_dim, best_params)

    # 3. Load the saved weights into the model
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)

    # 4. Set model to evaluation mode
    model.eval()

    # 5. Prepare the test data tensor
    X_test_t = torch.FloatTensor(X_test).to(device)

    # 6. Run Inference
    print("Running predictions on the Test Set...")
    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy().flatten()

    # 7. Calculate comprehensive metrics (Real Values using Inverse Transform)
    actuals_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    preds_real = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(actuals_real, preds_real))
    mae = mean_absolute_error(actuals_real, preds_real)
    r2 = r2_score(actuals_real, preds_real)

    print("\n=== FINAL TEST RESULTS (REAL PRICES) ===")
    print(f"RMSE (Root Mean Squared Error): ₹ {rmse:,.2f}")
    print(f"MAE  (Mean Absolute Error):     ₹ {mae:,.2f}")
    print(f"R²   (Coefficient of Det.):     {r2:.4f}")

    # Show a few sample predictions vs actuals
    print("\nSample Predictions:")
    for i in range(min(5, len(actuals_real))):
        diff = actuals_real[i] - preds_real[i]
        print(f"  Actual: ₹ {actuals_real[i]:,.2f} | Predicted: ₹ {preds_real[i]:,.2f} | Diff: ₹ {diff:,.2f}")

    return preds_real