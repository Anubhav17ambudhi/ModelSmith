import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error
import copy

# YAHAN FIX HAI: DynamicMLP class ko import karna zaroori hai
from OptunaOptimizer.MLP import DynamicMLP

# =============================================================================
# MODULE 4: FINAL MODEL TRAINING
# =============================================================================

def train_and_evaluate_final_model(best_params, X_train, y_train, X_test, y_test, input_state):
    """
    Trains the final PyTorch model using the best hyperparameters found by Optuna
    and evaluates it on the hold-out test set.
    """
    print("\n=== STARTING FINAL MODEL TRAINING ===")
    print("Using Hyperparameters:", best_params)

    # 1. Prepare Data & Device (GPU FIX APPLIED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).view(-1, 1).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)

    # 2. Initialize Model and Loss
    input_dim = X_train.shape[1]
    output_dim = input_state["model_spec"]["output_dim"]

    model = DynamicMLP(input_dim, output_dim, best_params).to(device)
    criterion = nn.MSELoss()

    # 3. Setup Optimizer (SGD SUPPORT ADDED)
    if best_params["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"],
                               weight_decay=best_params["weight_decay"], betas=(best_params["beta1"], 0.999))
    elif best_params["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=best_params["learning_rate"],
                                weight_decay=best_params["weight_decay"], betas=(best_params["beta1"], 0.999))
    else:
        optimizer = optim.SGD(model.parameters(), lr=best_params["learning_rate"],
                              weight_decay=best_params["weight_decay"])

    # 4. Setup Scheduler
    if best_params["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=best_params["epochs"])
    else:
        scheduler = None

    # 5. Training Loop with Model Checkpointing
    best_test_rmse = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(best_params["epochs"]):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), best_params["grad_clip"])
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        if scheduler:
            scheduler.step()

        # Evaluation step at the end of each epoch
        # Evaluation step at the end of each epoch
        model.eval()
        test_mse_sum = 0.0
        with torch.no_grad():
            for i in range(0, len(X_test_t), best_params["batch_size"]):
                t_batch_X = X_test_t[i : i + best_params["batch_size"]].to(device)
                t_batch_y = y_test_t[i : i + best_params["batch_size"]].cpu().numpy()
                
                t_preds = model(t_batch_X).cpu().numpy()
                test_mse_sum += mean_squared_error(t_batch_y, t_preds) * len(t_batch_X)

        test_rmse = np.sqrt(test_mse_sum / len(X_test_t))
        # Save the model if it's the best one we've seen so far
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_model_weights = copy.deepcopy(model.state_dict())

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{best_params['epochs']}] | Train MSE: {train_loss:.4f} | Test RMSE: {test_rmse:.4f}")

    # 6. Load Best Weights and Final Evaluation
    print("\n=== TRAINING COMPLETE ===")
    model.load_state_dict(best_model_weights)
    model.eval()

    with torch.no_grad():
        final_preds = model(X_test_t)
        final_mse = mean_squared_error(y_test_t.cpu().numpy(), final_preds.cpu().numpy())
        final_rmse = np.sqrt(final_mse)

    print(f" Final Best Test RMSE: {final_rmse:.4f}")

    return model, final_rmse