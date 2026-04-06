import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# main.py
import os
import json
import torch
import optuna
from dotenv import load_dotenv

# Apne banaye hue modules se functions import kar rahe hain
from ML_Pipeline.InputState import InputStateBuilder
from ML_Pipeline.ConstraintEngine import run_constraint_engine
from ML_Pipeline.PrepareDataset import prepare_datasets
from OptunaOptimizer.MLP import create_objective
from OptunaOptimizer.Train import train_and_evaluate_final_model
from OptunaOptimizer.SaveModel import test_saved_model  # <--- NAYA IMPORT

# .env file se API key load karne ke liye
load_dotenv()

def start_automl(csv_path, target, use_case, user_req):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env file! Please add it.")
        return
        
    print(f"\n🚀 Starting AutoML Pipeline for Target: '{target}'")
    
    # 1. Input State Builder
    builder = InputStateBuilder(api_key=api_key)
    input_state = builder.build(csv_path, use_case, user_req, target)
    
    # 2. Constraint Engine
    final_space = run_constraint_engine(input_state, api_key)
    
    # 3. Data Preparation
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_y = prepare_datasets(csv_path, target)
    
    # 4. Optuna Optimization
    objective = create_objective(X_train, y_train, X_val, y_val, input_state, final_space)
    study = optuna.create_study(direction=input_state["objective"]["optuna_direction"])
    study.optimize(objective, n_trials=10) # Testing ke liye 10 trials
    
    # 5. Final Training
    model, score = train_and_evaluate_final_model(study.best_params, X_train, y_train, X_test, y_test, input_state)
    
    # 6. Saving (Exporting artifacts)
    print("\n💾 Exporting Model Artifacts...")
    safe_name = target.replace(" ", "_").lower()
    model_save_path = f"{safe_name}_best_model.pth"
    config_save_path = f"{safe_name}_model_config.json"

    # PyTorch weights save karein
    torch.save(model.state_dict(), model_save_path)
    
    # Blueprint/Config save karein
    deployment_config = {
        "input_state": input_state,
        "best_params": study.best_params
    }
    with open(config_save_path, "w") as f:
        json.dump(deployment_config, f, indent=4)
        
    print(f"✅ Pipeline Complete. Best RMSE Score: {score:.4f}")
    print(f"📂 Artifacts saved successfully:\n  - {model_save_path}\n  - {config_save_path}")

    # 7. Final Verification (Testing the saved model)
    _ = test_saved_model(model_save_path, study.best_params, input_state, X_test, y_test)

import argparse

if __name__ == "__main__":
    # Command Line Arguments setup kar rahe hain
    parser = argparse.ArgumentParser(description="Run AutoML Pipeline")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the dataset CSV")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--use_case", type=str, required=True, help="Description of the use case")
    parser.add_argument("--req", type=str, required=True, help="User constraints and requirements")
    
    args = parser.parse_args()
    
    # Ab data command line se aayega, hardcoded nahi
    start_automl(
        csv_path=args.csv_path, 
        target=args.target, 
        use_case=args.use_case, 
        user_req=args.req
    )