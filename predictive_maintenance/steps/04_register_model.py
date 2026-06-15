COL_RENAME = {
    'Engine rpm'      : 'Engine_RPM',
    'Lub oil pressure': 'Lub_Oil_Pressure',
    'Fuel pressure'   : 'Fuel_Pressure',
    'Coolant pressure': 'Coolant_Pressure',
    'lub oil temp'    : 'Lub_Oil_Temp',
    'Coolant temp'    : 'Coolant_Temp',
    'Engine Condition': 'Engine_Condition',
}
TARGET       = 'Engine_Condition'
ALL_FEATURES = [
    'Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure',
    'Coolant_Pressure', 'Lub_Oil_Temp', 'Coolant_Temp',
    'temp_pressure_ratio'
]

import os, pickle
import pandas as pd
from huggingface_hub import HfApi
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
HF_USERNAME = "balamanian82"
MODEL_REPO  = f"{HF_USERNAME}/predictive-maintenance-engine-model"

with open("predictive_maintenance/models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

test_df = pd.read_csv("predictive_maintenance/data/processed/test.csv")
X_test, y_test = test_df[ALL_FEATURES], test_df[TARGET]

y_pred = (model.predict_proba(X_test)[:, 1] >= 0.35).astype(int)
print(f"Test Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"Test F1        : {f1_score(y_test, y_pred):.4f}")
print(f"Test Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Test Accuracy  : {accuracy_score(y_test, y_pred):.4f}")

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj="predictive_maintenance/models/best_model.pkl",
    path_in_repo="best_model.pkl",
    repo_id=MODEL_REPO,
    repo_type="model",
    token=HF_TOKEN
)
print(f"Model registered at: https://huggingface.co/{MODEL_REPO}")
