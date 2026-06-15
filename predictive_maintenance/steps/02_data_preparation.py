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

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HF_USERNAME  = "balamanian82"
DATASET_REPO = f"{HF_USERNAME}/predictive-maintenance-engine"

# Load raw CSV and normalise column names
df = pd.read_csv("predictive_maintenance/data/raw/engine_data.csv")
df.rename(columns=COL_RENAME, inplace=True)
print("Columns after rename:", df.columns.tolist())

# Feature engineering
df["temp_pressure_ratio"] = df["Lub_Oil_Temp"] / (df["Coolant_Pressure"] + 1e-6)

# Drop nulls
df = df.dropna()

X = df[ALL_FEATURES]
y = df[TARGET]

# random_state=1 consistent with notebook
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.10, stratify=y_temp, random_state=1)

train_df = X_train.copy(); train_df[TARGET] = y_train.values
val_df   = X_val.copy();   val_df[TARGET]   = y_val.values
test_df  = X_test.copy();  test_df[TARGET]  = y_test.values

os.makedirs("predictive_maintenance/data/processed", exist_ok=True)
train_df.to_csv("predictive_maintenance/data/processed/train.csv", index=False)
val_df.to_csv("predictive_maintenance/data/processed/val.csv",     index=False)
test_df.to_csv("predictive_maintenance/data/processed/test.csv",   index=False)
print(f"Splits — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

api = HfApi(token=HF_TOKEN)
for split in ["train", "val", "test"]:
    api.upload_file(
        path_or_fileobj=f"predictive_maintenance/data/processed/{split}.csv",
        path_in_repo=f"data/processed/{split}.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print(f"Uploaded {split}.csv to HF Hub.")
