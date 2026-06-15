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
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

train_df = pd.read_csv("predictive_maintenance/data/processed/train.csv")
val_df   = pd.read_csv("predictive_maintenance/data/processed/val.csv")

X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET]
X_val,   y_val   = val_df[ALL_FEATURES],   val_df[TARGET]
print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"After SMOTE — Class 0: {sum(y_train_sm==0)}, Class 1: {sum(y_train_sm==1)}")

param_dist = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma":            [0, 0.1, 0.2],
}

search = RandomizedSearchCV(
    XGBClassifier(random_state=1, eval_metric="logloss", verbosity=0),
    param_dist, n_iter=20, scoring="f1",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1),
    random_state=1, n_jobs=-1
)
search.fit(X_train_sm, y_train_sm)
best_xgb = search.best_estimator_
print(f"Best params: {search.best_params_}")

y_val_pred = (best_xgb.predict_proba(X_val)[:, 1] >= 0.35).astype(int)
print(f"Val Recall : {recall_score(y_val, y_val_pred):.4f}")
print(f"Val F1     : {f1_score(y_val, y_val_pred):.4f}")

os.makedirs("predictive_maintenance/models", exist_ok=True)
with open("predictive_maintenance/models/best_model.pkl", "wb") as f:
    pickle.dump(best_xgb, f)
print("Model saved to predictive_maintenance/models/best_model.pkl")
