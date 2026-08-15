"""
Train and save the final XGBoost fraud detection model.
Run this script once to produce the model artifact used by the API.
"""
import pandas as pd
import joblib
from xgboost import XGBClassifier

# Load training data
train_df = pd.read_parquet('data/processed/train.parquet')
X_train, y_train = train_df.drop(columns=['Class']), train_df['Class']

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',
    random_state=42
)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'models/xgboost_fraud_model.pkl')
print("Model saved to models/xgboost_fraud_model.pkl")