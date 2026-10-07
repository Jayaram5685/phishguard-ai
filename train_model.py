import os

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define the path to the data file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, 'DataFiles', '5.urldata.csv')
model_dir = os.path.join(current_dir, 'models')

# Load the dataset
data = pd.read_csv(data_file_path)

# Optional: strip whitespace from column names
data.columns = data.columns.str.strip()

# Extract features and labels
X = data.drop(columns=['Label', 'Domain'])  # Drop 'Domain' because it's string
y = data['Label']

# Encode categorical/string features if any
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(
    'WARNING: this is the legacy random split. The dataset contains massive '
    'near-duplicate overlap between train and test (see models/MODEL_CARD.md), '
    'so the metrics below measure memorisation, not generalisation. '
    'The domain-disjoint protocol lands in milestone P2/P8.'
)


def report(name: str, y_true, y_pred) -> None:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    print(
        f'{name}: accuracy={accuracy_score(y_true, y_pred) * 100:.2f}% '
        f'precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}'
    )


# Train a Tuned Random Forest model
print("Training Tuned Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
rf_model.fit(X_train, y_train)
report('Random Forest', y_test, rf_model.predict(X_test))

# Train a real, tuned XGBoost model
print("Training Tuned XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
report('XGBoost', y_test, xgb_model.predict(X_test))

# Save with explicit, honest names. One artifact per algorithm — upstream used
# to write the Random Forest to a file called "xg_boost.pkl".
os.makedirs(model_dir, exist_ok=True)
joblib.dump(rf_model, os.path.join(model_dir, 'random_forest.joblib'))
joblib.dump(xgb_model, os.path.join(model_dir, 'xgboost.joblib'))
print(f'Saved models to {model_dir}')
