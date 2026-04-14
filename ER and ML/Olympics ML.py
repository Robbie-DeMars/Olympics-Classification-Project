from pyspark.sql import SparkSession
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd

df = pd.read.parquet("ml_data")

X = df.drop(columns=["label", "athlete_id"])
y = df["label"]

# Label encoders
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    label_encoder = LabelEncoder()
    X[col] = label_encoder.fit_transform(X[col].astype(str))
    label_encoders[col] = label_encoder

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("LR Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

#  ================ RF as well to compare =============
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("RF Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# ================ Smote with RF as well ===============
smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

rf_smote = RandomForestClassifier(n_estimators=100)
rf_smote.fit(X_resampled, y_resampled)

y_pred_smote = rf_smote.predict(X_test)

print("Random Forest with SMOTE Accuracy:", accuracy_score(y_test, y_pred_smote))
print(classification_report(y_test, y_pred_smote))
