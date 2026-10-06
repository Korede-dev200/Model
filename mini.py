import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv("churn_data.csv")

# ==========================================
# 2. CLEAN & PREPROCESS
# ==========================================
# Impute missing values with median
df["MonthlySpend"] = df["MonthlySpend"].fillna(df["MonthlySpend"].median())

# Encode Categorical Variables
df_encoded = pd.get_dummies(df, columns=["Plan"], drop_first=True)

# Separate Features (X) and Target (y)
X = df_encoded.drop(columns=["CustomerID", "Churned"])
y = df_encoded["Churned"]

# ==========================================
# 3. TRAIN / TEST SPLIT & SCALING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. TRAIN MODEL
# ==========================================
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# ==========================================
# 5. EVALUATE MODEL
# ==========================================
y_pred = clf.predict(X_test_scaled)

print("--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

# ==========================================
# 6. FEATURE IMPORTANCE & BUSINESS RECOMMENDATION
# ==========================================
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(
    ascending=False
)

print("\n--- FEATURE IMPORTANCES ---")
print(importances)

# Conclusion & Recommendation
top_feature = importances.index[0]
print(f"\n==========================================")
print(f"BUSINESS CONCLUSION & RECOMMENDATION:")
print(f"The top driver of customer churn is: '{top_feature}'.")
print(
    f"Recommendation: For customers flagged with high risk of churn, the marketing/retention team"
)
print(
    f"should target intervention campaigns focused on {top_feature} (e.g. offering discounts on high MonthlySpend or proactive support for high SupportTickets)."
)
print(f"==========================================")