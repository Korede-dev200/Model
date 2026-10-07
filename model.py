import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

df = pd.read_csv('churn_data.csv')

# Question 1
print(df.shape)
print(df.info())
print(df.isna().sum())

# Question 3
print(df["MonthlySpend"].mean())
print(df["MonthlySpend"].median())
print(df["MonthlySpend"].describe())    

df["MonthlySpend"] = df["MonthlySpend"].fillna(df["MonthlySpend"].median())
print(df["MonthlySpend"].isna().sum())

print(df.shape)

#Question 2
df_encoded = pd.get_dummies(df, columns=["Plan"])
print(df_encoded.head())

# Question 4
X = df_encoded.drop(columns=["CustomerID", "Churned"])
y = df_encoded["Churned"]

print(df.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape)

# Question 5
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled[:5])

# Question 6
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# question 7
cm = confusion_matrix(y_test, predictions, labels=["No", "Yes"])
print(cm)

# Question 8
print("Accuracy can be misleading on imbalanced data because a model can score well just by favoring the majority class. In this dataset, 73.5% of customers didn't churn, so a model that always predicted 'No' would already achieve 73.5% accuracy without learning anything. My actual model reached 82% accuracy, only modestly higher and its recall was just 53%, meaning the model missed 47% of actual churners.")

# Question 9
tree_model = DecisionTreeClassifier(random_state=0)
tree_model.fit(X_train_scaled, y_train)
tree_accuracy = accuracy_score(y_test, tree_model.predict(X_test_scaled))

forest_model = RandomForestClassifier(random_state=0)
forest_model.fit(X_train_scaled, y_train)
forest_accuracy = accuracy_score(y_test, forest_model.predict(X_test_scaled))

print(f"Decision Tree Accuracy: {tree_accuracy:.4f}")
print(f"Random Forest Accuracy: {forest_accuracy:.4f}")

# Question 10
print("Q10. Your Decision Tree scores 98% accuracy on the training set but only 71% on the test set. Name this phenomenon and propose two concrete ways to fix it.\n Name of Phenomenon: Overfitting (or High Variance).\nTwo Concrete Solutions: \nPruning / Constraining Tree Depth: Set hyperparameters like max_depth (e.g., max_depth=5), min_samples_split, or min_samples_leaf to prevent the decision tree from growing arbitrarily deep leaves that memorize noise.\n\nUse an Ensemble Method or Regularization: Switch to a Random Forest Classifier (averaging multiple trees) or apply cost-complexity pruning to drop non-essential leaf branches.")

# Question 11
prop_data = {
    "size_sqm": [50, 75, 100, 120, 150, 200, 250, 300, 85, 110],
    "bedrooms": [1, 2, 2, 3, 3, 4, 4, 5, 2, 3],
    "location_score": [3, 5, 6, 7, 8, 9, 8, 10, 4, 6],
    "price": [
        25000000,
        38000000,
        50000000,
        62000000,
        80000000,
        110000000,
        135000000,
        170000000,
        42000000,
        58000000,
    ],
}
df_house = pd.DataFrame(prop_data)

X_house = df_house[["size_sqm", "bedrooms", "location_score"]]
y_house = df_house["price"]

X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_tr_h, y_tr_h)

y_pred_h = reg_model.predict(X_te_h)
r2 = r2_score(y_te_h, y_pred_h)
mae = mean_absolute_error(y_te_h, y_pred_h)

print(f"R² Score: {r2:.4f}")
print(f"MAE: ₦{mae:,.2f}")


# Question 13
print("MAE (Mean Absolute Error): Measures the average magnitude of absolute errors in original units. It treats all error sizes linearly.MSE \n(Mean Squared Error): Squares errors before averaging. Heavily penalizes large errors, but units are squared (e.g., Naira^2).\nRMSE (Root Mean Squared Error): Takes the square root of MSE to return errors back to original units (e.g.Naira) while retaining sensitivity to large outliers.\n\nWhen to prefer RMSE over MAE: Prefer RMSE when large prediction errors are disproportionately costly for the business. For example, underpredicting a luxury estate by ₦50 Million is vastly worse than making ten ₦5 Million errors; RMSE heavily penalizes that single massive error, forcing the model to minimize large blunders.")

# Question 14
scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")

print(f"5-Fold CV Accuracy Scores: {scores}")
print(f"Mean CV Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
print("Advantage over single split: A single train/test split can yield misleadingly high or low accuracy depending on how luck assigns data points to train vs. test. $k$-Fold Cross-Validation trains and tests $k$ separate times on $k$ different folds, ensuring every sample is used for both training and testing. This produces a much more stable, reliable metric with standard deviation confidence intervals.")

# Question 15
print("Overfitting (High Variance): Occurs when a model is too complex and memorizes noise/outliers in the training set.\nSymptom: High training performance (e.g., 99% accuracy) but poor test performance (e.g., 68% accuracy).\nUnderfitting (High Bias): Occurs when a model is too simple to capture the underlying structure of the data.\nSymptom: Poor performance on both training data and test data (e.g., 55% accuracy on training, 54% on test).")

# Question 16
print("What's wrong: Tuning hyperparameters repeatedly against the test set causes Data Leakage. The test set's information indirectly leaks into model selection decisions. The reported test accuracy becomes overly optimistic because the model was explicitly selected to fit that specific test set.What to do instead: Split data into 3 subsets: Train, Validation, and Test sets (or use k-fold cross-validation on the training set). Tune hyperparameters exclusively on the validation set/cross-validation folds, and evaluate on the held-out test set only once at the very end to measure true unbiased performance.")

# Question 12
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(y_te_h, y_pred_h, color="blue", label="Predictions")
plt.plot(
    [y_house.min(), y_house.max()],
    [y_house.min(), y_house.max()],
    "r--",
    lw=2,
    label="Perfect Prediction Line",
)

plt.xlabel("Actual Price (₦)")
plt.ylabel("Predicted Price (₦)")
plt.title("Q12: Actual vs. Predicted House Prices")
plt.legend()
plt.grid(True)
plt.show()

print("With a test set of only 2 properties (20% of 10 rows), the two predictions fall on opposite sides of the perfect prediction line—one slightly above (overpredicted) and one slightly below (underpredicted).Because the errors are balanced on either side of the line, there is no obvious systematic bias (such as consistently underpredicting or overpredicting). However, because the test set consists of only two data points, N=2 is too small to definitively prove whether the model's errors are purely random across all price ranges. A larger dataset would be needed to thoroughly evaluate error variance.")
