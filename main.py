import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Dataset URL
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

# Load dataset
df = pd.read_csv(url, header=None)

# Column names
headers = [
    "symboling","normalized-losses","make","fuel-type","aspiration",
    "num-of-doors","body-style","drive-wheels","engine-location",
    "wheel-base","length","width","height","curb-weight","engine-type",
    "num-of-cylinders","engine-size","fuel-system","bore","stroke",
    "compression-ratio","horsepower","peak-rpm","city-mpg",
    "highway-mpg","price"
]

df.columns = headers

# Show first 10 rows
print(df.head(10))

# Dataset shape
print("\nDataset Shape:")
print(df.shape)
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print((df == "?").sum())
import numpy as np

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

print("\nMissing Values After Replacing ?:")
print(df.isnull().sum())

# Convert columns to numeric
df["normalized-losses"] = pd.to_numeric(df["normalized-losses"])
df["bore"] = pd.to_numeric(df["bore"])
df["stroke"] = pd.to_numeric(df["stroke"])
df["horsepower"] = pd.to_numeric(df["horsepower"])
df["peak-rpm"] = pd.to_numeric(df["peak-rpm"])
df["price"] = pd.to_numeric(df["price"])

# Fill missing numerical values with mean
df["normalized-losses"] = df["normalized-losses"].fillna(df["normalized-losses"].mean())

df["bore"] = df["bore"].fillna(df["bore"].mean())

df["stroke"] = df["stroke"].fillna(df["stroke"].mean())

df["horsepower"] = df["horsepower"].fillna(df["horsepower"].mean())

df["peak-rpm"] = df["peak-rpm"].fillna(df["peak-rpm"].mean())

# Convert columns to numeric
df["normalized-losses"] = pd.to_numeric(df["normalized-losses"])
df["bore"] = pd.to_numeric(df["bore"])
df["stroke"] = pd.to_numeric(df["stroke"])
df["horsepower"] = pd.to_numeric(df["horsepower"])
df["peak-rpm"] = pd.to_numeric(df["peak-rpm"])
df["price"] = pd.to_numeric(df["price"])

# Fill missing numerical values
df["normalized-losses"] = df["normalized-losses"].fillna(df["normalized-losses"].mean())

df["bore"] = df["bore"].fillna(df["bore"].mean())

df["stroke"] = df["stroke"].fillna(df["stroke"].mean())

df["horsepower"] = df["horsepower"].fillna(df["horsepower"].mean())

df["peak-rpm"] = df["peak-rpm"].fillna(df["peak-rpm"].mean())

# Fill missing categorical values
df["num-of-doors"] = df["num-of-doors"].fillna(
    df["num-of-doors"].mode()[0]
)

# Drop rows with missing price
df.dropna(subset=["price"], inplace=True)

# Final check
print("\nFinal Missing Values:")
print(df.isnull().sum())

print("\nCleaned Dataset:")
print(df.head(10))

print("\nDescriptive Statistics:")
print(df.describe())

# Feature engineering
df["city-L/100km"] = 235 / df["city-mpg"]

print(df[["city-mpg", "city-L/100km"]].head())
# Horsepower binning
bins = [0, 100, 150, 300]
group_names = ["Low", "Medium", "High"]

df["horsepower-binned"] = pd.cut(
    df["horsepower"],
    bins=bins,
    labels=group_names,
    include_lowest=True
)

print(df["horsepower-binned"].value_counts())

# Normalize length feature
df["length-normalized"] = (
    df["length"] - df["length"].min()
) / (
    df["length"].max() - df["length"].min()
)

print(df["length-normalized"].head())
#histogram
plt.figure(figsize=(8,5))

plt.hist(df["price"], bins=20)

plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")

plt.show()
#horse prower histogram
plt.figure(figsize=(8,5))

plt.hist(df["horsepower"], bins=20)

plt.title("Horsepower Distribution")
plt.xlabel("Horsepower")
plt.ylabel("Frequency")

plt.show()

#BOXplot BODY vs STYLE

plt.figure(figsize=(10,6))

sns.boxplot(x="body-style", y="price", data=df)

plt.title("Price vs Body Style")

plt.show()

#correlation heat map

plt.figure(figsize=(12,8))

numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

# Features
X = df[["horsepower", "engine-size"]]

# Target
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R² Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
# actual VS Predicted
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Prices")

plt.show()

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)

print("Random Forest R²:", rf_r2)
print("Random Forest MAE:", rf_mae)
print("Random Forest MSE:", rf_mse)

# 70/30 split
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Train Random Forest again
rf_model2 = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model2.fit(X_train2, y_train2)

# Predict
rf_pred2 = rf_model2.predict(X_test2)

# Evaluate
print("\n70/30 Split Results")

print("R²:", r2_score(y_test2, rf_pred2))
print("MAE:", mean_absolute_error(y_test2, rf_pred2))
print("MSE:", mean_squared_error(y_test2, rf_pred2))


# Create price categories
df["price-category"] = pd.qcut(
    df["price"],
    3,
    labels=["Low", "Medium", "High"]
)

print(df["price-category"].value_counts())

X_class = df[["horsepower", "engine-size"]]

y_class = df["price-category"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42
)   

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_c, y_train_c)

y_pred_c = log_model.predict(X_test_c)

print("Accuracy:", accuracy_score(y_test_c, y_pred_c))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_c, y_pred_c))

print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_c))


plt.figure(figsize=(8,6))

plt.scatter(y_test, rf_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Random Forest: Actual vs Predicted Prices")

plt.show()
residuals = y_test - y_pred

plt.figure(figsize=(8,6))

plt.scatter(y_pred, residuals)

plt.axhline(y=0, color='red')

plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.show()
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)
plt.figure(figsize=(8,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance
)

plt.title("Feature Importance")

plt.show()