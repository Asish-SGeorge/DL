import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Salary_Data.csv")

X_single = data[['YearsExperience']]
y_single = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X_single, y_single, test_size=0.2, random_state=42
)

single_model = LinearRegression()
single_model.fit(X_train, y_train)

y_pred_single = single_model.predict(X_test)

print("\n===== SIMPLE LINEAR REGRESSION =====")
print("R2 Score:", r2_score(y_test, y_pred_single))
print("MSE:", mean_squared_error(y_test, y_pred_single))

data = pd.read_csv("50_Startups.csv")

X_multi = data[
    ['R&D Spend', 'Administration', 'Marketing Spend']
]
y_multi = data['Profit']

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

y_pred_multi = multi_model.predict(X_test)

print("\n===== MULTIPLE LINEAR REGRESSION =====")
print("R2 Score:", r2_score(y_test, y_pred_multi))
print("MSE:", mean_squared_error(y_test, y_pred_multi))

data = pd.read_csv("Salary_Data.csv")

X_single = data[['YearsExperience']]
y_single = data['Salary']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_single)

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y_single, test_size=0.2, random_state=42
)

poly_model = LinearRegression()
poly_model.fit(X_train, y_train)

y_pred_poly = poly_model.predict(X_test)

print("\n===== POLYNOMIAL REGRESSION =====")
print("R2 Score:", r2_score(y_test, y_pred_poly))
print("MSE:", mean_squared_error(y_test, y_pred_poly))

print("\nAll three regression experiments completed successfully.")