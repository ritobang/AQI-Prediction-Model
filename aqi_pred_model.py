import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

file_path = "city_day_cleaned.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df.drop(columns=['Date'], inplace=True)
df = pd.get_dummies(df, columns=['City'], drop_first=True)

Q1 = df['AQI'].quantile(0.25)
Q3 = df['AQI'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['AQI'] >= (Q1 - 1.5 * IQR)) & (df['AQI'] <= (Q3 + 1.5 * IQR))]

X = df.drop(columns=['AQI', 'AQI_Bucket'])
y = df['AQI']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

def evaluate_model(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    print(f"{model_name} Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R^2 Score: {r2_score(y_test, y_pred):.2f}\n")
    return mse

evaluate_model(y_test, y_pred_xgb, "XGBoost")

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_xgb, alpha=0.5, label='XGBoost', color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='black', label='Perfect Prediction')
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.legend()
plt.show()


def predict_aqi(date, city):
    date = pd.to_datetime(date)
    year, month, day = date.year, date.month, date.day

    input_data = pd.DataFrame([[year, month, day] + [0] * (X.shape[1] - 3)], columns=X.columns)

    city_column = f"City_{city}"
    if city_column in input_data.columns:
        input_data[city_column] = 1

    input_scaled = scaler.transform(input_data)

    predicted_aqi = xgb_model.predict(input_scaled)[0]
    print(f"Predicted AQI for {city} on {date.strftime('%Y-%m-%d')}: {predicted_aqi:.2f}")

date_input = input("Enter date (YYYY-MM-DD): ")
city_input = input("Enter city name: ")
predict_aqi(date_input, city_input)
