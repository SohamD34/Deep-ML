import math
import numpy as np
PI = 3.14159

class LinearRegression:
    def fit(self, X, y):
        self.coefficients = np.polyfit(X, y, 1)

    def predict(self, X):
        return np.polyval(self.coefficients, X)


def power_grid_forecast(consumption_data):
    # 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
    # 2) Perform linear regression on the detrended data.
    # 3) Predict day 15's base consumption.
    # 4) Add the day 15 fluctuation back.
    # 5) Round, then add a 5% safety margin (rounded up).
    # 6) Return the final integer.

    daily_fluctuations = []
    n = len(consumption_data)

    for i in range(1, n+1):
        fluc = 10 * math.sin(2 * PI * i / 10)
        daily_fluctuations.append(fluc)
    
    detrended_data = [consumption_data[i] - daily_fluctuations[i] for i in range(n)]

    linear_model = LinearRegression()
    X = [i for i in range(1, n+1)]
    linear_model.fit(np.array(X), detrended_data)

    forecast = linear_model.predict([[15]]).item(0)
    fluc = 10 * math.sin(2 * PI * 15 / 10)

    prediction = math.ceil(round(forecast + fluc) * 1.05)

    return prediction


print(power_grid_forecast([150, 165, 185, 195, 210, 225, 240, 260, 275, 290]))
