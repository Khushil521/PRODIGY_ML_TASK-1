import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data
# Create a DataFrame with house features and prices
data = {
    'SquareFootage': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    'Bedrooms': [3, 3, 3, 4, 4, 4, 5, 5, 5, 5],
    'Bathrooms': [2, 2, 2, 2.5, 2.5, 2.5, 3, 3, 3, 3],
    'Price': [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000]
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Predicting the price of a new house
new_house = np.array([[2500, 4, 3]])
predicted_price = model.predict(new_house)
print("Predicted price for the new house:", predicted_price[0])
