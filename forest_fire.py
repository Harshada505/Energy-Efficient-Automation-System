import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("Dataset.csv")  # Ensure this is your dataset file
data = np.array(data)

# Split the dataset into features and target variable
X = data[:, :-1]  # Features: Humidity and Light
y = data[:, -1]   # Target: Temperature

# Convert the data to the appropriate type
X = X.astype('float')
y = y.astype('float')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and train the gradient boosting regressor model
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

# Evaluate the model
y_pred = gbr.predict(X_test)
accuracy = 1 - mean_squared_error(y_test, y_pred) / np.var(y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model to a file using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(gbr, file)

print("Model trained and saved as model.pkl")
