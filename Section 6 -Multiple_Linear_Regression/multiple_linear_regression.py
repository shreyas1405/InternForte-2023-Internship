import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
dataset = pd.read_csv('50_Startups.csv')

# Split the dataset into features (X) and target variable (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),)], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train a Multiple Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Print the predicted and actual values
np.set_printoptions(precision=2)
print("Predicted vs Actual Values:")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
