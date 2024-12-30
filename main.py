#Written by Rishabh Raj Pandey
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sonar_data = pd.read_csv('/content/sonar.csv', header=None)
sonar_data.head()
sonar_data.shape
sonar_data[60].value_counts()
sonar_data.groupby(60).mean()
x= sonar_data.drop(columns=60, axis=1)
y= sonar_data[60]
# Training and Test Data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)


model = LogisticRegression()  # Model Training -> Logistic Regression
model.fit(X_train, Y_train) # training the logistic regression model with training data

# Model evaluation
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Mdel Accuracy on training data: ", training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Model Accuracy on testing data: ", test_data_accuracy)

# Making a Predictive System
# random sample data of mine from the sonar data
input_data =(0.0260,0.0192,0.0254,0.0061,0.0352,0.0701,0.1263,0.1080,0.1523,0.1630,0.1030,0.2187,0.1542,0.2630,0.2940,0.2978,0.0699,0.1401,0.2990,0.3915,0.3598,0.2403,0.4208,0.5675,0.6094,0.6323,0.6549,0.7673,1.0000,0.8463,0.5509,0.4444,0.5169,0.4268,0.1802,0.0791,0.0535,0.1906,0.2561,0.2153,0.2769,0.2841,0.1733,0.0815,0.0335,0.0933,0.1018,0.0309,0.0208,0.0318,0.0132,0.0118,0.0120,0.0051,0.0070,0.0015,0.0035,0.0008,0.0044,0.0077)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 'R':
    print("It is a Rock\n")
else:
    print("\nAlert,the object is a Mine\n")

input_data = (0.0366,0.0421,0.0504,0.0250,0.0596,0.0252,0.0958,0.0991,0.1419,0.1847,0.2222,0.2648,0.2508,0.2291,0.1555,0.1863,0.2387,0.3345,0.5233,0.6684,0.7766,0.7928,0.7940,0.9129,0.9498,0.9835,1.0000,0.9471,0.8237,0.6252,0.4181,0.3209,0.2658,0.2196,0.1588,0.0561,0.0948,0.1700,0.1215,0.1282,0.0386,0.1329,0.2331,0.2468,0.1960,0.1985,0.1570,0.0921,0.0549,0.0194,0.0166,0.0132,0.0027,0.0022,0.0059,0.0016,0.0025,0.0017,0.0027,0.0027)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 'R':
    print("\nIt is a Rock\n")
else:
    print("Alert,the object is a Mine\n")
