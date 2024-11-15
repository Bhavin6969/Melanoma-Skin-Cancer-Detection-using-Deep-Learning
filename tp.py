import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib  # Import joblib to save the model

# Load the dataset
df = pd.read_excel("C:/Users/Asus/Desktop/Minor Project/Final/Melanoma/cnn.xlsx")
X = df.drop(['target'], axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape, X_test.shape

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)  # Increase max_iter if necessary
log_reg.fit(X_train, y_train)

# Save the model
model_filename = "C:/Users/Asus/Desktop/Minor Project/Final/Melanoma/log_reg_model.joblib"
joblib.dump(log_reg, model_filename)
print(f"Model saved to {model_filename}")

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
print('Training set score: {:.4f}'.format(log_reg.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(log_reg.score(X_test, y_test)))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0, 0])
print('\nTrue Negatives(TN) = ', cm[1, 1])
print('\nFalse Positives(FP) = ', cm[0, 1])
print('\nFalse Negatives(FN) = ', cm[1, 0])

# Classification report
print(classification_report(y_test, y_pred))

# To load the model later
# loaded_model = joblib.load(model_filename)
# loaded_model.predict(X_test)  # You can use this to make predictions with the loaded model
