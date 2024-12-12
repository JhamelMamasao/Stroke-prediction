import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Handle missing values
data['bmi'].fillna(data['bmi'].mean(), inplace=True)

# One-hot encoding for categorical features
data = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

# Features and target
X1 = data.drop(columns=['id', 'stroke'], axis=1)
Y1 = data['stroke']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.2, random_state=105)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, Y_train_balanced = smote.fit_resample(X_train, Y_train)

# Check the class distribution in the training set after SMOTE
print("Class distribution in Y_train (after SMOTE):")
print(Y_train_balanced.value_counts())

# Initialize Naive Bayes model
model = GaussianNB()

# Train the Naive Bayes model
model.fit(X_train_balanced, Y_train_balanced)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, y_pred)
class_report = classification_report(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)

# Calculate specificity
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(class_report)
print("Confusion Matrix:")
print(conf_matrix)
print(f"Specificity: {specificity}")


# Save the trained model
joblib.dump((model, X_train_balanced.shape[1]), 'stroke_predictor.pkl')
print("Model saved as 'stroke_predictor.pkl'.")
