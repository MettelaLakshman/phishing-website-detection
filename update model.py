import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, plot_confusion_matrix
from sklearn.ensemble import VotingClassifier


# Load data
data0 = pd.read_csv('data.csv')

# Drop Domain column
data = data0.drop(['Domain'], axis=1).copy()
# Check for missing values
data.isnull().sum()
import warnings
warnings.filterwarnings("ignore")
# Shuffle rows in dataset
data = data.sample(frac=1).reset_index(drop=True)

# Split dataset into features and target
y = data['Label']
X = data.drop('Label', axis=1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
forest = RandomForestClassifier(random_state=42)
tree = DecisionTreeClassifier(random_state=42)
logreg = LogisticRegression(random_state=42)

# Train the models
forest.fit(X_train, y_train)
tree.fit(X_train, y_train)
logreg.fit(X_train, y_train)

# Create Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('Random Forest', forest), ('Decision Tree', tree), ('Logistic Regression', logreg)], voting='hard')
voting_clf.fit(X_train, y_train)
# Model evaluation
models = [forest, tree, logreg, voting_clf]
model_names = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Voting Classifier']

for model, name in zip(models, model_names):
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name} Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    cr = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(cr)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    plot_confusion_matrix(model, X_test, y_test, display_labels=['Benign', 'Phishing'], cmap=plt.cm.Blues,
                          normalize=None)
    plt.title(f'{name} Confusion Matrix')
    plt.show()
