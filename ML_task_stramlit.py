# Import necessary libraries
import streamlit as st
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Fetch dataset
glass_identification = fetch_ucirepo(id=42)
X = glass_identification.data.features
y = glass_identification.data.targets

# Sidebar to select algorithm and parameters
algorithm = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "SVM", "KNN"])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Main content
st.title("Glass Identification Streamlit App")

# Display dataset information
st.subheader("Dataset Information:")
st.write(glass_identification.metadata)

# Display variable information
st.subheader("Variable Information:")
st.write(glass_identification.variables)

# Display training and test set shapes
st.subheader("Data Shapes:")
st.write("Training data shape - X:", X_train.shape, " y:", y_train.shape)
st.write("Test data shape - X:", X_test.shape, " y:", y_test.shape)

# Train and evaluate selected algorithm
if algorithm == "Random Forest":
    st.subheader("Random Forest Classifier")
    n_estimators = st.slider("Number of Estimators", 1, 100, 10)
    clf = RandomForestClassifier(criterion="entropy", n_estimators=n_estimators)
elif algorithm == "SVM":
    st.subheader("Support Vector Machine")
    C_value = st.slider("C (Regularization Parameter)", 0.1, 10.0, 1.0)
    clf = SVC(kernel='linear', C=C_value)
elif algorithm == "KNN":
    st.subheader("K-Nearest Neighbors")
    n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)

# Train the selected classifier
clf.fit(X_train, y_train)

# Make predictions on the test data
predictions = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
st.subheader("Model Accuracy:")
st.write(f"{algorithm} Accuracy: {accuracy:.2%}")

# Display confusion matrix
st.subheader("Confusion Matrix:")
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels="Names", yticklabels="Type of Glass")
plt.show()