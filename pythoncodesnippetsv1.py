import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Iris dataset
def load_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df, iris.feature_names, iris.target_names

# Train the SVM model
def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Define Streamlit application
def main():
    st.title('Iris Flower Classification')
    
    st.write("## Load Data")
    df, feature_names, target_names = load_data()
    st.write(df.head())

    st.write("## Train Model")
    X = df[feature_names].values
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, scaler = train_model(X_train, y_train)

    st.write("## Model Accuracy")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model accuracy: {accuracy:.2f}")

    st.write("## Make Predictions")
    sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
    sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
    petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.5)
    petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_features_scaled = scaler.transform(input_features)
    prediction = model.predict(input_features_scaled)
    prediction_proba = model.predict_proba(input_features_scaled)

    st.write(f"Prediction: {target_names[prediction[0]]}")
    st.write(f"Prediction Probabilities: {prediction_proba}")

if __name__ == "__main__":
    main()
