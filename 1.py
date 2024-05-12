import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Title and image
st.title('Credit Card Fraud Detection')
st.image('IMAGE.png', use_column_width=True)  # Adjust 'your_image.png' with the path to your image file

# Read Data into a Dataframe
df = pd.read_csv('creditcard.csv')

# Prepare the data
X = df.drop('Class', axis=1)  # Assuming 'Class' is the target variable
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Manual transaction verification
if st.sidebar.checkbox('Manual transaction verification'):

    def verify_transaction(features):
        # Use the trained model to predict
        scaled_features = scaler.transform([features])  # Scale the input features
        prediction = model.predict(scaled_features)
        return "Fraudulent" if prediction[0] == 1 else "Legitimate"

    st.subheader("Manual Transaction Verification")
    st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

    example_features = st.text_input('Input All features (comma-separated)')

    if st.button("Submit"):
        try:
            example_features = list(map(float, example_features.split(',')))
            result = verify_transaction(example_features)
            st.success("Example verification result: " + result)

            # Evaluate the model after the user has entered input and received output
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Display the evaluation metrics
            st.write("Model Performance:")
            st.write("Training Accuracy:", train_accuracy)
            st.write("Testing Accuracy:", test_accuracy)

        except Exception as e:
            st.error("Error processing input: {}".format(e))
            st.write("Please ensure all inputs are numerical and properly formatted.")

