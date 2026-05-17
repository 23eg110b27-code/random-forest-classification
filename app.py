import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title
st.title("Random Forest Classification App")

# Upload CSV File
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Read dataset
    data = pd.read_csv(uploaded_file)

    # Show dataset
    st.subheader("Dataset")
    st.dataframe(data.head())

    # Select target column
    target = st.selectbox("Select Target Column", data.columns)

    # Features and target
    X = data.drop(columns=[target])
    y = data[target]

    # Convert categorical columns to numeric
    X = pd.get_dummies(X)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model Accuracy")
    st.success(f"Accuracy: {accuracy:.2f}")

    # Feature Importance
    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(importance_df)

else:
    st.info("Please upload a CSV file.")