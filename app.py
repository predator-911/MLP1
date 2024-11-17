import os
import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

# --- Load Models ---
# Function to load the digit recognizer model
@st.cache_resource
def load_digit_recognizer_model():
    model_path = r'D:\hardhat\KAGGLE\DigitRecogniser\model.h5'
    return tf.keras.models.load_model(model_path)

# Function to load the NLP model
@st.cache_resource
def load_nlp_model():
    model_path = r'D:\hardhat\KAGGLE\NLPDisasterTweets\nlp_model_pipeline.pkl'
    return joblib.load(model_path)

# Function to load the house price prediction models
@st.cache_resource
def load_house_price_model():
    model_path = r'D:\hardhat\KAGGLE\HousePrice\random_forest_model.h5'
    model = tf.keras.models.load_model(model_path)
    
    scaler_path = r'D:\hardhat\KAGGLE\HousePrice\scaler.pkl'
    scaler = joblib.load(scaler_path)
    
    columns_path = r'D:\hardhat\KAGGLE\HousePrice\train_columns.pkl'
    columns = joblib.load(columns_path)
    
    return model, scaler, columns

# Function to load the store sales model
@st.cache_resource
def load_store_sales_model():
    model_path = r'D:\hardhat\KAGGLE\StoreSales\xgb_model_current_version.pkl'
    return joblib.load(model_path)

# Function to load the spaceship titanic model
@st.cache_resource
def load_spaceship_model():
    model_path = r'D:\hardhat\KAGGLE\SPACESHIP\spaceship_titanic_model.pkl'
    return joblib.load(model_path)

# Function to load the titanic model
@st.cache_resource
def load_titanic_model():
    model_path = r'D:\hardhat\KAGGLE\TITANIC\titanic_model.pkl'
    return joblib.load(model_path)

# --- Main App UI ---
def main():
    st.title("Unified ML Platform")
    
    # Sidebar for model selection
    model = st.sidebar.selectbox(
        "Select a model to use:",
        ["Digit Recognizer", "NLP Disaster Tweets", "House Price Prediction", 
         "Store Sales Prediction", "Spaceship Titanic Prediction", "Titanic Prediction"]
    )

    if model == "Digit Recognizer":
        digit_recognizer()

    elif model == "NLP Disaster Tweets":
        nlp_disaster_tweets()

    elif model == "House Price Prediction":
        house_price_prediction()

    elif model == "Store Sales Prediction":
        store_sales_prediction()

    elif model == "Spaceship Titanic Prediction":
        spaceship_titanic_prediction()

    elif model == "Titanic Prediction":
        titanic_prediction()

# --- Digit Recognizer Function ---
def digit_recognizer():
    model = load_digit_recognizer_model()

    st.header("Digit Recognizer")
    image = st.file_uploader("Upload an image of a digit (28x28 pixels)", type=["png", "jpg", "jpeg"])
    
    if image is not None:
        # Process the image and make prediction
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Predicting...")
        
        # Preprocess and predict logic for digit
        # Example processing:
        # image = process_image(image)  # Example function to process image
        prediction = model.predict(image)
        st.write(f"Predicted Digit: {prediction}")

# --- NLP Disaster Tweets Function ---
def nlp_disaster_tweets():
    model = load_nlp_model()

    st.header("NLP Disaster Tweets")
    tweet = st.text_area("Enter a tweet to classify")
    
    if tweet:
        st.write("Predicting...")
        prediction = model.predict([tweet])
        if prediction[0] == 1:
            st.write("Disaster!")
        else:
            st.write("Not a disaster.")

# --- House Price Prediction Function ---
def house_price_prediction():
    model, scaler, columns = load_house_price_model()

    st.header("House Price Prediction")
    
    # Input fields for house features
    sqft = st.number_input("Square Feet", min_value=1)
    bedrooms = st.number_input("Bedrooms", min_value=1)
    bathrooms = st.number_input("Bathrooms", min_value=1)
    # Add all necessary inputs here for the model
    features = [sqft, bedrooms, bathrooms]
    
    if st.button("Predict Price"):
        st.write("Predicting...")
        # Preprocess input data and make prediction
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)  # Scaling input features
        prediction = model.predict(features)
        st.write(f"Predicted House Price: ${prediction[0]:,.2f}")

# --- Store Sales Prediction Function ---
def store_sales_prediction():
    model = load_store_sales_model()

    st.header("Store Sales Prediction")
    
    # Input fields for store sales features
    store_type = st.selectbox("Store Type", ["Type A", "Type B", "Type C"])
    size = st.number_input("Store Size (in sq ft)", min_value=1)
    # Add any other features as required
    
    if st.button("Predict Sales"):
        st.write("Predicting...")
        features = [store_type, size]  # Example features
        prediction = model.predict([features])
        st.write(f"Predicted Store Sales: ${prediction[0]:,.2f}")

# --- Spaceship Titanic Prediction Function ---
def spaceship_titanic_prediction():
    model = load_spaceship_model()

    st.header("Spaceship Titanic Prediction")
    
    # Input fields for spaceship passenger features
    passenger_class = st.selectbox("Passenger Class", ["1", "2", "3"])
    age = st.number_input("Age", min_value=1)
    # Add other features as required
    
    if st.button("Predict"):
        st.write("Predicting...")
        features = [passenger_class, age]  # Example features
        prediction = model.predict([features])
        st.write(f"Prediction: {prediction[0]}")

# --- Titanic Prediction Function ---
def titanic_prediction():
    model = load_titanic_model()

    st.header("Titanic Prediction")
    
    # Input fields for Titanic passenger features
    pclass = st.selectbox("Pclass", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=1)
    # Add other features as required
    
    if st.button("Predict Survival"):
        st.write("Predicting...")
        features = [pclass, sex, age]  # Example features
        prediction = model.predict([features])
        st.write(f"Survival Prediction: {prediction[0]}")

# Run the app
if __name__ == "__main__":
    main()
