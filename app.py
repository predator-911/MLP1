import os
import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import requests

# --- Utility Functions ---
def download_file(url, local_path):
    """Download a file from a URL if it doesn't already exist locally."""
    if not os.path.exists(local_path):
        st.write(f"Downloading {os.path.basename(local_path)}...")
        response = requests.get(url)
        with open(local_path, "wb") as f:
            f.write(response.content)

# --- Model Loader Functions ---
@st.cache_resource
def load_digit_recognizer_model():
    url = "https://raw.githubusercontent.com/predator-911/DigitRecogniser/main/model.h5"
    local_path = "digit_model.h5"
    download_file(url, local_path)
    return tf.keras.models.load_model(local_path)

@st.cache_resource
def load_nlp_model():
    url = "https://raw.githubusercontent.com/predator-911/NLP/main/nlp_model_pipeline.pkl"
    local_path = "nlp_model_pipeline.pkl"
    download_file(url, local_path)
    return joblib.load(local_path)

@st.cache_resource
def load_house_price_model():
    model_url = "https://raw.githubusercontent.com/predator-911/HousePrice/main/random_forest_model.h5"
    scaler_url = "https://raw.githubusercontent.com/predator-911/HousePrice/main/scaler.pkl"
    columns_url = "https://raw.githubusercontent.com/predator-911/HousePrice/main/train_columns.pkl"
    
    model_path = "house_price_model.h5"
    scaler_path = "scaler.pkl"
    columns_path = "train_columns.pkl"
    
    download_file(model_url, model_path)
    download_file(scaler_url, scaler_path)
    download_file(columns_url, columns_path)
    
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    columns = joblib.load(columns_path)
    
    return model, scaler, columns

@st.cache_resource
def load_store_sales_model():
    url = "https://raw.githubusercontent.com/predator-911/StoreSales/main/xgb_model_current_version.pkl"
    local_path = "store_sales_model.pkl"
    download_file(url, local_path)
    return joblib.load(local_path)

@st.cache_resource
def load_spaceship_model():
    url = "https://raw.githubusercontent.com/predator-911/Spaceship/main/spaceship_titanic_model.pkl"
    local_path = "spaceship_model.pkl"
    download_file(url, local_path)
    return joblib.load(local_path)

@st.cache_resource
def load_titanic_model():
    url = "https://raw.githubusercontent.com/predator-911/Titanic/main/titanic_model.pkl"
    local_path = "titanic_model.pkl"
    download_file(url, local_path)
    return joblib.load(local_path)

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

# --- Model-Specific Functions ---
def digit_recognizer():
    model = load_digit_recognizer_model()
    st.header("Digit Recognizer")
    image = st.file_uploader("Upload an image of a digit (28x28 pixels)", type=["png", "jpg", "jpeg"])
    
    if image is not None:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Predicting...")
        # Placeholder preprocessing logic
        # prediction = model.predict(processed_image)
        st.write("Predicted Digit: (Placeholder)")

def nlp_disaster_tweets():
    model = load_nlp_model()
    st.header("NLP Disaster Tweets")
    tweet = st.text_area("Enter a tweet to classify")
    
    if tweet:
        st.write("Predicting...")
        prediction = model.predict([tweet])
        st.write("Disaster!" if prediction[0] == 1 else "Not a disaster.")

def house_price_prediction():
    model, scaler, columns = load_house_price_model()
    st.header("House Price Prediction")
    sqft = st.number_input("Square Feet", min_value=1)
    bedrooms = st.number_input("Bedrooms", min_value=1)
    bathrooms = st.number_input("Bathrooms", min_value=1)
    features = [sqft, bedrooms, bathrooms]
    
    if st.button("Predict Price"):
        st.write("Predicting...")
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)
        st.write(f"Predicted House Price: ${prediction[0]:,.2f}")

def store_sales_prediction():
    model = load_store_sales_model()
    st.header("Store Sales Prediction")
    store_type = st.selectbox("Store Type", ["Type A", "Type B", "Type C"])
    size = st.number_input("Store Size (in sq ft)", min_value=1)
    
    if st.button("Predict Sales"):
        st.write("Predicting...")
        prediction = model.predict([[store_type, size]])
        st.write(f"Predicted Store Sales: ${prediction[0]:,.2f}")

def spaceship_titanic_prediction():
    model = load_spaceship_model()
    st.header("Spaceship Titanic Prediction")
    passenger_class = st.selectbox("Passenger Class", ["1", "2", "3"])
    age = st.number_input("Age", min_value=1)
    
    if st.button("Predict"):
        st.write("Predicting...")
        prediction = model.predict([[passenger_class, age]])
        st.write(f"Prediction: {prediction[0]}")

def titanic_prediction():
    model = load_titanic_model()
    st.header("Titanic Prediction")
    pclass = st.selectbox("Pclass", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=1)
    
    if st.button("Predict Survival"):
        st.write("Predicting...")
        prediction = model.predict([[pclass, sex, age]])
        st.write(f"Survival Prediction: {prediction[0]}")

# --- Run the App ---
if __name__ == "__main__":
    main()
