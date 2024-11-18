import streamlit as st
import streamlit.components.v1 as components

# --- Main App UI ---
def main():
    st.title("Unified ML Platform")

    # Sidebar for model selection
    model = st.sidebar.selectbox(
        "Select a model to use:",
        [
            "Digit Recognizer",
            "House Price Prediction",
            "NLP Disaster Tweets",
            "Spaceship Titanic Prediction",
            "Titanic Prediction",
            "About",
            "Feedback",
            "Analytics"
        ]
    )

    # Based on the selected model, show the corresponding page
    if model == "Digit Recognizer":
        digit_recognizer()
    elif model == "House Price Prediction":
        house_price_prediction()
    elif model == "NLP Disaster Tweets":
        nlp_disaster_tweets()
    elif model == "Spaceship Titanic Prediction":
        spaceship_titanic_prediction()
    elif model == "Titanic Prediction":
        titanic_prediction()
    elif model == "About":
        about_page()
    elif model == "Feedback":
        feedback_page()
    elif model == "Analytics":
        analytics_page()

# --- Model-Specific Functions ---

def digit_recognizer():
    st.header("Digit Recognizer")
    # Embed the existing web app for digit recognizer
    components.iframe("https://digitrecogniser-zzcjt35a9vgsmk5p235szk.streamlit.app/", height=800)

def house_price_prediction():
    st.header("House Price Prediction")
    # Embed the existing web app for house price prediction
    components.iframe("https://9yx5jqt7gemca4xn8mncwk.streamlit.app/", height=800)

def nlp_disaster_tweets():
    st.header("NLP Disaster Tweets")
    # Embed the existing web app for NLP disaster tweets prediction
    components.iframe("https://nncsjh9jxmwixpfstqtcszbp.streamlit.app/", height=800)

def spaceship_titanic_prediction():
    st.header("Spaceship Titanic Prediction")
    # Embed the existing web app for spaceship titanic prediction
    components.iframe("https://hrbwzuxseql4zdrvflh5y6.streamlit.app/", height=800)

def titanic_prediction():
    st.header("Titanic Prediction")
    # Embed the existing web app for Titanic prediction
    components.iframe("https://5krauxjee9gubhqhszvaam.streamlit.app/", height=800)

# --- About Page ---
def about_page():
    st.header("About Unified ML Platform")
    st.write("""
        This platform brings together several machine learning models that demonstrate 
        various predictions including digit recognition, house price prediction, 
        natural language processing, and Titanic survival prediction. 

        Each app uses different techniques and algorithms to provide predictions 
        based on the input provided by the user. 
        Enjoy exploring the different models!
    """)

# --- Feedback Page ---
def feedback_page():
    st.header("Feedback Form")
    feedback = st.text_area("Please provide your feedback here:")
    submit_button = st.button("Submit")
    if submit_button:
        st.success("Thank you for your feedback!")

# --- Analytics Page ---
def analytics_page():
    st.header("Analytics")
    st.write("This page will show user analytics for the platform.")
    # You can implement tracking using Streamlit’s session_state or Google Analytics, etc.
    # Example: Display number of times each model was accessed, etc.

# --- Run the App ---
if __name__ == "__main__":
    main()
