import gradio as gr

# --- Model Functions ---

def digit_recognizer():
    return "Visit the [Digit Recognizer Web App](https://digitrecogniser-zzcjt35a9vgsmk5p235szk.streamlit.app/)."

def house_price_prediction():
    return "Visit the [House Price Prediction Web App](https://9yx5jqt7gemca4xn8mncwk.streamlit.app/)."

def nlp_disaster_tweets():
    return "Visit the [NLP Disaster Tweets Web App](https://nncsjh9jxmwixpfstqtcszbp.streamlit.app/)."

def spaceship_titanic_prediction():
    return "Visit the [Spaceship Titanic Prediction Web App](https://hrbwzuxseql4zdrvflh5y6.streamlit.app/)."

def titanic_prediction():
    return "Visit the [Titanic Prediction Web App](https://5krauxjee9gubhqhszvaam.streamlit.app/)."

# --- Unified Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# Unified ML Platform")
    gr.Markdown("### Choose a model to explore:")
    
    with gr.Row():
        digit_btn = gr.Button("Digit Recognizer")
        house_btn = gr.Button("House Price Prediction")
        nlp_btn = gr.Button("NLP Disaster Tweets")
        spaceship_btn = gr.Button("Spaceship Titanic Prediction")
        titanic_btn = gr.Button("Titanic Prediction")

    # Output display
    output = gr.Markdown()

    # Button click events
    digit_btn.click(digit_recognizer, inputs=None, outputs=output)
    house_btn.click(house_price_prediction, inputs=None, outputs=output)
    nlp_btn.click(nlp_disaster_tweets, inputs=None, outputs=output)
    spaceship_btn.click(spaceship_titanic_prediction, inputs=None, outputs=output)
    titanic_btn.click(titanic_prediction, inputs=None, outputs=output)

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch()
