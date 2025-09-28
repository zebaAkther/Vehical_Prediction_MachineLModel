import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Part 1: Load Model and Encoders with Caching ---
@st.cache_resource
def load_model():
    model = joblib.load('xgbr_model.joblib')
    encoders = joblib.load('label_encoders.joblib')
    return model, encoders

model, encoders = load_model()

# Optional debug output to check loaded encoders
st.write("Encoders loaded:", list(encoders.keys()))

# --- Part 2: App Title and User Input Features Function with Safe Encoder Access ---

st.title('🚗 Vehicle Price Prediction')
st.markdown("Enter the vehicle details in the sidebar to get an estimated selling price.")

st.sidebar.header('Vehicle Features')

def user_input_features():
    year = st.sidebar.slider('Year', 1990, 2025, 2015)

    make_classes = getattr(encoders['make'], 'classes_', None)
    if make_classes is None:
        st.error("Make encoder classes missing.")
        make_classes = ['unknown']
    make = st.sidebar.selectbox('Make', make_classes)

    model_classes = getattr(encoders['model'], 'classes_', None)
    if model_classes is None:
        st.error("Model encoder classes missing.")
        model_classes = ['unknown']
    model_input = st.sidebar.selectbox('Model', model_classes)

    trim_classes = getattr(encoders['trim'], 'classes_', None)
    if trim_classes is None:
        st.error("Trim encoder classes missing.")
        trim_classes = ['unknown']
    trim = st.sidebar.selectbox('Trim', trim_classes)

    body_classes = getattr(encoders['body'], 'classes_', None)
    if body_classes is None:
        st.error("Body encoder classes missing.")
        body_classes = ['unknown']
    body = st.sidebar.selectbox('Body Type', body_classes)

    transmission_classes = getattr(encoders['transmission'], 'classes_', None)
    if transmission_classes is None:
        st.error("Transmission encoder classes missing.")
        transmission_classes = ['unknown']
    transmission = st.sidebar.selectbox('Transmission', transmission_classes)

    state_classes = getattr(encoders['state'], 'classes_', None)
    if state_classes is None:
        st.error("State encoder classes missing.")
        state_classes = ['unknown']
    state = st.sidebar.selectbox('State', state_classes)

    condition = st.sidebar.slider('Condition (1-5)', 1.0, 5.0, 3.5, 0.1)
    odometer = st.sidebar.number_input('Odometer (miles)', min_value=0, max_value=500000, value=50000)

    color_classes = getattr(encoders['color'], 'classes_', None)
    if color_classes is None:
        st.error("Color encoder classes missing.")
        color_classes = ['unknown']
    color = st.sidebar.selectbox('Color', color_classes)

    interior_classes = getattr(encoders['interior'], 'classes_', None)
    if interior_classes is None:
        st.error("Interior encoder classes missing.")
        interior_classes = ['unknown']
    interior = st.sidebar.selectbox('Interior Color', interior_classes)

    seller_classes = getattr(encoders['seller'], 'classes_', None)
    if seller_classes is None:
        st.error("Seller encoder classes missing.")
        seller_classes = ['unknown']
    seller = st.sidebar.selectbox('Seller', seller_classes)

    mmr = st.sidebar.number_input('Manheim Market Report (MMR)', min_value=0, value=20000)

    data = {
        'year': year,
        'make': make,
        'model': model_input,
        'trim': trim,
        'body': body,
        'transmission': transmission,
        'state': state,
        'condition': condition,
        'odometer': odometer,
        'color': color,
        'interior': interior,
        'seller': seller,
        'mmr': mmr,
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Part 3: Encode User Inputs and Make Prediction ---

encoded_df = input_df.copy()

for col in encoders:
    try:
        encoded_df[col] = encoders[col].transform(encoded_df[col])
    except ValueError:
        st.error(f"Category '{encoded_df[col].iloc[0]}' in '{col}' was not seen during training. Prediction may be inaccurate.")
        encoded_df[col] = encoders[col].transform([encoders[col].classes_[0]])[0]

st.subheader('User Input Features')
st.write(input_df)

if st.button('Predict Price'):
    prediction = model.predict(encoded_df)
    price_str = f"${prediction[0]:,.2f}"
    st.subheader('Prediction')
    st.success(f'The estimated selling price of the vehicle is: **{price_str}**')
