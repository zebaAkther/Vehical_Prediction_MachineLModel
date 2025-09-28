import streamlit as st
import pandas as pd
import joblib

# Load the model and encoders with caching to avoid reload on rerun
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load('xgbr_model.joblib')
    encoders = joblib.load('label_encoders.joblib')
    return model, encoders

model, encoders = load_model_and_encoders()

st.title('🚗 Vehicle Price Prediction')
st.markdown("Enter the vehicle details in the sidebar to get an estimated selling price.")
st.sidebar.header('Vehicle Features')

def safe_get_classes(encoders, key):
    classes = getattr(encoders.get(key, None), 'classes_', None)
    if classes is None:
        st.error(f"Encoder classes not found for '{key}', defaulting to ['unknown'].")
        return ['unknown']
    return classes

def user_input_features():
    year = st.sidebar.slider('Year', 1990, 2025, 2015)
    make = st.sidebar.selectbox('Make', safe_get_classes(encoders, 'make'))
    model_input = st.sidebar.selectbox('Model', safe_get_classes(encoders, 'model'))
    trim = st.sidebar.selectbox('Trim', safe_get_classes(encoders, 'trim'))
    body = st.sidebar.selectbox('Body Type', safe_get_classes(encoders, 'body'))
    transmission = st.sidebar.selectbox('Transmission', safe_get_classes(encoders, 'transmission'))
    state = st.sidebar.selectbox('State', safe_get_classes(encoders, 'state'))
    condition = st.sidebar.slider('Condition (1-5)', 1.0, 5.0, 3.5, 0.1)
    odometer = st.sidebar.number_input('Odometer (miles)', min_value=0, max_value=500000, value=50000)
    color = st.sidebar.selectbox('Color', safe_get_classes(encoders, 'color'))
    interior = st.sidebar.selectbox('Interior Color', safe_get_classes(encoders, 'interior'))
    seller = st.sidebar.selectbox('Seller', safe_get_classes(encoders, 'seller'))
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
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Encode the inputs safely
encoded_df = input_df.copy()

for col in encoders:
    try:
        encoded_df[col] = encoders[col].transform(encoded_df[col])
    except (ValueError, KeyError):
        st.error(f"Value '{encoded_df[col].iloc[0]}' not recognized for feature '{col}'. Using default encoder value.")
        encoded_df[col] = encoders[col].transform([encoders[col].classes_[0]])[0]

st.subheader('User Input Features')
st.write(input_df)

if st.button('Predict Price'):
    prediction = model.predict(encoded_df)
    price_str = f"${prediction[0]:,.2f}"
    st.subheader('Prediction')
    st.success(f'The estimated selling price of the vehicle is: **{price_str}**')
