import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from PIL import Image

# Set the page config
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon=":car:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    },
    page_background_color='#f44336',  # Red
    page_text_color='#ffffff',  # White
)

# Load the car details CSV
cars_data = pd.read_csv('Cars.csv')

# Function to extract the car brand from the name
def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

# Apply the function to extract brand names from the 'name' column
cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Load the pre-trained model
model = pk.load(open('model.pkl', 'rb'))

# Header of the Web App
st.title('Car Price Prediction ML Model', anchor=None)

# Input fields for the user to interact with
col1, col2 = st.columns(2)

with col1:
    name = st.selectbox('Select Car Brand', cars_data['name'].unique(), key='name')
    fuel = st.selectbox('Fuel type', cars_data['fuel'].unique(), key='fuel')
    transmission = st.selectbox('Transmission type', cars_data['transmission'].unique(), key='transmission')
    owner = st.selectbox('Owner type', cars_data['owner'].unique(), key='owner')

with col2:
    year = st.number_input('Car Manufactured Year', min_value=1994, max_value=2024, step=1, key='year')
    km_driven = st.number_input('No of kms Driven', min_value=11, max_value=200000, step=1, key='km_driven')
    seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique(), key='seller_type')
    mileage = st.number_input('Car Mileage (km/l)', min_value=10, max_value=40, step=1, key='mileage')
    engine = st.number_input('Engine Capacity (CC)', min_value=700, max_value=5000, step=1, key='engine')
    max_power = st.number_input('Max Power (bhp)', min_value=0, max_value=200, step=1, key='max_power')
    seats = st.number_input('Number of Seats', min_value=2, max_value=10, step=1, key='seats')

# Predict button
if st.button("Predict"):
    # Data for prediction
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    # Encode categorical variables
    input_data_model['owner'] = input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
                                                                   'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5])
    input_data_model['fuel'] = input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4])
    input_data_model['seller_type'] = input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3])
    input_data_model['transmission'] = input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2])
    input_data_model['name'] = input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                                                 'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                                                 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                                                 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                                                 'Ambassador', 'Ashok', 'Isuzu', 'Opel'], range(1, 32))

    try:
        # Perform prediction
        car_price = model.predict(input_data_model)
        st.success(f'Estimated Car Price: ₹ {car_price[0]:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
