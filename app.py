import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

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
    page_background_color='#00a2ff',  # Blue
    page_text_color='#ffffff',  # White
)

# Load the pre-trained model
model = pk.load(open('model.pkl', 'rb'))

# Header of the Web App
st.title('Car Price Prediction ML Model', anchor=None)

# Load the car details CSV
cars_data = pd.read_csv('Cars.csv')

# Function to extract the car brand from the name
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Input fields for the user to interact with
col1, col2 = st.columns(2)

with col1:
    name = st.selectbox('Select Car Brand', cars_data['name'].unique(), key='name')
    fuel = st.selectbox('Fuel type', cars_data['fuel'].unique(), key='fuel')
    transmission = st.selectbox('Transmission type', cars_data['transmission'].unique(), key='transmission')
    owner = st.selectbox('Seller type', cars_data['owner'].unique(), key='owner')

with col2:
    year = st.number_input('Car Manufactured Year', min_value=1994, max_value=2024, step=1, key='year')
    km_driven = st.number_input('No of kms Driven', min_value=11, max_value=200000, step=1, key='km_driven')
    seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique(), key='seller_type')
    mileage = st.number_input('Car Mileage', min_value=10, max_value=40, step=1, key='mileage')
    engine = st.number_input('Engine CC', min_value=700, max_value=5000, step=1, key='engine')
    max_power = st.number_input('Max Power', min_value=0, max_value=200, step=1, key='max_power')
    seats = st.number_input('No of Seats', min_value=2, max_value=10, step=1, key='seats')

# Predict button
if st.button("Predict"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
                                      'Fourth & Above Owner', 'Test Drive Car'],
                                     [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                     'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                     'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                     'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                     'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                     20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], inplace=True)

    try:
        car_price = model.predict(input_data_model)
        st.success(f'Estimated Car Price: ₹ {car_price[0]:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
