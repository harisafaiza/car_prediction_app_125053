import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Load the pre-trained model
model = pk.load(open('model.pkl', 'rb'))

# Header of the Web App
st.header('Car Price Prediction ML Model')

# Load the car details CSV
cars_data = pd.read_csv('Cars.csv')

# Function to extract the car brand from the name
def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

# Apply the function to extract brand names from the 'name' column
cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Input fields for the user to interact with
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage (km/l)', 10, 40)
engine = st.slider('Engine Capacity (CC)', 700, 5000)
max_power = st.slider('Max Power (bhp)', 0, 200)
seats = st.slider('Number of Seats', 2, 10)

# Predict button
if st.button("Predict"):
    # Data for prediction
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    # Encode categorical variables
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
                                       'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                      'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                      'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                      'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                      'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                                     range(1, 32), inplace=True)

    try:
        # Perform prediction
        car_price = model.predict(input_data_model)
        st.success(f'Estimated Car Price: ₹ {car_price[0]:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
