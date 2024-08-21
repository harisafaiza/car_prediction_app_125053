import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Load the pre-trained model
model = pk.load(open('model.pkl', 'rb'))

# Inspect the model's attributes
print(f"Model features: {model.feature_names_in_}")
print(f"Model coefficients: {model.coef_}")

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
year = st.number_input('Car Manufactured Year', min_value=1994, max_value=2024, step=1)
km_driven = st.number_input('No of kms Driven', min_value=11, max_value=200000, step=1)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())

# Prepare the input data
input_data_model = pd.DataFrame(
    [[name, year, km_driven, fuel, seller_type, transmission, owner]],
    columns=model.feature_names_in_
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

# Perform prediction
if st.button("Predict"):
    try:
        car_price = model.predict(input_data_model)
        st.success(f'Estimated Car Price: ₹ {car_price[0]:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
