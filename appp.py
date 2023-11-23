import pandas as pd
import streamlit as st
import numpy as np
import math
import pickle

#import the model
#pipe = pickle.load(open('pipe.pkl','rb'))
#df = pickle.load(open('df.pkl', 'rb'))


pipe = pd.read_pickle('pipe.pkl')
df = pd.read_pickle('df.pkl')
def predict_price(company, type, ram, weight, touchscreen, ips, screen_size, cpu, hdd, ssd, gpu, os):
    # Predict the price for the given laptop features
    predicted_price = pipe.predict([[company, type, ram, weight, touchscreen, ips, screen_size, cpu, hdd, ssd, gpu, os]])
    return predicted_price[0]

# Set page title and layout
# st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

# Add title and description to the app
st.title("Laptop Price Predictor")
st.markdown("Enter the laptop features below to predict its price.")

# Create input fields for laptop features
#brand
company = st.selectbox('Brand', df['Company'].unique())


#type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

#Ram
ram = st.selectbox('RAM(in GB)',[2, 4, 8, 16, 24, 32, 64])


#Weight
weight = st.number_input('Weight of the laptop')

#touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

#ips
ips = st.selectbox('IPS', ['No', 'Yes'])

#screen size
screen_size = st.number_input('Screen Size')

#resolution
resolution = st.selectbox('Select Resolution', ['1920x1080', '1600x900', '3840x2160', '3200x1800','2880x1800', '2560x1600', '2560x1440', '2304x1440'])

#cpu
cpu = st.selectbox('Brand', df['Cpu brand'].unique())


#hdd
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD(in GB)', [0,8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', df['Gpu brand'].unique())

os = st.selectbox('OS', df['os'].unique())

touchscreen = 1 if touchscreen == 'Yes' else 0
ips = 1 if ips == 'Yes' else 0

# Convert resolution to numeric values (assuming it's in the format 'XxY')
X_res = int(resolution.split('x')[0])
Y_res = int(resolution.split('x')[1])

# Predict button
if st.button("Predict"):
    try:
        # Call predict_price function with user inputs
        predicted_price = predict_price(company, type, ram, weight, touchscreen, ips, screen_size, cpu, hdd, ssd, gpu, os)
        final_price = 83 * predicted_price
        inr = float(final_price * 83);
        st.success(f"The predicted price is: ${inr:.2f}")

    except ValueError as ve:
        st.error(f"ValueError: {ve}. Please enter valid input.")
    except Exception as e:
        st.error(f"An error occurred: {e}")



# Display dataset if user wants to see it
if st.checkbox("Show Dataset"):
    st.dataframe(df)