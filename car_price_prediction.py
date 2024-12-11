import streamlit as st
import pandas as pd
import datetime
import pickle

def main():
    st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")
    st.title("Car Price Prediction")

    st.markdown("""
    Welcome to the Car Price Prediction app! Please enter the details of the car below to get an estimated selling price.
    """)

    # Load the model
    try:
        with open("car_price_model.pkl", "rb") as file:
            model = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Input fields
    st.header("Car Details")
    p1 = st.number_input("Present Price (in lakhs)", 0.0, 50.0, step=0.5)
    p2 = st.number_input("Kms Driven", 0, 500000, step=1000)
    p3 = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    p4 = st.selectbox("Seller Type", ["Dealer", "Individual"])
    p5 = st.selectbox("Transmission", ["Manual", "Automatic"])
    p6 = st.slider("How many owners", 0, 3)

    date_time = datetime.datetime.now()
    years = st.number_input("Car purchased year", 1990, date_time.year, step=1)
    p7 = date_time.year - years

    data_new = pd.DataFrame({
        'Year': years,
        'Present_Price': p1,
        'Kms_Driven': p2,
        'Fuel_Type': p3,
        'Seller_Type': p4,
        'Transmission': p5,
        'Owner': p6
    }, index=[0])

    if st.button("Predict"):
        try:
            pred = model.predict(data_new)
            st.success(f"You can sell your car at ₹{pred[0]:.2f} lakhs")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

    st.markdown("""
    ---
    **Note:** This prediction is based on the data you provided and the model's accuracy. Actual selling prices may vary.
    """)

if __name__ == '__main__':
    main()