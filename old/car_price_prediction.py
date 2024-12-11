import streamlit as st
import pandas as pd
import datetime
import xgboost as xgb

def main():
    st.title("Car Price Prediction")

    # Load the model
    model = xgb.XGBRegressor()
    try:
        model.load_model("xgb_model.json")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

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
        'Present_Price': p1,
        'Kms_Driven': p2,
        'Fuel_Type': p3,
        'Seller_Type': p4,
        'Transmission': p5,
        'Owner': p6,
        'Age': p7
    }, index=[0])

    if st.button("Predict"):
        try:
            pred = model.predict(data_new)
            st.success("You can sell your car at {:.2f} lakhs".format(pred[0]))
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == '__main__':
    main()