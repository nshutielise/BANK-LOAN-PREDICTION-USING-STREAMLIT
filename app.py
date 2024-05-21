import streamlit as st
import pickle
import pandas as pd

def process_form(gender, married, dependents, education, self_employed,
                 applicant_income, coapplicant_income, loan_amount,
                 loan_amount_term, credit_history, property_area):
    # Convert categorical variables to numeric
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    dependents = 3 if dependents == "3+" else int(dependents)
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    # Map property area string values to numeric representations
    property_area_mapping = {"Urban": 2, "Rural": 0, "Semiurban": 1}
    property_area_numeric = property_area_mapping[property_area]
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area_numeric]  # Use numeric representation
    })

    # Load the trained model
    model_path = '/content/logistic_regression_model.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display result
    if prediction[0] == 1:
        st.success(f"Loan Approved with a probability of {prediction_proba[0][1]:.2f}")
    else:
        st.error(f"Loan Not Approved with a probability of {prediction_proba[0][0]:.2f}")

def display_form():
    st.subheader("Enter Your Details")

    with st.form(key='loan_form'):
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        applicant_income = st.number_input("Applicant Income", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0)
        loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
        credit_history = st.selectbox("Credit History", [0, 1])
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        process_form(gender, married, dependents, education, self_employed,
                     applicant_income, coapplicant_income, loan_amount,
                     loan_amount_term, credit_history, property_area)

if __name__ == '__main__':
    display_form()

