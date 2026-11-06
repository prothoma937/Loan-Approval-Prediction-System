#gradio app 

import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the Model
with open("loan_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# 2. The Logic Function
def predict_loan(gender, married, dependents, education, self_employed,
                 applicant_income, coapplicant_income, loan_amount,
                 loan_term, credit_history, property_area):

    # same feature engineering as training
    total_income = applicant_income + coapplicant_income
    emi = loan_amount / loan_term

    # Pack inputs into a DataFrame
    # The column names must match your training data exactly
    input_df = pd.DataFrame([[
        gender, married, dependents, education, self_employed,
        np.log1p(applicant_income), np.log1p(coapplicant_income),
        np.log1p(loan_amount), loan_term, float(credit_history),
        property_area, np.log1p(total_income), emi

    ]],
      columns=[
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Property_Area', 'Total_Income', 'EMI'
    ])

    # Predict
    prediction = model.predict(input_df)[0]

    # Return result
    return "Loan Approved" if prediction == 1 else "Loan Rejected"

# 3. The App Interface
inputs = [
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Radio(["Yes", "No"], label="Married"),
    gr.Radio(["0", "1", "2", "3+"], label="Dependents"),
    gr.Radio(["Graduate", "Not Graduate"], label="Education"),
    gr.Radio(["Yes", "No"], label="Self Employed"),
    gr.Number(label="Applicant Income", value=5000),
    gr.Number(label="Coapplicant Income", value=0),
    gr.Number(label="Loan Amount (in thousands)", value=130),
    gr.Number(label="Loan Term (months)", value=360),
    gr.Radio(["1", "0"], label="Credit History (1 = good, 0 = bad)"),
    gr.Radio(["Urban", "Semiurban", "Rural"], label="Property Area")
]

app = gr.Interface(
    fn=predict_loan,
      inputs=inputs,
        outputs="text",
        title="Loan Approval Predictor")

app.launch(share=True)