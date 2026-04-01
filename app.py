import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
model = pickle.load(open('model/model.pkl','rb'))

st.set_page_config(page_title="Loan Prediction", layout="wide")

# ==============================
# 🧠 SESSION STATE (PAGE CONTROL)
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "main"

if "result_df" not in st.session_state:
    st.session_state.result_df = None

# ==============================
# 📊 RESULT PAGE
# ==============================
if st.session_state.page == "result":

    st.title("📊 Prediction Results")

    df = st.session_state.result_df

    if df is not None:
        st.success("✅ Predictions Generated Successfully!")
        st.dataframe(df)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Results",
            csv,
            "loan_predictions.csv",
            "text/csv"
        )

    if st.button("⬅️ Back to Main Page"):
        st.session_state.page = "main"

    st.stop()

# ==============================
# 🏦 MAIN PAGE
# ==============================

st.title("🏦 Loan Default Prediction System")
st.markdown("### Predict loan approval using ML (Individual + Bulk CSV)")

# Tabs
tab1, tab2 = st.tabs(["🔹 Single Prediction", "📂 Bulk CSV Prediction"])

# ==============================
# 🔹 TAB 1: SINGLE PREDICTION
# ==============================
with tab1:
    st.subheader("Enter Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male","Female"])
        married = st.selectbox("Married", ["Yes","No"])
        dependents = st.selectbox("Dependents", ["0","1","2","3+"])

    with col2:
        education = st.selectbox("Education", ["Graduate","Not Graduate"])
        self_emp = st.selectbox("Self Employed", ["Yes","No"])
        credit = st.selectbox("Credit History", [1,0])

    with col3:
        area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])
        term = st.number_input("Loan Term", value=360)
        loan_amt = st.number_input("Loan Amount")

    col4, col5 = st.columns(2)

    with col4:
        income = st.number_input("Applicant Income")

    with col5:
        co_income = st.number_input("Coapplicant Income")

    # Encoding
    dependents = 3 if dependents=='3+' else int(dependents)
    gender = 1 if gender=='Male' else 0
    married = 1 if married=='Yes' else 0
    education = 1 if education=='Graduate' else 0
    self_emp = 1 if self_emp=='Yes' else 0
    area = {'Urban':2,'Semiurban':1,'Rural':0}[area]

    # Feature Engineering
    total_income = income + co_income
    emi = loan_amt / term if term != 0 else 0

    features = np.array([[gender, married, dependents, education, self_emp,
                          income, co_income, loan_amt, term, credit,
                          area, total_income, emi]])

    if st.button("🔍 Predict Loan Status"):
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        st.markdown("---")

        if pred == 1:
            st.success(f"✅ Loan Approved (Confidence: {prob:.2f})")
        else:
            st.error(f"❌ Loan Rejected (Risk Score: {1-prob:.2f})")

# # ==============================
# 📂 TAB 2: CSV BULK PREDICTION
# ==============================
with tab2:
    st.subheader("Upload CSV File for Bulk Prediction")

    # =============================
    # STORE FILE IN SESSION
    # =============================
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None

    file = st.file_uploader("Upload CSV", type=["csv"])

    # Save uploaded file
    if file is not None:
        st.session_state.uploaded_df = pd.read_csv(file)

    # Use stored dataframe
    if st.session_state.uploaded_df is not None:

        df = st.session_state.uploaded_df.copy()

        st.write("📊 Uploaded Data Preview:")
        st.dataframe(df.head(10))

        # =============================
        # 🗑️ REMOVE BUTTON
        # =============================
        if st.button("🗑️ Remove Dataset"):
            st.session_state.uploaded_df = None
            st.rerun()

        # =============================
        # 🔍 MISSING VALUES (PRO UI)
        # =============================
        st.markdown("### 🔍 Missing Values Analysis")

        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ['Column Name', 'Missing Values']
        missing_df = missing_df[missing_df['Missing Values'] > 0]

        if missing_df.empty:
            st.success("✅ No missing values found in the dataset!")
        else:
            missing_df['Percentage (%)'] = (
                missing_df['Missing Values'] / len(df) * 100
            ).round(2)

            missing_df = missing_df.sort_values(by='Missing Values', ascending=False)

            st.dataframe(missing_df, use_container_width=True)

        # =============================
        # SAFE PREPROCESSING
        # =============================
        df.fillna({
            'Gender': 'Male',
            'Married': 'Yes',
            'Dependents': '0',
            'Self_Employed': 'No',
            'LoanAmount': df['LoanAmount'].median(),
            'Loan_Amount_Term': 360,
            'Credit_History': 1
        }, inplace=True)

        df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

        df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
        df['Married'] = df['Married'].map({'Yes':1,'No':0})
        df['Education'] = df['Education'].map({'Graduate':1,'Not Graduate':0})
        df['Self_Employed'] = df['Self_Employed'].map({'Yes':1,'No':0})
        df['Property_Area'] = df['Property_Area'].map({'Urban':2,'Semiurban':1,'Rural':0})

        df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']

        X = df[['Gender','Married','Dependents','Education','Self_Employed',
                'ApplicantIncome','CoapplicantIncome','LoanAmount',
                'Loan_Amount_Term','Credit_History','Property_Area',
                'Total_Income','EMI']]

        # =============================
        # PREDICT BUTTON
        # =============================
        if st.button("🚀 Predict Uploaded Data"):

            preds = model.predict(X)
            probs = model.predict_proba(X)[:,1]

            df['Prediction'] = preds
            df['Confidence'] = probs
            df['Prediction'] = df['Prediction'].map({1:'Approved',0:'Rejected'})

            st.session_state.result_df = df
            st.session_state.page = "result"
            