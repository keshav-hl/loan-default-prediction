import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open('model/model.pkl','rb'))

st.set_page_config(page_title="Loan Prediction", layout="wide")

# ==============================
# 🧠 SESSION STATE
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "main"

if "result_df" not in st.session_state:
    st.session_state.result_df = None

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Prediction"

# ==============================
# 📊 RESULT PAGE
# ==============================
if st.session_state.page == "result":

    df = st.session_state.result_df

    st.title("📊 Loan Analysis Dashboard")

    if df is not None:

        # ==============================
        # 🔥 TOP RIGHT BUTTONS (PRO UI)
        # ==============================
        col_title, col_btn1, col_btn2 = st.columns([6,1,1])

        with col_btn1:
            if st.button("📋 Predictions"):
                st.session_state.view_mode = "Prediction"

        with col_btn2:
            if st.button("📊 Charts"):
                st.session_state.view_mode = "Charts"

        st.markdown("---")

        # ==============================
        # 📋 PREDICTION TABLE VIEW
        # ==============================
        if st.session_state.view_mode == "Prediction":

            st.subheader("📋 Prediction Results")

            def highlight(row):
                if row['Prediction'] == 'Rejected':
                    return ['background-color: rgba(220, 38, 38, 0.12)'] * len(row)
                else:
                    return ['background-color: rgba(37, 99, 235, 0.12)'] * len(row)

            st.dataframe(df.style.apply(highlight, axis=1), use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "loan_predictions.csv")

        # ==============================
        # 📊 CHART DASHBOARD VIEW (GRID STYLE)
        # ==============================
        elif st.session_state.view_mode == "Charts":

            st.subheader("📊 Analytics Dashboard")

            # ==============================
            # 📈 METRICS
            # ==============================
            st.markdown("### 📈 Overview Metrics")
            colm1, colm2, colm3 = st.columns(3)

            with colm1:
                st.metric("Total Applications", len(df))

            with colm2:
                approved = (df['Prediction'] == 'Approved').sum()
                st.metric("Approved Loans", approved)

            with colm3:
                rejected = (df['Prediction'] == 'Rejected').sum()
                st.metric("Rejected Loans", rejected)

            st.markdown("---")

            # ==============================
            # 📊 ROW 1 → PIE + HISTOGRAM
            # ==============================
            col1, col2 = st.columns(2)

            # PIE CHART
            with col1:
                st.markdown("#### 🟢 Loan Approval Distribution")
                fig1, ax1 = plt.subplots(figsize=(5,5))
                colors = ['#22c55e', '#ef4444']
                df['Prediction'].value_counts().plot(
                    kind='pie',
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90,
                    wedgeprops={'edgecolor': 'white'},
                    textprops={'fontsize': 10},
                    ax=ax1
                )
                ax1.set_ylabel('')
                ax1.axis('equal') 
                plt.tight_layout() 
                st.pyplot(fig1, use_container_width=True)

            # HISTOGRAM
            with col2:
                st.markdown("#### 📊 Confidence Score Distribution")
                fig2, ax2 = plt.subplots(figsize=(5,5))
                ax2.hist(df['Confidence'], bins=15, edgecolor='black')
                ax2.set_xlabel("Confidence Score")
                ax2.set_ylabel("Frequency")
                ax2.grid(alpha=0.3)
                st.pyplot(fig2, use_container_width=True)

            st.markdown("---")

            # ==============================
            # 📊 ROW 2 → SCATTER + BAR
            # ==============================
            col3, col4 = st.columns(2)

            # SCATTER
            with col3:
                st.markdown("#### 💰 Income vs Loan Amount")
                fig3, ax3 = plt.subplots(figsize=(4,4))
                colors = df['Prediction'].map({'Approved':'green','Rejected':'red'})
                ax3.scatter(
                    df['Total_Income'],
                    df['LoanAmount'],
                    c=colors,
                    alpha=0.7,
                    edgecolors='black'
                )
                ax3.set_xlabel("Total Income")
                ax3.set_ylabel("Loan Amount")
                ax3.grid(alpha=0.3)
                st.pyplot(fig3, use_container_width=True)

            # BAR CHART
            with col4:
                st.markdown("#### 🏘️ Property Area vs Approval")
                fig4, ax4 = plt.subplots(figsize=(4,4))
                area_group = df.groupby('Property_Area')['Prediction'].value_counts().unstack()
                area_group.plot(kind='bar', ax=ax4)
                ax4.set_xlabel("Property Area")
                ax4.set_ylabel("Count")
                ax4.grid(axis='y', alpha=0.3)
                st.pyplot(fig4, use_container_width=True)

    if st.button("⬅️ Back to Main Page"):
        st.session_state.page = "main"

    st.stop()

# ==============================
# 🏦 MAIN PAGE
# ==============================
st.title("🏦 Loan Default Prediction System")
st.markdown("### Predict loan approval using ML (Individual + Bulk CSV)")

tab1, tab2 = st.tabs(["🔹 Single Prediction", "📂 Bulk Prediction"])

# ==============================
# 🔹 SINGLE PREDICTION
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

    dependents = 3 if dependents=='3+' else int(dependents)
    gender = 1 if gender=='Male' else 0
    married = 1 if married=='Yes' else 0
    education = 1 if education=='Graduate' else 0
    self_emp = 1 if self_emp=='Yes' else 0
    area = {'Urban':2,'Semiurban':1,'Rural':0}[area]

    total_income = income + co_income
    emi = loan_amt / term if term != 0 else 0

    features = np.array([[gender, married, dependents, education, self_emp,
                          income, co_income, loan_amt, term, credit,
                          area, total_income, emi]])

    if st.button("🔍 Predict Loan Status"):

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        if pred == 1:
            st.success(f"✅ Loan Approved (Confidence: {prob:.2f})")
        else:
            st.error(f"❌ Loan Rejected (Risk Score: {1-prob:.2f})")

# ==============================
# 📂 BULK CSV
# ==============================
with tab2:

    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        st.session_state.uploaded_df = pd.read_csv(file)

    if st.session_state.uploaded_df is not None:

        df = st.session_state.uploaded_df.copy()

        st.dataframe(df.head())

        if st.button("🗑️ Remove Dataset"):
            st.session_state.uploaded_df = None
            st.rerun()

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

        if st.button("🚀 Predict Uploaded Data"):

            preds = model.predict(X)
            probs = model.predict_proba(X)[:,1]

            df['Prediction'] = preds
            df['Confidence'] = probs
            df['Prediction'] = df['Prediction'].map({1:'Approved',0:'Rejected'})

            st.session_state.result_df = df
            st.session_state.page = "result"
