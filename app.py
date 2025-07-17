import streamlit as st
import pandas as pd
import pickle
import nbformat
from streamlit_option_menu import option_menu
from streamlit.components.v1 import html
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the CSV
@st.cache_data
def load_data():
    return pd.read_csv("telco.csv")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('Random Forest_model.pkl')

# Prediction function
def make_prediction(model, input_data):
    return model.predict([input_data])[0]

def show_notebook():
    st.title("📒 Jupyter Notebook: Customer Churn Project")

    with open("customer-churn-ensemble-learning.html", 'r', encoding='utf-8') as f:
        html_data = f.read()
    st.components.v1.html(html_data, height=1000, scrolling=True)

# UI Navigation Menu
st.set_page_config(page_title="Customer Churn App", layout="wide")

with st.sidebar:
    selected = option_menu("Menu", ["Explore Data", "Model Comparison",  "View Notebook"], 
                           icons=["table", "bar-chart", "book"],
                           menu_icon="cast", default_index=0)

# Explore Data
if selected == "Explore Data":
    st.title("📊 Explore Telco Churn Data")
    df = load_data()
    st.write(df.head(10))
    st.write("Shape:", df.shape)
    st.write("Churn Value Counts:")
    st.bar_chart(df['Churn'].value_counts())

# Model Comparison Placeholder (If you want graphs here)
elif selected == "Model Comparison":
    st.title("🤖 Model Performance (RandomForest / XGBoost / LightGBM)")
    st.write("This section can include ROC AUC, accuracy, precision-recall, confusion matrix, etc.")
    st.success("Already done in your notebook! You can show results here too manually.")

elif selected == "View Notebook":
    show_notebook() 

# Prediction Page
#''' elif selected == "Make Prediction":
 #   st.title("🔍 Predict Churn from User Input")

  #  model = load_model()

    # Example inputs (must match model features)
   # st.subheader("Enter Customer Details")

    #gender = st.selectbox("Gender", ["Male", "Female"])
#    senior = st.selectbox("Senior Citizen", [0, 1])
 #   partner = st.selectbox("Has Partner?", ["Yes", "No"])
  #  tenure = st.slider("Tenure (months)", 0, 72, 12)
   # monthly_charges = st.number_input("Monthly Charges", value=70.0)
    #total_charges = st.number_input("Total Charges", value=1000.0)

    # Convert inputs into correct format
#    encoder = LabelEncoder()
 #   gender = 1 if gender == "Male" else 0
  #  partner = 1 if partner == "Yes" else 0

   # input_features = [gender, senior, partner, tenure, monthly_charges, total_charges]

    #if st.button("Predict Churn"):
     #   prediction = make_prediction(model, input_features)
    #    st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}") '''

