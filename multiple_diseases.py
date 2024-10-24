import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.preprocessing import StandardScaler

try:
    diabetes_model = pickle.load(open("diabetes_model.sav", 'rb'))
    heart_model = pickle.load(open('heart_disease_model.sav','rb'))
    parkinsons_model = pickle.load(open('parkinsons_disease_model.sav','rb'))
    kidney_model = pickle.load(open('ChronicKidneyDisease.sav','rb'))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error: {e}")
# sidebar for navigation
with st.sidebar:
    select = option_menu(
        "Multiple Diseases Prediction",
        ["Diabetes Prediction","Heart Diseases Prediction","Parknison Prediction",'Chronic Kidney Disease Prediction',"About"],
        icons = ['activity','heart-pulse-fill','bi bi-person-walking','bi bi-lungs-fill','file-person'],
        default_index =0)


if (select == "Diabetes Prediction"):
    st.title("Diabetes Prediction")

    #Split the column into two columns
    col1,col2 = st.columns(2)

    with col1:
        Pregnancies = st.text_input("No of  Pregnancies",placeholder = 'e.g. 2')
    with col2:
        Glucose = st.text_input("Glucose Level",placeholder = 'e.g. 120')
    with col1:
        BloodPressure = st.text_input("BloodPressure Level",placeholder = 'e.g. 70')
    with col2:
        SkinThickness = st.text_input("SkinThickness Value",placeholder = 'e.g. 20')
    with col1:
        Insulin = st.text_input("Insulin Level",placeholder = 'e.g. 80')
    with col2:
        BMI = st.text_input("BMI Value",placeholder = 'e.g. 30')
    with col1:
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value",placeholder = 'e.g. 0.4532')
    with col2:
        Age = st.text_input("Age",placeholder = 'e.g. 30')

    if st.button("Diabetes Test Result"):
        diab_prediction = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        if(diab_prediction[0] == 1):
            st.error("The person is Diabetic")
        else:
            st.success("The person is Non Diabetic")


# Heart Disease Prediction Page
if select == 'Heart Diseases Prediction':
    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input('Age',placeholder = 'e.g. 35')
    with col2:
        sex = st.text_input('Sex',placeholder ='  0: Female     1: Male')
    with col1:
        cp = st.text_input('Chest Pain types',placeholder ='0: Typical angina 1: Atypical angina 2: Non-anginal pain 3: Asymptomatic')
    with col2:
        trestbps = st.text_input('Resting Blood Pressure',placeholder = 'mm Hg (millimeters of mercury)')
    with col1:
        chol = st.text_input('Serum Cholestoral',placeholder = 'mg/dl (milligrams per deciliter)')
    with col2:
        fbs = st.text_input('Fbs (Fasting Blood Sugar):',placeholder = '0: Fbs ≤ 120 mg/dl    1: Fbs > 120 mg/dl')
    with col1:
        restecg = st.text_input('Restecg (Resting Electrocardiographic Results)',placeholder = '0: Normal 1: Having ST-T wave abnormality  2: Showing probable')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved',placeholder = 'e.g. 172')
    with col1:
        exang = st.text_input('Exercise Induced Angina',placeholder = '0: No    1: Yes')
    with col2:
        oldpeak = st.text_input('ST depression induced by exercise',placeholder = 'e.g.  1.4')
    with col1:
        slope = st.text_input('Slope of the peak exercise ST segment',placeholder = '0: Upsloping    1: Flat    2: Downsloping')
    with col2:
        ca = st.text_input('Major vessels colored by flourosopy',placeholder = 'number of major vessels (0–3)')
    with col1:
        thal = st.text_input('Thalium stress test result:',placeholder = '1: Normal    2: Fixed defect   3: Reversible defect')

    # code for Prediction
    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        heart_prediction = heart_model.predict([user_input])
        if heart_prediction[0] == 1:
            st.error('The person is having heart disease')
        else:
            st.success('The person does not have any heart disease')


if select == 'Parknison Prediction':

    # page title
    st.title("Parkinson's Disease Prediction")

    col1,col2 = st.columns(2)

    with col1:
        Fo = st.text_input('MDVP:Fo(Hz)',placeholder = 'MDVP:Fo(Hz)')

    with col2:
        Fhi = st.text_input('MDVP:Fhi(Hz)',placeholder ='MDVP:Fhi(Hz)')

    with col1:
        Flo = st.text_input('MDVP:Flo(Hz)',placeholder ='MDVP:Flo(Hz)')

    with col2:
        Jitter = st.text_input('MDVP:Jitter(%)',placeholder = 'MDVP:Jitter(%)')

    with col1:
        Abs = st.text_input('MDVP:Jitter(Abs)',placeholder = 'MDVP:Jitter(Abs)')

    with col2:
        RAP = st.text_input('MDVP:RAP',placeholder = 'MDVP:RAP')

    with col1:
        PPQ = st.text_input('MDVP:PPQ',placeholder = 'MDVP:PPQ')

    with col2:
        DDP = st.text_input('Jitter:DDP',placeholder = 'Jitter:DDP')

    with col1:
        Shimmer = st.text_input('MDVP:Shimmer',placeholder = 'MDVP:Shimmer')

    with col2:
        db = st.text_input('MDVP:Shimmer(dB)',placeholder = 'MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3',placeholder = 'Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5',placeholder = 'Shimmer:APQ5')

    with col1:
        APQ = st.text_input('MDVP:APQ',placeholder = 'MDVP:APQ')

    with col2:
        DDA = st.text_input('Shimmer:DDA',placeholder = 'Shimmer:DDA')

    with col1:
        NHR = st.text_input('NHR',placeholder = 'NHR')

    with col2:
        HNR = st.text_input('HNR',placeholder = 'HNR')

    with col1:
        RPDE = st.text_input('RPDE',placeholder = 'RPDE')

    with col2:
        DFA = st.text_input('DFA',placeholder = 'DFA')

    with col1:
        spread1 = st.text_input('spread1',placeholder = 'spread1')

    with col2:
        spread2 = st.text_input('spread2',placeholder = 'spread2')

    with col1:
        D2 = st.text_input('D2',placeholder = 'D2')

    with col2:
        PPE = st.text_input('PPE',placeholder = 'PPE')

    if st.button('Parkisons Disease Test Result'):

        user_input = [Fo,Fhi,Flo,Jitter,Abs,RAP,PPQ,DDP,Shimmer,db,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]
        user_input = np.array(user_input)
        
        input_data_reshaped = user_input.reshape(1,-1)

        # scaler = StandardScaler()
        # std_data = scaler.transform(input_data_reshaped)
        
        prediction = parkinsons_model.predict(input_data_reshaped)
        
        if(prediction[0]==1):
            st.error("The person is affcted by the Parkinsons")
        else:
            st.success("The person is does not affcted by the Parkinsons")

if select == 'Chronic Kidney Disease Prediction':
    st.title("Chronic Kidney Disease Prediction")

    col1,col2 = st.columns(2)
    with col1 :
        Bp = st.text_input('Blood Pressure',placeholder = 'e.g. 76.4550')
    with col2:
        Sg = st.text_input('Specific Gravity',placeholder = 'e.g. 1.017712')
    with col1:
        Al = st.text_input('Albumin',placeholder = 'e.g. 1.0150')
    with col2:
        Su = st.text_input('Sugar',placeholder = 'e.g. 0.395000')
    with col1:
        Rbc = st.text_input('Red Blood Cells',placeholder = 'e.g. 0.88250')
    with col2:
        Bu = st.text_input('Blood Urea',placeholder = 'e.g. 57.40550')
    with col1:
        Sc = st.text_input('Serum Creatinine',placeholder = 'e.g. 3.07235')
    with col2:
        Sod = st.text_input('Sodium',placeholder = 'e.g. 137.529025')
    with col1:
        Pot = st.text_input('Potassium',placeholder = 'e.g. 4.627850')
    with col2:
        Hemo = st.text_input('Hemoglobin',placeholder='e.g. 12.52690')
    with col1:
        Wbcc = st.text_input('White Blood Cell Count',placeholder = 'e.g. 8406.090')
    with col2:
        Rbcc = st.text_input('Red Blood Cell Count',placeholder = 'e.g. 4.708275')
    with col1:
        Htn = st.text_input('Hypertension',placeholder = 'e.g. 1 (Yes) or 0(No)')

    if st.button('Kidney Disease Test Result'):
        input_data = [Bp,Sg,Al,Su,Rbc,Bu,Sc,Sod,Pot,Hemo,Wbcc,Rbcc,Htn]
        user_input = [float(x) for x in input_data]
        prediction = kidney_model.predict([user_input])
        if(prediction[0]==1):
            st.error("The person is affcted by the Chronic kidney Diseases")
        else:
            st.success("The person is does not affcted by Chronic kidney Diseases")

if select == "About":
    st.title("About App")

    st.markdown("Welcome to HealthPredict, your trusted companion for proactive health management. Our innovative application harnesses the power of Machine Learning to predict multiple diseases, focusing specifically on diabetes , parkinsons and heart diseases.")
    
    st.markdown("<h3>Key Features :</h3>",unsafe_allow_html=True)
    # st.markdown("The Multiple Diseases Prediction app helps users check their risk for three major health conditions: diabetes, heart diseases, and Parkinson's disease. It uses machine learning to analyze health information and give a quick prediction.")
    st.markdown("<b>Diabetes Prediction: </b>This feature checks if you are at risk of diabetes based on factors like blood sugar levels, age, weight, and more. It helps you know early if you should take action to manage your health.",unsafe_allow_html=True)
    st.markdown("<b>Heart Disease Prediction: </b>The app looks at things like cholesterol, blood pressure, and heart rate to tell if you may be at risk of heart disease. Knowing this can help you make lifestyle changes or consult a doctor.",unsafe_allow_html=True)
    st.markdown("<b>Parkinson’s Disease Prediction: </b>Parkinson’s is a disease that affects movement and coordination. The app analyzes symptoms like tremors or slow movements to check if you might be at risk. Early detection can help you take steps for better treatment.",unsafe_allow_html=True)
    st.markdown("<b>Chronic Kidney Disease Prediction: </b>Chronic Kidney Disease (CKD) is a condition where the kidneys gradually lose their ability to filter waste from the blood. This app analyzes symptoms such as fatigue, swelling, or changes in urination to assess your risk of CKD. Early detection allows for timely intervention, helping you manage the disease and improve long-term health outcomes.",unsafe_allow_html = True)

    st.markdown("<h3>How It Works :</h3>",unsafe_allow_html=True)
    st.markdown("<li>You enter your health information into the app (like blood sugar, blood pressure, etc.).</li>",unsafe_allow_html=True)
    st.markdown("<li>The app uses this information to give you a score, showing how likely you are to have each condition.</li>",unsafe_allow_html=True)
    st.markdown("<li>Based on the results, you can decide to visit a doctor or make lifestyle changes</li>",unsafe_allow_html=True)

    st.markdown("<h3>Why Use the App?</h3>",unsafe_allow_html=True)
    st.markdown("<li>Get quick and easy predictions for diabetes, heart disease, Kidney diseases and Parkinson’s.</li>",unsafe_allow_html=True)
    st.markdown("<li>Find out early if you are at risk and take steps to improve your health.</li>",unsafe_allow_html=True)
    st.markdown("<li>Keep track of your health by checking regularly.</li>",unsafe_allow_html=True)
    st.markdown("<li>Use the information to make better health decisions.</li>",unsafe_allow_html=True)

    st.markdown("")
    st.markdown("<h6><center>Copyright © Vasantha Raj - All rights reserved</center></h6>",unsafe_allow_html=True)





# Inject CSS to change font style and other properties
st.markdown(
    """
    <style>
    /* Change the font style for the entire app */
    body {
        font-family: serif;

    }

    /* Change the font style for specific elements */
    h1,h3 {
        font-family: serif;
        # color: darkblue;
    }

    .stButton button {
        font-family: sans-serif;
        font-size: 16px;
        color: white;
        background-color: green;
    }

    /* Additional styling can be added here */
    .stMarkdown {
        font-size: 30px;
        # color: red;
    }
    </style>
    """,
    unsafe_allow_html=True
)
