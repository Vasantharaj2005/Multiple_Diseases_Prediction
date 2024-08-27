import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

try:
    diabetes_model = pickle.load(open("diabetes_model.sav", 'rb'))
    heart_model = pickle.load(open('heart_disease_model.sav','rb'))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error: {e}")


# diabetes_model = pickle.load(open("diabetes_model.sav",'rb'))

# heart_model = pickle.load(open('heart_disease_model.sav','rb'))

# sidebar for navigation
with st.sidebar:
    select = option_menu(
        "Multiple Diseases Prediction",
        ["Diabetes Prediction","Heart Diseases Prediction","Data Visualization","About"],
        icons = ['activity','heart-pulse-fill','file-bar-graph','file-person'],
        default_index =0)


if (select == "Diabetes Prediction"):
    st.title("Diabetes Prediction using ML")

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
        #st.markdown('Thalium stress test result:0: Normal 1: Fixed defect 2: Reversible defect 3: Not described',unsafe_allow_html = True)
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

if select == "Data Visualization":
    #page title
    st.title("📊 Data Visulization")

    work_dir = os.path.dirname(os.path.abspath(__file__))

    # folder_path = f"{work_dir}/Multiple_Diseases_Prediction"
    folder_path = f"C:/Users/vasan/OneDrive/Desktop/Git MDp/Multiple_Diseases_Prediction"

    file_list = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    #select box
     
    selected_file = st.selectbox("Select the Files",file_list,index = None)
    if selected_file:
        #gethig the complete path of the selected files
        
        file_path = os.path.join(folder_path,selected_file)
        
        df = pd.read_csv(file_path)
        
        col1,col2 = st.columns(2)
        columns = df.columns.tolist()

        with col1:
            st.write("")
            st.write(df.head())

        with col2:
            x_axis = st.selectbox("Select the Column for X-axis",options = columns+["None"],index = None)
            y_axis = st.selectbox("Select the Column for Y-axis",options = columns+["None"],index = None)

        plot_list = ["Line Plot","Bar Chart","Scatter Plot","Distribution Plot"]

        selected_plot = st.selectbox("Select the Plot type",options = plot_list,index = None)

    if st.button("Generate Plot"):

        fig,ax = plt.subplots(figsize = (6,4))

        if selected_plot == "Line Plot":
            sns.lineplot(x = df[x_axis],y = df[y_axis], ax = ax)
        elif selected_plot == "Bar Chart":
            sns.barplot(x = df[x_axis],y = df[y_axis], ax = ax)
        elif selected_plot == "Scatter Plot":
            sns.scatterplot(x = df[x_axis],y = df[y_axis], ax = ax)
        elif selected_plot == "Distribution Plot":
            sns.histplot(x = df[x_axis],y = df[y_axis],kde = True, ax = ax)
        # elif selected_plot == "Count Plot":
        #     sns.countplot(x = df[x_axis],y = df[y_axis], ax = ax)

        #aduj label size
        ax.tick_params(axis="x",labelsize = 10)
        ax.tick_params(axis="y",labelsize = 10)
        
        plt.title(f"{selected_plot} of {x_axis} and {y_axis}",fontsize = 12)

        plt.xlabel(x_axis,fontsize = 10)
        plt.ylabel(y_axis,fontsize = 10)
        
        st.pyplot(fig)

if select == "About":
    st.title("About App")

    st.markdown("Welcome to HealthPredict, your trusted companion for proactive health management. Our innovative application harnesses the power of Machine Learning to predict multiple diseases, focusing specifically on diabetes and heart diseases.")
    
    st.markdown("<h3>Mission</h3>",unsafe_allow_html=True)

    st.markdown("At HealthPredict, our mission is to empower individuals with the knowledge and tools they need to take charge of their health. By providing accurate and timely predictions, we aim to help users make informed decisions and adopt healthier lifestyles.")
    st.markdown("<h3>How It Works</h3>",unsafe_allow_html=True)

    st.markdown("HealthPredict utilizes advanced Machine Learning algorithms to analyze your health data and predict the likelihood of developing diabetes or heart diseases. Our app takes into account various factors, such as medical history, lifestyle choices, and biometric data, to provide personalized insights and recommendations.")

    st.markdown("<h3>Features</h3>",unsafe_allow_html=True)

    st.markdown("<b>Disease Prediction: </b>Receive early warnings for diabetes and heart diseases based on your health data.",unsafe_allow_html=True)
    st.markdown("<b>Personalized Insights: </b>Understand your risk factors and get tailored advice to improve your health.",unsafe_allow_html=True)
    st.markdown("<b>User-Friendly Interface: </b>Enjoy a seamless experience with our intuitive design and easy-to-navigate interface.",unsafe_allow_html=True)
    st.markdown("<b>Data Privacy:</b> Your health data is securely stored and used only for the purpose of providing accurate predictions.",unsafe_allow_html=True)

    st.markdown("<h3>Why Choose HealthPredict?</h3>",unsafe_allow_html=True)
    st.markdown("<b>Accuracy: </b>Our app uses state-of-the-art Machine Learning models trained on extensive datasets to ensure high accuracy in disease prediction.",unsafe_allow_html=True)
    st.markdown("<b>Proactive Health Management: </b>By identifying potential health issues early, you can take preventive measures and seek medical advice when necessary.",unsafe_allow_html=True)
    st.markdown("<b>Continuous Improvement: </b>We are committed to regularly updating our algorithms and features to provide the best possible service to our users.",unsafe_allow_html=True)





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
