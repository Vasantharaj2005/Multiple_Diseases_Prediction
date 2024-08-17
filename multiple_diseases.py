import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("streamlit-option-menu")

# Now you can import the package
from streamlit_option_menu import option_menu

# Rest of your code
diabetes_model = pickle.load(open('C:/Users/vasan/OneDrive/Desktop/MultipleDiseases/dibetes_model.sav','rb'))

heart_model = pickle.load(open('C:/Users/vasan/OneDrive/Desktop/MultipleDiseases/heart_disease_model.sav','rb'))

# sidebar for navigation
with st.sidebar:
    select = option_menu(
        "Multiple Diseases Prediction",
        ["Diabetes Prediction","Heart Diseases Prediction","Data Visualization","About"],
        icons = ['activity','heart-pulse-fill','file-bar-graph','file-person'],
        default_index =0)


if (select == "Diabetes Prediction"):
    st.title("Diabetes Prediction using ML")

    col1,col2,col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("No of  Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("BloodPressure Level")
    with col1:
        SkinThickness = st.text_input("SkinThickness Value")
    with col2:
        Insulin = st.text_input("Insulin Level")
    with col3:
        BMI = st.text_input("BMI Value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value")
    with col2:
        Age = st.text_input("Age")

    diab_digosis = ''

    if st.button("Diabetes Test Result"):
        diab_prediction = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

        if(diab_prediction[0] == 1):
            diab_digosis = "The person is Diabetic"
        
        else:
            diab_digosis = "The person is Non Diabetic"

    if( diab_digosis == "The person is Diabetic"):
        st.warning(diab_digosis)
    else:
        st.success(diab_digosis)



# Heart Disease Prediction Page
if select == 'Heart Diseases Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col1:
        cp = st.text_input('Chest Pain types')
        # option = ["0: Typical angina","1: Atypical angina","2: Non-anginal pain","3: Asymptomatic"]
        # select = st.selectbox("Chest Pain Types",option)
        # if(select == '0: Typical angina'):
        #     cp =0
        # elif(select == '1: Atypical angina'):
        #     cp = 1
        # elif(select == '2: Non-anginal pain'):
        #     cp = 2
        # elif(select == '3: Asymptomatic'):
        #     cp = 3

    with col2:
        trestbps = st.text_input('Resting Blood Pressure')

    with col1:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col2:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col1:
        exang = st.text_input('Exercise Induced Angina')

    with col2:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col1:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col2:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        #st.markdown('Thalium stress test result:0: Normal 1: Fixed defect 2: Reversible defect 3: Not described',unsafe_allow_html = True)
        thal = st.text_input('Thalium stress test result: 0: Normal 1: Fixed defect 2: Reversible defect 3: Not described''')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)


if select == "Data Visualization":
    #page configuration
    # st.set_page_config(page_title = "Visualization Page",
    #               layout = "centered",
    #               page_icon ="📊"
    #               )

    #page title
    st.title("📊 Data Visuliser - Web App")

    work_dir = os.path.dirname(os.path.abspath(__file__))

    folder_path = f"{work_dir}/Data"

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

        plot_list = ["Line Plot","Bar Chart","Scatter Plot","Distribution Plot","Count Plot"]

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
        elif selected_plot == "Count Plot":
            sns.countplot(x = df[x_axis],y = df[y_axis], ax = ax)

        #aduj label size
        ax.tick_params(axis="x",labelsize = 10)
        ax.tick_params(axis="y",labelsize = 10)
        
        plt.title(f"{selected_plot} of {x_axis} and {y_axis}",fontsize = 12)

        plt.xlabel(x_axis,fontsize = 10)
        plt.ylabel(y_axis,fontsize = 10)
        
        st.pyplot(fig)

if select == "About":
    st.title("About Our App")

    st.markdown("Welcome to HealthPredict, your trusted companion for proactive health management. Our innovative application harnesses the power of Machine Learning to predict multiple diseases, focusing specifically on diabetes and heart diseases.")
    
    st.markdown("<h3>Our Mission</h3>",unsafe_allow_html=True)

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
