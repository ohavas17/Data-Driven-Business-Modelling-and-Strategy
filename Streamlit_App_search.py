# Load packages (comments for more special stuff)

import sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle # un-pickling stuff from training notebook
from sklearn.preprocessing import StandardScaler
import numpy as np
import itertools # we need that to flatten ohe.categories_ into one list for columns
import streamlit as st
def read_objects():
    rfc_100 = pickle.load(open('rfc_100_model.pkl','rb'))
    ohe = pickle.load(open('ohe.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl','rb'))
    data_merged_1 = pickle.load(open('data_merged_1.pkl', 'rb'))
    X_cat = list(itertools.chain(*ohe.categories_))
    return  rfc_100, ohe, scaler, data_merged_1, X_cat
rfc_100, ohe, scaler, data_merged_1, X_cat= read_objects()

st.set_page_config(page_title='The Maternity Foundation',
                    page_icon="👶",
                    layout='wide')

from PIL import Image
image = Image.open('maternity.png')

st.image(image)
st.title('The Onboarding Module prediction')
user_id = st.text_input("Enter the ID of the user, you want to predict onboarding module:")

# Search for the row with the specified ID using the loc method
new_df_cat = data_merged_1[data_merged_1.index == user_id]
st.write('This is the user characteristic that will be used for the prediction')
st.dataframe(new_df_cat)
st.caption('The parameters that will be used to predict the onboarding module for the user UserinfoDistrict, HealthcareWorker, Profession, Experience, WorkPlace and SafeDeliveryApp')
if st.button('Predict👩‍⚕️👨‍⚕️'):
    
    new_values_cat = pd.DataFrame(ohe.transform(new_df_cat), columns = X_cat , index=[0])
    # transformed_nummerical = scaler.transform(new_values_cat)
    predicted_value = rfc_100.predict(new_values_cat)

    st.title("Onboarding Module")
    if predicted_value == '473100':    
        st.write('Active_Management_of_Third_Stage_Labour')
    if predicted_value == '473101':
        st.write('Infection_Prevention')
    if predicted_value == '473102': 
        st.write('Manual_Removal_of_Placenta')
    if predicted_value == '473103':
        st.write('Female_Genital_Mutilation')
    if predicted_value == '473104':
        st.write('Hypertension')
    if predicted_value == '473105':
        st.write('Neonatal_Resuscitation')
    if predicted_value == '473106':
        st.write('Post_Partum_Hemorrhage')
    if predicted_value == '473107':
        st.write('Maternal_Sepsis')
    if predicted_value == '473108':
        st.write('Newborn_Management')
    if predicted_value == '473109':
        st.write('Post_Abortion_Care')
    if predicted_value == '473110':
        st.write('Prolonged_Labour')
    if predicted_value == '595063':
        st.write('Low_Birth_Weight')
    if predicted_value == '603672':
        st.subheader('COVID-19')
    if predicted_value == '631971':
        st.write('Normal_Labour_and_Birth')
    if predicted_value == '649662':
        st.write('Care_of_The_Sick_Newborn')
    if predicted_value == '786735':
        st.write('Safe_Abortion')