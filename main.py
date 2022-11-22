import streamlit as st
import numpy as np
import pandas as pd
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
import shap
import streamlit.components.v1 as components
from sklearn.svm import SVC
import plotly.figure_factory as ff
import plotly.express as px

st.set_page_config(
    page_title="Let's not waste our time with bad Wine!",
    page_icon="/Users/VictoriaVigot/OneDrive/Documents/Work/DSTI/ML_streamlit/istockphoto.jpg")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Wine Quality Prediction')
st.subheader('Are you sure the bottle you are going to buy has high chance of being good ? '
             'This machine learning app uses Support Vector Machine to let you test a wine composition '
             'against a wide sample of wines tested and scored according to their chemical properties')
col1, col2 = st.columns([1, 1])



with col1:
    st.image("/Users/VictoriaVigot/OneDrive/Documents/Work/DSTI/ML_streamlit/wine_5.jpg", caption='Wine is the best')
with col2:
    st.write("""According to experts, the wine is differentiated according to its smell, flavor, and color, but not everybody is a 
    not a wine expert to say that wine is good or bad. What will we do then? Here’s the use of Machine Learning comes, 
    and more particularly the use of a Support Vector Machine model to rank wine according to some of their properties.
To the ML model, we first need to have data for that you don’t need to go anywhere just click here for the wine quality dataset. This dataset was picked up from the Kaggle.""")

st.subheader("To predict default/ failure to pay back status, you need to follow the steps below:")
st.markdown(""" 
1. Enter/choose the parameters that best descibe your applicant on the left side bar; 
2. Press the "Predict" button and wait for the result. 
""")
st.subheader("Below you could find prediction result: ")

st.sidebar.title("Wine composition Info")
st.sidebar.image("/Users/VictoriaVigot/OneDrive/Documents/Work/DSTI/ML_streamlit/istockphoto.jpg", width=100)
st.sidebar.write("Please choose parameters that descibe your wine")


# data processing and running the SVM algo:
# load wine data
cleaner_type = {"nature": {"white": 1.0, "red": 2.0}}
wine = pd.read_csv("/Users/VictoriaVigot/OneDrive/Documents/Work/DSTI/ML_streamlit/winequalityN.csv").rename(columns={"type": "nature",
                                                                               "fixed acidity": "fixed_acidity",
                                                                               "volatile acidity":"volatile_acidity",
                                                                               "citric acid":"citric_acid",
                                                                               "residual sugar":"residual_sugar",
                                                                               "free sulfur dioxide": "free_sulfur_dioxide",
                                                                               "total sulfur dioxide":"total_sulfur_dioxide",
                                                                               "alcohol":"alcohol_degree"
                                                                               }).dropna(axis=0).reset_index(drop=True)
wine = wine.replace(cleaner_type)

X = wine.iloc[:, :-1]
y = wine.iloc[:, -1] 
# stratify the y value (float)
X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.20, random_state=42, stratify=y)


nature = st.sidebar.radio("Please select your wine nature", ('white', 'red'))
fixed_acidity = st.sidebar.slider("Fixed acidity: ", min_value=3.0, max_value=20.0,step=0.2)
volatile_acidity = st.sidebar.slider("Volatile acidity:",min_value=0.1, max_value=2.0, step=0.1)
citric_acid = st.sidebar.slider("Citric acid value", min_value=0.0, max_value=2.0, step=0.1)
residual_sugar = st.sidebar.slider("Residual sugar value", min_value=0.5, max_value=25.0, step=0.5)
# = st.sidebar.selectbox('Please choose your employment length', ("< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years","8 years","9 years","10+ years") )
chlorides = st.sidebar.slider("Chlorides level:", min_value=0.05, max_value=1.0, step=0.05)
free_sulfur_dioxide = st.sidebar.slider("Sulphur dioxide level: ",min_value= 0, max_value=300, step= 10)
total_sulphur_dioxide = st.sidebar.slider("Total sulphur dioxide level: ",min_value= 10, max_value=500, step= 10)
density = st.sidebar.slider("Density: ",min_value=0.9, max_value=1.05,step=0.01)
pH = st.sidebar.slider("pH level: ",min_value=2.5, max_value=4.5,step=0.1)
sulphates = st.sidebar.slider("Sulphates content rate: ",min_value=0.1, max_value=2.0,step=0.1)
alcohol_degree =st.sidebar.slider("Alcohol degree:",min_value=7.0, max_value=16.0,step=0.5)


def preprocess(nature, fixed_acidity, volatile_acidity,
                citric_acid, residual_sugar, chlorides,
                free_sulfur_dioxide, total_sulphur_dioxide,
               density, pH, sulphates, alcohol_degree):
    # enter user input data
    user_input_dict = {'nature': [nature],
                       'fixed_acidity': [fixed_acidity],
                       'volatile_acidity': [volatile_acidity],
                       'citric_acid': [citric_acid],
                       'residual_sugar': [residual_sugar],
                       'chlorides': [chlorides],
                       'free_sulfur_dioxide': [free_sulfur_dioxide],
                       'total_sulfur_dioxide': [total_sulphur_dioxide],
                       'density': [density],
                       'pH': [pH],
                       'sulphates': [sulphates],
                        'alcohol_degree': [alcohol_degree]}

    user_input = pd.DataFrame(data=user_input_dict)

    cleaner_type = {"nature": {"white": 1.0, "red": 2.0}}
    user_input = user_input.replace(cleaner_type)
    return user_input


user_input = preprocess(nature, fixed_acidity, volatile_acidity,
                citric_acid, residual_sugar, chlorides,
                free_sulfur_dioxide, total_sulphur_dioxide,
               density, pH, sulphates, alcohol_degree)


model = SVC(probability=True)
model.fit(X_train, y_train)


st.subheader('Result for the User selection - Wine quality Level')
st.write("A general note of 1 to 10 will be given to the wine composition you selected based on a scores given to a set of more than 6000 other wines.")
btn_predict = st.sidebar.button("Predict")
st.write(""" This histogram shows the probability of each note for the wine composition you specified""")



if btn_predict:
    pred_proba = model.predict_proba(user_input)
    pred = model.predict(user_input)
    pred_proba_df = pd.DataFrame(data=
                             {'Quality note': list(set(y)),
                              'Probability value for your selection': pred_proba[0]})
    st.title("This is the note for the wine you composed ! : {}".format(pred[0]))
    st.line_chart(data=pred_proba_df, x='Quality note', y='Probability value for your selection')

    if pred[0] < 5:
            st.error("This wine is crap! Don't even use it to wipe the floor!")
    else:
            st.success("Let's try a glass of this one")
