import streamlit as st
#from flask import Flask, request
import requests
import shap
import streamlit.components.v1 as components
import pandas as pd
import json
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt 
import pickle

data=pd.read_csv('df_test_sample.csv')
listid=data['SK_ID_CURR'].tolist()
model = pickle.load(open('model.pkl','rb'))
shap_values=np.load('shap-values.npy')


endpoint='https://backend-predict.herokuapp.com'

def score(id):
    response = requests.post(endpoint+'/predict', json={'text': id})
    score = response.json()
    return score

def indix(id):
    i=data.loc[data['SK_ID_CURR'] == id].index.values
    return i

def shap_plot(j):
    explainerModel = shap.TreeExplainer(model)
    
    p = shap.decision_plot(explainerModel.expected_value[0],shap_values[j],feature_names= list(data.columns), ignore_warnings=True)
    return(p)



#titre et autre
st.title("CREDIT PREDICTION")
st.header('Influence des criteres sur le choix')
st.image('shap_summary.png')

id_input = st.selectbox("Choisissez l'identifiant d'un client", data.SK_ID_CURR)

result=score(id_input)
st.metric(label= 'probabilite de remboursement', value=1-result[0])



st.title('Graphe de decision')
st.set_option('deprecation.showPyplotGlobalUse', False)
p=shap_plot(indix(id_input))
#p=shap.decision_plot(expected_values, shap_v_1, valid_x, ignore_warnings=True)
st.pyplot(p)

enumeration=['NAME_INCOME_TYPE_Working','CODE_GENDER_M', 'NAME_FAMILY_STATUS_Married', 'REGION_RATING_CLIENT_W_CITY', 'AMT_CREDIT' ]
fig, ax = plt.subplots()
sns.boxplot(data=data[enumeration], ax = ax, flierprops={"marker": "x"}, color='skyblue', showcaps=True)
plt.setp(ax.get_xticklabels(), rotation=90)
client_data=data.query(f'SK_ID_CURR == {id_input}')
  
for k in  enumeration:
    ax.scatter(k, client_data[k].values, marker='X', s=100, color = 'black', label = 'Client selectionné')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())
st.pyplot(fig)

