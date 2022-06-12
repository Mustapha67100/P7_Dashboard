import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import joblib
import os
from pathlib import Path

#CURRENT_FOLDER = Path.cwd()
CURRENT_FOLDER= os.getcwd()
PROJECT_FOLDER = Path(CURRENT_FOLDER)
DATA_FOLDER = PROJECT_FOLDER
# Liste des clients ID
lst_id=joblib.load(DATA_FOLDER/'lst_id.joblib')



#affichage formulaire
st.title('Dashboard Scoring Credit')

url_FastAPI = os.environ.get(
    'API_URL',
    'http://127.0.0.1:8000/'
)

def request_prediction(model_uri, client_id):
    headers = {"Content-Type": "application/json"}
    url=model_uri+'predict/{}'.format(client_id)
    print(url)
    response = requests.request(
        method='GET',
        url=url,
        headers=headers,
        )
    # print response
    print(response)
    if response.status_code != 200:
       raise Exception(
           "Request failed with status {}, {}".format(response.status_code, response.text))
    return response.json()
    

def main():
    #url_FastAPI = 'http://127.0.0.1:8000/'
    
    # Affichage du titre et du sous-titre
    st.title("Implémenter un modèle de scoring")
    st.markdown("<i>API répondant aux besoins du projet 7 pour le parcours Data Scientist OpenClassRoom</i>", unsafe_allow_html=True)
    
    # Requête permettant de récupérer la liste des ID clients
    #lst_id = data_test[['SK_ID_CURR']]
    #lst_id = lst_id['SK_ID_CURR'].tolist()
    client_id = st.sidebar.selectbox("Veuillez saisir l\'identifiant d\'un client", lst_id)
    #client_id = st.number_input(label='client_id', min_value=100001, max_value=100005, step=4)
    
    # Affichage solvabilité client
    st.header("**Analyse dossier client**")
    predict_btn = st.button('Prédire')
    st.markdown("<u>Probabilité de risque de faillite du client :</u>", unsafe_allow_html=True)
    if predict_btn:
        prediction= request_prediction(url_FastAPI, client_id)
        #pred = pd.DataFrame(prediction, columns=prediction.keys())
        #pred = pd.DataFrame(eval(prediction), columns=['0','1'])
        pred = pd.DataFrame(
        # prediction est un dico venant de l'API
        prediction['proba'],
        columns=['0','1']
        ) 
         
        st.dataframe(pred)
        print(pred)
        # Visialisation de probabilité
        st.bar_chart(pred)
        
        #labels = 'Target:0', 'Target :1'
        labels = ["Solvable", "Non solvable"]
        Probabilty = [pred.iloc[0,0], pred.iloc[0,1]]
        fig1, ax1 = plt.subplots()
        ax1.pie(Probabilty, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)
        # Seuil de solvabilité
        Seuil =0.50 
        if pred.iloc[0,0] >= Seuil:
            st.sidebar.markdown("<u>Différence solvabilité / non solvabilité</u>", unsafe_allow_html=True)
            st.sidebar.markdown("<i> Ce client  est solvable</i>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown("<i> Ce client  n'est pas solvable</i>", unsafe_allow_html=True)   
    return      
  
if __name__ == '__main__':
	main()
