import streamlit as st
from streamlit_shap import st_shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import joblib
import os
import shap
from pathlib import Path
import streamlit.components.v1 as components

#CURRENT_FOLDER = Path.cwd()
CURRENT_FOLDER= os.getcwd()
PROJECT_FOLDER = Path(CURRENT_FOLDER)
DATA_FOLDER = PROJECT_FOLDER
# Liste des clients ID
lst_id=joblib.load(DATA_FOLDER/'lst_id.joblib')
shap_values=joblib.load(DATA_FOLDER/'shap_values.joblib')

data_test=joblib.load(DATA_FOLDER/'data_test_sub_cutoff.joblib')
#model=joblib.load(DATA_FOLDER/'model_rf_20000.joblib')




#affichage formulaire
st.title('Dashboard Scoring Credit')

url_FastAPI = os.environ.get(
    'API_URL',
    'http://127.0.0.1:8000/'
)
#
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
    client_id = st.sidebar.selectbox("Veuillez saisir l\'identifiant d\'un client", lst_id)
    
    # Requête permettant de récupérer la liste des variable
    VAR1 = st.sidebar.selectbox("Veuillez saisir VAR1", data_test.columns)
    VAR2 = st.sidebar.selectbox("Veuillez saisir VAR2", data_test.columns)
    
    # Affichage solvabilité client
    st.header("**Analyse dossier client**")
    predict_btn = st.button('Prédire')
    
    st.markdown("<u>Probabilité de risque de faillite du client :</u>", unsafe_allow_html=True)
    
    # Calcul de la prédiction ( probabilité)
    if predict_btn:
        prediction= request_prediction(url_FastAPI, client_id)
        
        pred = pd.DataFrame(
        # prediction est un dico venant de l'API
        prediction['proba'],
        columns=['0','1']
        ) 
        # Afficher les résultats dans un dataframe
        st.dataframe(pred)
        
        # Visualiser  la prediction sous forme d'un "PIE"
        
        labels = ["Solvable", "Non solvable"]
        Probabilty = [pred.iloc[0,0], pred.iloc[0,1]]
        fig1, ax1 = plt.subplots()
        ax1.pie(Probabilty, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)
        
        # Features Importance : Shap Explainer (visualiser des informations descriptives relatives à un client)
        st.markdown("<u>Informations descriptives relatives à un client</u>", unsafe_allow_html=True)
        st_shap(shap.plots.beeswarm(shap_values))
        st_shap(shap.plots.waterfall(shap_values[lst_id.index(client_id)]))
        
        # Comparer les informations descriptives relatives à un client 
        # à l’ensemble des clients ou à un groupe de clients similaires.
        # Cas 1 : étude en fonction d'une seule variable
        
        st.markdown("<u>Comparer les informations descriptives relatives à un client à l’ensemble des clients</u>", unsafe_allow_html=True)
        arr=data_test[[VAR1]]
        fig2, ax2 = plt.subplots()
        ax2.hist(arr, bins=40)
        v = data_test.loc[client_id, VAR1]
        legend = f"Client {client_id}"
        pd.DataFrame(
         {
         VAR1: [v, v,],
         legend : [0, 10000, ],
         }
         ).plot.line(x=VAR1, y=legend, ax=ax2)
        st.pyplot(fig2)
      
        # Cas 2 : étude en fonction de deux varibla
        fig3, ax3 = plt.subplots()
        data_test.plot.scatter(VAR1,VAR2,
            c = "blue",
            marker = '1',
            alpha = 0.2,
            ax=ax3,
        )
        data_test[data_test.index == client_id].plot.scatter(VAR1,VAR2,
            c = "green",
            marker = 'X', s=100,
            alpha = 1,
            ax = ax3,
        )
        st.pyplot(fig3)
        

        # Fonction de Coût : Aide à la décision en fonction du seuil de solvabilité
        st.markdown("<u>Fonction de Coût : Aide à la décision en fonction du seuil de solvabilité</u>",unsafe_allow_html=True )
        Seuil_1 = 0.3
        Seuil_2 = 0.5
        if pred.iloc[0,1] <= Seuil_1:
            st.sidebar.markdown("<u>Différence solvabilité / non solvabilité</u>", unsafe_allow_html=True)
            st.sidebar.markdown("<i> Ce client  est solvable</i>", unsafe_allow_html=True)
            st.success("Success :Client fiable  ")
            st.balloons()
        elif pred.iloc[0,1] >= Seuil_1 and pred.iloc[0,1] <= Seuil_2:
            st.sidebar.markdown("<i> Ce client  n'est pas solvable</i>", unsafe_allow_html=True) 
            st.warning("Warning : Client risqué")
        else :  
            st.sidebar.markdown("<i> Ce client  n'est pas solvable</i>", unsafe_allow_html=True) 
            st.error("Error : Client à rejeter")
    
    return      
    
if __name__ == '__main__':
	main()
