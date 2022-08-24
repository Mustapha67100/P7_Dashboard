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


#CURRENT_FOLDER = Path.cwd()
CURRENT_FOLDER= os.getcwd()
PROJECT_FOLDER = Path(CURRENT_FOLDER)
DATA_FOLDER = PROJECT_FOLDER

# Liste des clients ID
lst_id=joblib.load(DATA_FOLDER/'lst_id.joblib')
shap_values=joblib.load(DATA_FOLDER/'shap_values.joblib')
#shap_values=joblib.load(DATA_FOLDER/'shap_values_log.joblib')
data_test=joblib.load(DATA_FOLDER/'data_test_sub_cutoff.joblib')

# Titre du Dashbord
st.title('Dashboard Scoring Credit : Implémenter un modèle de scoring')

# Adresse URL de l'API
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
    
    # Sous titre de description de l'API
    st.markdown("<i>API répondant aux besoins du projet 7 pour le parcours Data Scientist OpenClassRoom</i>", unsafe_allow_html=True)
    
    # Requête permettant de récupérer la liste des ID clients
    client_id = st.sidebar.selectbox("Veuillez saisir l\'identifiant d\'un client", lst_id)
    
    # Requête permettant de récupérer la liste des variable
    VAR1 = st.sidebar.selectbox("Veuillez saisir VAR1", data_test.columns)
    VAR2 = st.sidebar.selectbox("Veuillez saisir VAR2", data_test.columns)
   
    # Affichage solvabilité client
    #st.header("**Lecement de la prédiction**")
    #predict_btn = st.button('Prédire') 
    
    # Calcul de la prédiction ( probabilité)
    #if predict_btn:
    prediction= request_prediction(url_FastAPI, client_id)
    pred = pd.DataFrame(
    # prediction est un dico venant de l'API
    prediction['proba'],
    columns=['0','1']
    ) 
    #st.header("**Résultats de la prédiction**")
    #st.markdown("<u>Probabilité de risque de faillite d'un client :</u>", unsafe_allow_html=True)
    # Afficher les résultats de la prédiction dans un dataframe
    #st.dataframe(pred)
        
    # Visualiser  la prediction sous forme d'un "PIE"
    labels = ["Solvable", "Non solvable"]
    Probabilty = [pred.iloc[0,0], pred.iloc[0,1]]
    
    # Fonction de Coût : Aide à la décision en fonction du seuil de solvabilité    
    st.header("**Aide à la décision en fonction du seuil de solvabilité**")
    st.markdown("<u> Aide à la décision en fonction du seuil de solvabilité déterminé par la Fonction de Coût :</u>",unsafe_allow_html=True )
    Seuil_1 = 0.3
    Seuil_2 = 0.5
    proba=pred.iloc[0,1]
    if proba <= Seuil_1:
    #st.sidebar.markdown("<u>Différence solvabilité / non solvabilité</u>", unsafe_allow_html=True)
     #st.sidebar.markdown("<i> Ce client  est solvable</i>", unsafe_allow_html=True)
        st.success(f"Client fiable : Probabilité de défaut {proba:4.2f} inférieure à {Seuil_1:4.2f}")
    elif  proba <= Seuil_2:
    #st.sidebar.markdown("<i> Ce client  n'est pas solvable</i>", unsafe_allow_html=True) 
        st.warning(f"Client risqué : Probabilité de défaut {proba:4.2f} comprise entre {Seuil_1:4.2f} et {Seuil_2:4.2f}" )
    else :  
    #st.sidebar.markdown("<i> Ce client  n'est pas solvable</i>", unsafe_allow_html=True) 
        st.error(f"Client à rejeter : Probabilité de défaut {proba:4.2f} supérieure à {Seuil_2:4.2f}")
    
    
    #fig1, ax1 = plt.subplots()
    #ax1.pie(Probabilty, labels=labels, autopct='%1.1f%%', startangle=90)
    #ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    #st.pyplot(fig1, clear_figure=True)
    
    # Features Importance : Shap Explainer (visualiser des informations descriptives relatives à un client)
    st.header("**Features importance**")
    st.markdown("<u>Informations descriptives relatives à un client donné</u>", unsafe_allow_html=True)
        
    #st_shap(shap.plots.beeswarm(shap_values))
    st_shap(shap.plots.waterfall(shap_values[lst_id.index(client_id)],max_display=10 ))
    
                     
        #
        
    # Comparer les informations descriptives relatives à un client 
    # à l’ensemble des clients ou à un groupe de clients similaires.
    # Cas 1 : étude en fonction d'une seule variable
    st.header("Positionnement d'un client par rapport à un groupe de clients")
    st.markdown("<u>Comparaison entre les informations descriptives relatives à un client donné et celles d'un groupe de clients</u>", unsafe_allow_html=True)
    st.markdown("<u>Cas 1 : étude en fonction d'une seule variable</u>", unsafe_allow_html=True)
    st.markdown("## Comparer les informations descriptives relatives à un client à l’ensemble des clients", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots()
    ax2.hist(data_test[[VAR1]], bins=40)
    y_bounds = ax2.get_ylim()
    v = data_test.loc[client_id, VAR1]
    legend_label = f"Client {client_id}"
    pd.DataFrame(
        {
        VAR1: [v, v,],
        legend_label : y_bounds,
        }
    ).plot.line(x=VAR1, y=legend_label, ax=ax2)
    st.pyplot(fig2)
        
    # Cas 2 : étude en fonction de deux varibles
    st.markdown("<u>Cas 2 : étude en fonction de deux varibles</u>", unsafe_allow_html=True)
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
        
    
    #return      
   
if __name__ == '__main__':
	main()