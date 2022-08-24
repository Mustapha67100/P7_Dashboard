API
Installation initiale: pip install fastapi
                       pip install uvicorn

Démarrer l'app localement : uvicorn main:app --reload
URL 1 à tester: http://localhost:8000/docs
URL 2 à tester: http://localhost:8000/predict/343918

sur Heroku : 
URL 1 à tester : https://my-scoring-model.herokuapp.com/docs
URL 2 à tester : https://my-scoring-model.herokuapp.com/predict/343918

Dashboard
Installation initiale: pip install streamlit
                       pipi install shap
                       pip install streamlit_shap 

Démarrer l'app localement : streamlit run app.py
URL 1 à tester: http://localhost:8501
Quelques clients type :  
                ID_client 343918 : Client fiable (seuil de probabilité de la classe 1 est inférieur de 0.3)
                ID_client 180189 : Client risqué (seuil de probabilité de la classe 1 compris en 0.3 et 0.5)
                ID_client 382144 : Client à rejeter (seuil de probabilité de la classe 1 est superieur à 0.5 )

sur Heroku : 
URL  à tester : https://my-scoring-datshboard.herokuapp.com/