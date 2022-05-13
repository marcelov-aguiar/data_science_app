from distutils.log import error
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from utils import utils
import pickle

def app():
    base_path = os.path.dirname(os.path.realpath(__file__))

    path_deploy = os.path.abspath(os.path.join(base_path,'..' ,'deploy'))

    path_model = os.path.join(path_deploy, 'Model.m')

    path_input = os.path.abspath(os.path.join(sys.path[0],'..' ,'data','process'))

    if not(os.path.isfile(os.path.join(path_model))):
        st.warning("Please upload data through `Upload Data` page!")
        st.warning('Train your model in the `Machine Learning`.')
    else:
        st.markdown("## Data Upload")

        # Upload the dataset and save as csv
        st.markdown("### Upload a csv file for prediction.") 
        st.write("\n")

        # Code to read a single file 
        uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])

        if uploaded_file is not None:
            try:
                df_data = pd.read_csv(uploaded_file, sep=';')
            except Exception as e:
                st.warning(e)
                df_data = pd.read_excel(uploaded_file)

            # contém o nome das colunas usadas no treino do modelo
            meta_data = utils.read_json(os.path.join(path_input,'meta_data.json'))

            # verifica se as colunas usadas no treino estão presentes no arquivo
            # que o usuário fez o upload
            has_column = True
            aux_y_var = []
            aux_y_var.append(meta_data['y_column_name'])
            all_columns = list(meta_data['training_column_name']) + aux_y_var
            for column in all_columns:
                if not(column in df_data.columns):
                    st.error(f"The column {column} used in the training is not " +
                              "present in the upload file.")
                    has_column = False

            X_var = meta_data['training_column_name']
            y_var = meta_data['y_column_name']

        if st.button("Data Prediction") and has_column:
            model = pickle.load(open(path_model, 'rb'))
            df_data['data'] = pd.to_datetime(df_data['data'])
            df_data = df_data.sort_values(by='data')

            df_data['pred'] = model.predict(df_data[X_var])
            fig = plt.figure(figsize=(20,6))
            plt.title(f"Predict {y_var}")
            plt.xlabel("Date")
            plt.ylabel(f"{y_var}")

            plt.plot(df_data['data'].values, df_data['pred'].values, 'X',
                     color='firebrick', label='Predicted', linewidth=2, zorder=10)

            plt.plot(df_data['data'].values, df_data[f'{y_var}'].values, 'o--',
                     color='royalblue', label='Real', linewidth=2)

            plt.legend()
            st.pyplot(fig)
            # fig = go.Figure()
# 
            # fig.add_trace(go.Scatter(x=df_data['data'],
            #                          y=df_data['pred'],
            #                          mode='markers',
            #                          name=f'{y_var} Pred'))
# 
            # fig.add_trace(go.Scatter(x=df_data['data'],
            #                          y=df_data[X_var],
            #                          line=dict(color='firebrick'),
            #                          name=f'{y_var} Real'))

            