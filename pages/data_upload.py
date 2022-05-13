from operator import length_hint
import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

def app():
    """Objetivo dessa pagina é fazer o upload do arquivo selecionado pelo usuário
    e levantar algumas informações do DataFrame, como:
    - Nome das colunas.
    - Tipo de cada coluna.
    - Se as colunas sao numericas.
    - Quantidade de dados nao numéricos em cada coluna.
    - Quantidade de dados nulos.
    - Quantidade de linhas duplicadas no DataFrame.
    """
    st.markdown("## Data Upload")

    # Upload the dataset and save as csv
    st.markdown("### Upload a csv file for analysis.") 
    st.write("\n")

    # Code to read a single file 
    uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, sep=';')
        except Exception as e:
            st.warning(e)
            data = pd.read_excel(uploaded_file)
    
    if st.button("Load Data"):
        
        if (uploaded_file is not None):
            st.info("File uploaded successfully")

            # Raw data
            st.markdown("#### These are the first five lines of the dataframe")
            st.dataframe(data.head(5))
            
            path_output = os.path.abspath(os.path.join(sys.path[0],'..' ,'data','process'))
            if not os.path.isdir(path_output):
                os.mkdir(path_output)
            data.to_csv(os.path.join(path_output, 'main_data.csv'), index=False, sep=',')

            st.markdown("#### This is some information from DataFrame")
            info = dict()
            info['column name'] = list()
            info['columns type'] = list()
            info['column is numeric'] = list()
            info['amount of non-numeric data'] = list()
            info['amount of null data'] = list()

            for column in data.columns:
                info['column name'].append(column)
                
                info['columns type'].append(str(data[column].dtype))
                
                is_numeric = pd.to_numeric(data[column], errors='coerce').notnull().all()
                info['column is numeric'].append(str(is_numeric))

                lenght_not_numeric = data[pd.to_numeric(data[column], 
                                                        errors='coerce').isnull()].shape[0]
                info['amount of non-numeric data'].append(str(lenght_not_numeric))

                length_date_null = data[column].isnull().sum()
                info['amount of null data'].append(str(length_date_null))

            
            df_info = pd.DataFrame(info)
            st.dataframe(df_info)

            date_duplicated = data.duplicated().sum()
            st.write(f"This DataFrame has {date_duplicated} duplicate rows "+
                     f"out of a total of {data.shape[0]}")

            st.write("Go to the Change Metadata page to change the type of columns or "+
                     "select the columns that will go through the next steps")
            
            # removendo arquivo metadata caso ele ja tenha sido gravado
            if (os.path.isfile(os.path.join(path_output, 'main_data_change.csv'))):
                os.remove(os.path.join(path_output, 'main_data_change.csv'))
        else:
            st.warning('Select the file before uploading it')
    