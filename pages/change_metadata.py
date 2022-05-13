import streamlit as st
import pandas as pd
import os
import sys
from utils import utils
import traceback

def app():
    """Responsavel por alterar o tipo das colunas 
    do DataFrame e remover dados nulos e linhas duplicadas.
    Aqui é criado o arquivo "main_data_change.csv" com as alterações
    feitas pelo usuário.
    """
    path_input = os.path.abspath(os.path.join(sys.path[0],'..' ,'data','process'))

    if not(os.path.isfile(os.path.join(path_input, 'main_data.csv'))):
        st.warning("Please upload data through `Upload Data` page!")
    else:
        # esta verificação é necessária para ler o dataframe final
        # e não ficar lendo o primeiro dataframe mesmo após a modificação
        if (os.path.isfile(os.path.join(path_input, 'main_data_change.csv'))):
            name_file = 'main_data_change.csv'
        else:
            name_file = 'main_data.csv'

        data = pd.read_csv(os.path.join(path_input, name_file), sep=',')

        # lista com os possíveis tipos a serem convertidos
        columns_type = ['int64', 'float64', 'category', 'object', 'datetime64[ns]']

        col1, col2 = st.columns(2)

        # Selecione o nome da coluna a ser convertida
        column_name = col1.selectbox("Select column name", options=data.columns.values)

        # Selecione o tipo da coluna
        column_type = col2.selectbox("Select column type", options=columns_type)

        if st.button("Change Colum"):
            try:
                # tenta fazer a alteração
                data[column_name] = data[column_name].astype(column_type)

                # grava em um arquivo metadata json o nome da coluna e o seu
                # respectivo tipo convertido

                # se o arquivo metadata não existir, faz criação dele
                if not(os.path.isfile(os.path.join(path_input, 'meta_data.json'))):
                    meta_data = dict()
                    meta_data[column_name] = column_type
                    
                    utils.write_json(os.path.join(path_input,'meta_data.json'),
                               meta_data)
                else:
                    # se o arquivo metadata ja existir, primeiro faz a leitura em seguida a
                    # escrita
                    meta_data = utils.read_json(os.path.join(path_input,'meta_data.json'))

                    meta_data[column_name] = column_type

                    utils.write_json(os.path.join(path_input,'meta_data.json'),
                               meta_data)

                data.to_csv(os.path.join(path_input, 'main_data_change.csv'), index=False, sep=',')
                st.info("Successfully converted column!")
            except Exception as e:
                st.error(f"Erro to convert! - {e}")
                st.error(f"{traceback.format_exc()}")

        # Pergunta ao usuário se ele deseja remover as colunas nulas
        dropna = st.radio("You want to remove null columns?",
                        ('No', 'Yes'))

        # Pergunta ao usuário se ele deseja remover as linhas duplicadas
        duplicated = st.radio("You want to remove the duplicate lines?",
                        ('No', 'Yes'))

        if st.button("Remove Data"):
            # executa os comando caso a opção for "Yes" e salva os 
            # todos os dados modificado em "main_data_change.csv"
            if dropna == "Yes":
                len_before = len(data)
                data = data.dropna()
                len_after = len(data)
                data.to_csv(os.path.join(path_input, 'main_data_change.csv'), index=False, sep=',')
                st.info(f"{len_before - len_after} null data successfully removed!")
            if duplicated == "Yes":
                date_duplicated = data.duplicated().sum()
                data[~data.duplicated()]
                st.info(f"{date_duplicated} duplicate data successfully removed!")
                data.to_csv(os.path.join(path_input, 'main_data_change.csv'), index=False, sep=',')