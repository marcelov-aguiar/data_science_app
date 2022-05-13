import streamlit as st
import pandas as pd
import os
import sys
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns
import pickle
from typing import List
from utils import utils


def plot_scatter_performance(y_actual: np.array,
                             y_pred: np.array,
                             subtitle: str,
                             y_var: str,
                             metrics: pd.DataFrame) -> Figure:
    """Reponsável por fazer o plot do gráfico scatter com
    os valores preditos (`y_pred`) e os valores reais (`y_actual`).
    Faz o plot também de uma regra a 45º do eixo X, ou seja, os 
    valores preditos e reais deve ficar acima dessa reta para que
    a predição tenha uma boa acurácia. Além disso, é exibidos algumas
    métricas conforme o DataFrame `metrics`.

    Parameters
    ----------
    y_actual : np.array
        Valores reais da variável.
    y_pred : np.array
        Valores preditos pelo modelo.
    subtitle : str
        Legenda para identifica o gráfico.
    y_var : str
        Nome da variável que está sendo predita.
    metrics : pd.DataFrame
        DataFrame com as métricas calculadas anteriormente.
        As colunas possui o nome das métricas e as linhas o valor
        delas.


    Returns
    -------
    Figure
        Retorna uma figura, a ser plotada, com o gráfico
        do tipo scatter com os valores preditos e reais.
    """
    fig, ax = plt.subplots(figsize=(8,8))

    max_value = max([y_actual.max(),y_pred.max()]) + max([y_actual.max(),y_pred.max()])*0.05
    min_value = min([y_actual.min(),y_pred.min()]) - min([y_actual.min(),y_pred.min()])*0.05
    identity_line = np.linspace(min_value,max_value)

    ax.plot(identity_line,identity_line,'-', color='darkgrey', linewidth=3)
    sns.regplot(x=y_actual, y=y_pred, color='darkcyan', fit_reg=True, ax=ax, truncate=False)


    ax.set_ylim((min_value,max_value))
    ax.set_xlim((min_value,max_value))
    ax.set_ylabel(f'{y_var} predicted', fontsize=16)
    ax.set_xlabel(f'{y_var} real',fontsize=16)
    ax.set_title('Pred X Real - '+ subtitle, fontsize=18)
    
    if metrics is not None:
        test_text = metrics.iloc[0].to_string(float_format='{:,.2f}'.format)
        ax.text(0.1,0.8,test_text,color='indigo',bbox=dict(facecolor='none', edgecolor='indigo', boxstyle='round,pad=1'),
                transform = ax.transAxes, fontsize=9)
    return fig


def plot_coef(modelo: LinearRegression,
              columns_name: List[str]) -> Figure:
    """Reponsável por plotar um gráfico com o peso
    de cada variável em `columns_name` do `modelo`.

    Parameters
    ----------
    modelo : LinearRegression
        Objeto com o modelo linar treinado.
    columns_name : List[str]
        Lista com o nome das colunas.

    Returns
    -------
    Figure
        Retorna uma figura, a ser plotada, com o peso 
        de cada variável do modelo.
    """
    df_feature = pd.DataFrame()
    df_feature['Feature'] = columns_name
    df_feature["Coef"] = modelo.coef_
    df_feature.sort_values("Coef", inplace=True, ascending=True)

    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(10,5))
    plt.barh(df_feature["Feature"],df_feature["Coef"], color = 'darkblue')
    plt.barh(df_feature["Feature"][df_feature["Coef"]<0],
             df_feature["Coef"][df_feature["Coef"]<0], color='maroon')
    plt.title('Model coefficients')
    plt.xticks(rotation=90)
    
    return fig


def plot_all(y_var: str,
             y_train: pd.Series,
             y_pred_train: pd.Series,
             X_test: pd.Series,
             y_test: pd.Series,
             pipeline: Pipeline):
    """Responsáve por:
    - Calcular as métricas de treino e teste.
    - Plotar as métricas junto com o regplot do seaborn.
    - Plotar os resíduos.

    Parameters
    ----------
    y_var : str
        Nome da variável a ser predita.
    y_train : pd.Series
        Dados de treino da variável predita.
    y_pred_train : pd.Series
        Dados preditos da variável de treino.
    X_test : pd.Series
        Dados de entrada do teste para predição.
    y_test : pd.Series
        Dados de teste da variável predita.
    pipeline : Pipeline
        Pipeline sklearn contendo o modelo treinado.
    """
    # Plot dos dados de treino
    st.markdown("#### These are the training metrics")
    col1, col2, col3, col4 = st.columns(4)
    R2 = r2_score(y_train, y_pred_train).round(2)
    MAE = mean_absolute_error(y_train, y_pred_train).round(2)
    RMSE = mean_squared_error(y_train, y_pred_train, squared=False).round(2)
    MAPE = mean_absolute_percentage_error(y_train, y_pred_train).round(2)
    col1.metric(label="R2", value=R2)
    col2.metric(label="MAE", value=MAE)
    col3.metric(label="RMSE", value=RMSE)
    col4.metric(label="MAPE", value=MAPE)

    metrics = pd.DataFrame(columns=['MAE','RMSE','R2', 'MAPE'], index=range(0,1))
    metrics['MAE'][0] = MAE
    metrics['MAPE'][0] = MAPE
    metrics['RMSE'][0] = RMSE
    metrics['R2'][0] = R2
    fig = plot_scatter_performance(y_train,y_pred_train,'train', y_var, metrics)
    st.pyplot(fig)

    # Plot dos dados de teste
    st.markdown("#### These are the test metrics")
    y_pred_test = pipeline.predict(X_test)
    col1, col2, col3, col4 = st.columns(4)
    R2 = r2_score(y_test, y_pred_test).round(2)
    MAE = mean_absolute_error(y_test, y_pred_test).round(2)
    RMSE = mean_squared_error(y_test, y_pred_test, squared=False).round(2)
    MAPE = mean_absolute_percentage_error(y_test, y_pred_test).round(2)
    col1.metric(label="R2", value=R2)
    col2.metric(label="MAE", value=MAE)
    col3.metric(label="RMSE", value=RMSE)
    col4.metric(label="MAPE", value=MAPE)

    metrics = pd.DataFrame(columns=['MAE','RMSE','R2', 'MAPE'], index=range(0,1))
    metrics['MAE'][0] = MAE
    metrics['MAPE'][0] = MAPE
    metrics['RMSE'][0] = RMSE
    metrics['R2'][0] = R2
    fig = plot_scatter_performance(y_train,y_pred_train,'test', y_var, metrics)
    st.pyplot(fig)

    # Faz o plot dos resíduos
    st.markdown("#### Residual plot - Test")
    fig, ax = plt.subplots(figsize=(15,6))
    sns.residplot(x=y_pred_test, y=y_test)
    st.pyplot(fig)


def save_info_model(path_input: str,
                    X_var: List[str],
                    y_var: str,
                    pipeline: Pipeline,
                    path_deploy: str):
    """Salva o modelo e as informações sobre ele.

    Parameters
    ----------
    path_input : str
        Path onde se localiza o meta data vindo dos outros
        processamento.
    X_var : List[str]
        Lista com o nome das variável de entrada para o modelo.
    y_var : str
        Nome da variável de saída do modelo
    pipeline : Pipeline
        Pipeline sklearn contendo o modelo treinado.
    path_deploy : str
        Path de saída onde será salvo o modelo (serialização).
    """
    # grava em um arquivo metadata json o nome das variáveis
    # usada no treino do modelo

    # se o arquivo metadata não existir, faz criação dele
    if not(os.path.isfile(os.path.join(path_input, 'meta_data.json'))):
        meta_data = dict()
        meta_data['training_columns_name'] = X_var
        
        utils.write_json(os.path.join(path_input,'meta_data.json'),
                   meta_data)
    else:
        # se o arquivo metadata ja existir, primeiro faz a leitura em seguida a
        # escrita
        meta_data = utils.read_json(os.path.join(path_input,'meta_data.json'))

        meta_data['training_column_name'] = X_var
        meta_data['y_column_name'] = y_var

        utils.write_json(os.path.join(path_input,'meta_data.json'),
                   meta_data)
     
    pickle.dump(pipeline, open(path_deploy, 'wb'))


def app():
    """Reponsável por:
    - Perguntar ao usuário o nome da variável a ser preditida.
    - Permitir ao usuário selecionar as variáveis de entrada do modelo.
    - Permitir ao usuário selecionar o tipo de modelo a ser treinado e testado.
    - Permitir ao usuário selecionar o tamanho dos dados de teste.
    - Permitir ao usuário escolher se deseja usar StandardScaler.
    - Executa o treino e o teste do modelo exibindo as métricas de avaliação dele.
    """
    path_input = os.path.abspath(os.path.join(sys.path[0],'..' ,'data','process'))

    if not(os.path.isfile(os.path.join(path_input, 'main_data.csv'))):
        st.warning("Please upload data through `Upload Data` page!")
    else:
        if (os.path.isfile(os.path.join(path_input, 'main_data_change.csv'))):
            # Dá preferência em ler o arquivo que o usuário modificou na etapa anterior
            name_file = 'main_data_change.csv'
        else:
            # Caso usuário não fez nenhuma modificação, faz a leitura do arquivo original
            name_file = 'main_data.csv'

        df_data = pd.read_csv(os.path.join(path_input, name_file), sep=',')
        
        columns_name = df_data.columns.values

        col1, col2 = st.columns(2)

        # obtém o nome da variável a ser preditva
        y_var = col1.selectbox("Select the variable to be predicted (y)",
                               options=columns_name)

        # obtem as variáveis de entrada do modelo
        # exclui a variável selecionada anteiormente `y_var`
        options_X = list(set(columns_name) - set(y_var))
        X_var = col2.multiselect("Select the variables to be used for prediction (X)",
                                 options=options_X)

        # verifica se o tamanho de x não é zero
        x_var_is_empty = False
        if len(X_var) == 0:
            st.error("You have to put in some X variable and it cannot be left empty.")
            x_var_is_empty = True

        # Verifica se a variável a ser predita está na lista
        # das variáveis que são entrada do modelo
        y_var_in_x_var = False
        if y_var in X_var:
            st.error("Warning! Y variable cannot be present in your X-variable.")
            y_var_in_x_var = True
        
        # Lista com o nome dos modelos que poderão ser treinados
        models_names = ['LinearRegression',
                        'LogisticRegression',
                        'RandomForestRegressor']

        # Usuário poderá selecionar o modelo a ser treinado
        model_select = st.selectbox("Select the model for training the data.",
                               options=models_names)

        df_X = df_data[X_var]
        df_y = df_data[y_var]

        # Usuário poderá escolher o tamanho dos dados de teste
        test_size = st.slider('What is the size of the test data?',
                              0.0, 1.0, 0.3)
        
        # Usuário poderá usar StandardScaler nos dados
        has_scaler = False
        has_scaler = st.radio("Do you want to use StandardScaler?",
                        ('True', 'False'))
        
        if has_scaler == 'True':
            has_scaler = True
        else:
            has_scaler = False
        
        if model_select == 'RandomForestRegressor':
            col1, col2, col3 = st.columns(3)
            n_estimators = col1.number_input('Step size for n_estimators',10)
            criterion = col2.selectbox('criterion',('squared_error', 'absolute_error'))
            max_depth = int(col3.number_input('Maximum depth of the tree'))
            #col1, col2, col3 = st.columns(3)
            min_samples_split = \
                col1.number_input('Minimum number of samples required',2)
            n_jobs = int(col2.number_input('Number of jobs to run in parallel'))

            if (n_jobs == 0):
                n_jobs = None
            if (max_depth == 0):
                max_depth = None

        if st.button("Training and save model") and (not x_var_is_empty) and (not y_var_in_x_var):
            X_train, X_test, y_train, y_test = train_test_split(df_X,
                                                                df_y,
                                                                test_size=test_size,
                                                                random_state=42)
            
            if model_select == 'LinearRegression':
                # Ajuste do modelo
                model = LinearRegression()
                if has_scaler:
                    scaler = StandardScaler()
                    pipeline = Pipeline([('scaler', scaler),
                                         ('regressor', model)])
                else:
                    pipeline = Pipeline([('regressor', model)])
                pipeline.fit(X_train, y_train)
                y_pred_train = pipeline.predict(X_train)

                plot_all(y_var, y_train, y_pred_train, X_test, y_test, pipeline)

                st.markdown("#### Importance of variables")
                fig = plot_coef(pipeline['regressor'], X_var)
                st.pyplot(fig)

                # Salva o modelo em um arquivo binário
                base_path = os.path.dirname(os.path.realpath(__file__))
                path_deploy = os.path.abspath(os.path.join(base_path, '..',
                                                           'deploy',
                                                           'Model.m'))

                save_info_model(path_input, X_var, y_var, pipeline, path_deploy)

            if model_select == 'LogisticRegression':
                # Ajuste do modelo
                model = LogisticRegression()
                if has_scaler:
                    scaler = StandardScaler()
                    pipeline = Pipeline([('scaler', scaler),
                                         ('regressor', model)])
                else:
                    pipeline = Pipeline([('regressor', model)])
                pipeline.fit(X_train, y_train)
                y_pred_train = pipeline.predict(X_train)

                plot_all(y_var, y_train, y_pred_train, X_test, y_test, pipeline)

                # Salva o modelo em um arquivo binário
                base_path = os.path.dirname(os.path.realpath(__file__))
                path_deploy = os.path.abspath(os.path.join(base_path, '..',
                                                           'deploy',
                                                           'Model.m'))

                save_info_model(path_input, X_var, y_var, pipeline, path_deploy)
            
            if model_select == 'RandomForestRegressor':
                # Ajuste do modelo
                
                model = RandomForestRegressor(n_estimators = n_estimators,
                                              criterion=criterion,
                                              max_depth=max_depth,
                                              min_samples_split=min_samples_split,
                                              n_jobs=n_jobs,
                                              random_state = 42)
                if has_scaler:
                    scaler = StandardScaler()
                    pipeline = Pipeline([('scaler', scaler),
                                         ('regressor', model)])
                else:
                    pipeline = Pipeline([('regressor', model)])
                pipeline.fit(X_train, y_train)
                y_pred_train = pipeline.predict(X_train)

                plot_all(y_var, y_train, y_pred_train, X_test, y_test, pipeline)

                st.markdown("#### Importance of variables")
                fig, ax = plt.subplots()
                importances = pipeline['regressor'].feature_importances_
                forest_importances = pd.Series(importances, index=X_var)
                forest_importances.plot.bar(ax=ax)
                #ax.set_title("Feature importances")
                #ax.set_ylabel("Mean decrease in impurity")
                st.pyplot(fig)

                # Salva o modelo em um arquivo binário
                base_path = os.path.dirname(os.path.realpath(__file__))
                path_deploy = os.path.abspath(os.path.join(base_path, '..',
                                                           'deploy',
                                                           'Model.m'))

                save_info_model(path_input, X_var, y_var, pipeline, path_deploy)
