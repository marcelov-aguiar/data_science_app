import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os, sys

def app():
    """É reponsável por ler o arquivo modificado pelo usuário, caso ele exista,
    em seguida dá varias opções de plot de gráficos para o usuário selecionar.
    Conforme o tipo de gráfico, aparece as opções para o usuário selecionar
    as colunas.
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
        
        col1, col2,col3 = st.columns(3)

        graph_type = col1.selectbox("Select the type of chart",
                                    options=['subplot',
                                             'distplot',
                                             'boxplot',
                                             'boxplot_group_by',
                                             'histogram',
                                             'pairplot'])
        columns_name_df = df_data.columns.values

        ####### secao para aparecer o selectbox conforme o tipo de gráfico #######
        if graph_type == "subplot":
            column_index = col2.selectbox("Select index name column",
                                          options=columns_name_df,
                                          index =0)

            # new values é para remover o nome da coluna que já foi selecionada
            new_values = list(set(columns_name_df) - set([column_index]))
            column_name_sub = col3.selectbox("Select y-axis column name",
                                             options=new_values)

        elif graph_type == 'distplot':
            column_name_dist = col2.selectbox("Select x-axis column name",
                                              options=columns_name_df)
        elif graph_type == 'boxplot':
            column_name_box = col2.selectbox("Select y-axis column name",
                                             options=columns_name_df)
        elif graph_type == 'boxplot_group_by':
            column_name_box_two_y = col2.selectbox("Select y-axis column name",
                                                   options=columns_name_df)
            new_values_two = list(set(columns_name_df) - set([column_name_box_two_y]))
            column_name_box_two_x = col3.selectbox("Select the columns to be grouped",
                                                   options=new_values_two)
        elif graph_type == 'histogram':
            column_name_hist = col2.selectbox("Select x-axis column name",
                                              options=columns_name_df)
        elif graph_type == 'pairplot':
            column_name_pair_y = col2.selectbox("Select y-axis column name",
                                                options=columns_name_df)
            column_name_pair_x = st.multiselect(
                                                'Select variables for scattering',
                                                list(columns_name_df),
                                                [columns_name_df[1],columns_name_df[2]])

        ####### plotagem dos graficos #######
        if st.button("Load graph"):

            if graph_type == 'subplot':
                df_data[column_index] = pd.to_datetime(df_data[column_index])

                df_data = df_data.sort_values(by=column_index)

                fig = px.line(df_data, 
                              x=column_index,
                              y=column_name_sub,
                              title=f'Graph of {column_name_sub}')

                st.plotly_chart(fig, use_container_width=True)
        
            elif graph_type == 'boxplot':
                fig = px.box(df_data, y=column_name_box, title=f'Graph of {column_name_box}')
                st.plotly_chart(fig)
            
            elif graph_type == 'distplot':
                ax = sns.displot(df_data, x=column_name_dist, kde=True)
                ax.figure.set_size_inches(12,6)
                plt.xlabel(column_name_dist, fontsize=16)
                plt.ylabel('Frequency', fontsize=16)
                plt.title(f"Frequency distribution of {column_name_dist}", fontsize=20)
                st.pyplot(ax)
                #'''
                #fig = ff.create_distplot([data[column_name]], [column_name])
                #st.plotly_chart(fig)
            elif graph_type == 'histogram':
                fig = px.histogram(df_data,
                                   x=column_name_hist,
                                   title=f'Histogram of {column_name_hist}')

                st.plotly_chart(fig)
            
            elif graph_type == 'boxplot_group_by':
                groups = df_data[column_name_box_two_x].unique()
                if len(groups) > 20:
                    st.error(f"Number of unique values of variable {column_name_box_two_x} " +
                    "is greater than 20")
                else:
                    fig = px.box(df_data, x=column_name_box_two_x, y=column_name_box_two_y)
                    st.plotly_chart(fig)

            elif graph_type == 'pairplot':
                ax = sns.pairplot(df_data,
                                  y_vars=[column_name_pair_y],
                                  x_vars=column_name_pair_x)
                ax.fig.suptitle('Dispersion between variables',fontsize=20,y=1.1)
                st.pyplot(ax)
