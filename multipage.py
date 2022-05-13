import streamlit as st
 
class MultiPage(): 
    """Responsavel por criar varias pagina e deixa-las 
        disponiveis em um selectbox
    """
    def __init__(self) -> None:
        self.pages = []
    
    def add_page(self, title: str, func) -> None: 
        """ Responsavel por adicionar um dicionario, contendo o titulo e a funcao
            que renderiza a pagina, no atributo self.pages

        Parameters
        ----------
        title : str
            Titulo da pagina
        func : function
            Funcao que renderiza a pagina
        """

        self.pages.append(
            {
                "title": title, 
                "function": func
            }
        )

    def run(self):
        """ Executa a funcao que renderiza cada pagina
        """
        page = st.sidebar.selectbox(
            'App Navigation', 
            self.pages, 
            format_func=lambda page: page['title']
        )

        # run the app function 
        page['function']()