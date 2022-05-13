import json

def read_json(path: str) -> dict:
    """Responsavel por ler um arquivo
    json dado o seu path

    Parameters
    ----------
    path : str
        Caminho do arquivo json

    Returns
    -------
    dict
        Retorno o arquivo json em um dicionario
    """
    with open(path) as fp:
        meta_data = json.load(fp)
    return meta_data

def write_json(path: str, data: dict) -> None:
    """Responsavel por escrever o dado armazenado
    no argumento data em um arquivo json

    Parameters
    ----------
    path : str
        Caminho onde o arquivo vai ser salvo
    data : dict
        Dado a ser escrito no arquivo json
    """
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=4)