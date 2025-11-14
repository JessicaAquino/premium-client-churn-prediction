#loader.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)
def cargar_datos(path:str)->pd.DataFrame|None:
    '''
    carga un dataset desde 'path' y devuelve un pandas.Dataframe.
    '''
    logger.info(f"Cargando datos")
    try:
        df=pd.read_csv(path)
        logger.info(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.error(f"Error al cargar el dataset : {e}")
        #return None
        raise
    

    

