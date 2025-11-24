import polars as pl
import logging
import os

logger = logging.getLogger(__name__)

def ensure_dirs(*paths: str):
    """Create directories if they don't exist."""
    for path in paths:
        if not path:
            logger.warning("Skipped creating directory: path is None or empty")
            continue
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")

def load_data(path: str, format: str = "csv") -> pl.DataFrame | None:
    """
    Carga el archivo ubicado en 'path' y lo devuelve en un dataframe

    Parameters:
    -----------
    path : str
        Ruta del archivo a cargar
  
    Returns:
    --------
    pl.DataFrame
        DataFrame con los datos cargados
    """
    
    logger.info(f"Starting data loading from '{path}'")

    try:
        if format == "csv":
            df = pl.read_csv(path)
            logger.info(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        else:
            logger.error(f"Unsupported file format: '{format}'")
            return None
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
