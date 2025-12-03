import random
from src.config import SEMILLAS

semillas_nuevas=SEMILLAS.copy()
random.seed(42)

def create_semilla(n_semillas:int):
    while len(semillas_nuevas) < n_semillas :
        sem_nueva=random.randint(1,999999)
        if sem_nueva not in semillas_nuevas:
            semillas_nuevas.append(sem_nueva)
    return(semillas_nuevas[:n_semillas])




