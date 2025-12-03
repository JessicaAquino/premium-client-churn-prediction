
import itertools
lista_1 = [5, 4, 3, 2, 1]

def combinacion_listas_total(lista_i) -> list:

    todas_combinaciones = [
        list(c)
        for r in range(1, len(lista_i) + 1)
        for c in itertools.combinations(lista_i, r)
    ]
    # todas_combinaciones = [l for l in todas_combinaciones if len(l)> 3]
    return todas_combinaciones


