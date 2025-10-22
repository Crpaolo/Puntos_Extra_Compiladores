#Opcional: helpers generales

def limpiar_texto(texto):
    # Función para limpiar o preprocesar texto de entrada
    return texto.strip()



# utils.py
from typing import Dict, List, Set, Tuple
import re

EPSILON = "ε"  # símbolo para epsilon

def parse_grammar(texto: str) -> Tuple[Dict[str, List[List[str]]], Set[str], Set[str], str, str]:
    grammar: Dict[str, List[List[str]]] = {}
    terminales: Set[str] = set()
    no_terminales: Set[str] = set()
    start_symbol = ""
    line_num = 0

    for raw in texto.splitlines():
        linea = raw.strip()
        line_num += 1
        if not linea:
            continue
        if '->' not in linea:
            return {}, set(), set(), "", f"Error línea {line_num}: Cada producción debe contener '->'"

        izquierda, derecha = linea.split('->', 1)
        izquierda = izquierda.strip()
        derecha = derecha.strip()

        # Validación: lado izquierdo un NT (en mayúsculas)
        if not izquierda or not izquierda.isupper():
            return {}, set(), set(), "", f"Error línea {line_num}: El no terminal a la izquierda debe ser MAYÚSCULA"

        if not start_symbol:
            start_symbol = izquierda

        no_terminales.add(izquierda)
        # Soportamos alternativas con '|'
        alternativas = [alt.strip() for alt in derecha.split('|')]

        for alt in alternativas:
            if alt == "" or alt.lower() == "epsilon" or alt == EPSILON:
                prod = [EPSILON]
            else:
                prod = alt.split()

            # Detectar NT/terminal
            for simbolo in prod:
                if simbolo == EPSILON:
                    continue
                if simbolo.isupper():
                    no_terminales.add(simbolo)
                else:
                    terminales.add(simbolo)

            grammar.setdefault(izquierda, []).append(prod)

    # Limpieza: ε no es terminal
    terminales.discard(EPSILON)
    return grammar, terminales, no_terminales, start_symbol, ""



def augment_grammar(grammar: Dict[str, List[List[str]]],
                    start_symbol: str) -> Tuple[Dict[str, List[List[str]]], str]:
    new_start = start_symbol + "'"
    while new_start in grammar:
        new_start += "'"
    g2 = {A: [prod[:] for prod in prods] for A, prods in grammar.items()}
    g2[new_start] = [[start_symbol]]
    return g2, new_start


import re

# ---------------------------------------------------------------------------
# TOKENIZACIÓN DE ENTRADA
# ---------------------------------------------------------------------------

def tokenize_input(texto: str):
    # Elimina espacios sobrantes y reemplaza comas pegadas (int id,id → int id , id)
    texto = texto.replace(",", " , ")
    # Separa por espacios y limpia
    tokens = [t for t in re.split(r"\s+", texto.strip()) if t]
    return tokens
