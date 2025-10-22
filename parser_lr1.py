from typing import Dict, List, Set, Tuple
from utils import parse_grammar, EPSILON, augment_grammar

__all__ = [
    "compute_first",
    "compute_follow",
    "analizar_gramatica",
    "analizar_y_augment",
]

# ---------------------------------------------------------------------------
# FIRST
# ---------------------------------------------------------------------------

def compute_first(grammar: Dict[str, List[List[str]]]) -> Dict[str, Set[str]]:
    # Inicialización: FIRST(A) = ∅
    first: Dict[str, Set[str]] = {A: set() for A in grammar.keys()}

    def _first_of_symbol(sym: str) -> Set[str]:
        """FIRST del símbolo: si es NT, usa first[sym], si es terminal → {sym}, si ε → {ε}."""
        if sym == EPSILON:
            return {EPSILON}
        if sym.isupper():  # No terminal
            if sym not in first:
                first[sym] = set()
            return first[sym]
        # Terminal
        return {sym}

    changed = True
    while changed:
        changed = False
        # Para cada A → α
        for A, prods in grammar.items():
            before_size = len(first[A])

            for prod in prods:
                nullable_prefix = True
                for Y in prod:
                    FY = _first_of_symbol(Y)
                    first[A].update(FY - {EPSILON})
                    if EPSILON not in FY:
                        nullable_prefix = False
                        break
                if nullable_prefix:
                    first[A].add(EPSILON)

            if len(first[A]) > before_size:
                changed = True

    return first


# ---------------------------------------------------------------------------
# FOLLOW
# ---------------------------------------------------------------------------

def compute_follow(grammar: Dict[str, List[List[str]]],
                   first: Dict[str, Set[str]],
                   start_symbol: str) -> Dict[str, Set[str]]:
    follow: Dict[str, Set[str]] = {A: set() for A in grammar.keys()}
    # Regla 3: $ en FOLLOW(S)
    if start_symbol in follow:
        follow[start_symbol].add("$")
    else:
        # Robustez si el start no estuvo en grammar por algún motivo externo.
        follow[start_symbol] = {"$"}

    def _first_of_string(symbols: List[str]) -> Set[str]:
        if not symbols:
            return {EPSILON}
        acc: Set[str] = set()
        nullable_prefix = True
        for s in symbols:
            if s.isupper():
                Fs = first.get(s, set())
            elif s == EPSILON:
                Fs = {EPSILON}
            else:
                Fs = {s}  # terminal
            acc |= (Fs - {EPSILON})
            if EPSILON not in Fs:
                nullable_prefix = False
                break
        if nullable_prefix:
            acc.add(EPSILON)
        return acc

    changed = True
    while changed:
        changed = False
        for A, prods in grammar.items():
            for prod in prods:
                # Recorre posiciones donde haya un no terminal B
                for i, B in enumerate(prod):
                    if not B.isupper():
                        continue
                    beta = prod[i+1:]  # secuencia después de B
                    first_beta = _first_of_string(beta)

                    before_size = len(follow[B])

                    # Regla 1: FIRST(β) - {ε} ⊆ FOLLOW(B)
                    follow[B] |= (first_beta - {EPSILON})

                    # Regla 2: si β es vacío o ε ∈ FIRST(β) → FOLLOW(A) ⊆ FOLLOW(B)
                    if not beta or EPSILON in first_beta:
                        follow[B] |= follow[A]

                    if len(follow[B]) > before_size:
                        changed = True

    return follow


# ---------------------------------------------------------------------------
# Front-ends "puros" para la app (sin Streamlit)
# ---------------------------------------------------------------------------

def analizar_gramatica(texto: str) -> Tuple[Dict, Dict, Dict, str, str]:
    grammar, terminales, no_terminales, start_symbol, error = parse_grammar(texto)
    if error:
        return {}, {}, {}, "", error

    first = compute_first(grammar)
    follow = compute_follow(grammar, first, start_symbol)
    return grammar, first, follow, start_symbol, ""


def analizar_y_augment(texto: str) -> Tuple[
    Dict, Dict, Dict, str,   # grammar, first, follow, start
    Dict, Dict, Dict, str,   # g_aug, first_aug, follow_aug, S_aug
    str                      # error
]:
    grammar, first, follow, start_symbol, error = analizar_gramatica(texto)
    if error:
        return {}, {}, {}, "", {}, {}, {}, "", error

    g_aug, S_aug = augment_grammar(grammar, start_symbol)
    first_aug = compute_first(g_aug)
    follow_aug = compute_follow(g_aug, first_aug, S_aug)

    return grammar, first, follow, start_symbol, g_aug, first_aug, follow_aug, S_aug, ""
