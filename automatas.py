from typing import Dict, List, Set, Tuple, Optional, Any
from utils import EPSILON
from collections import deque
import html

# Tipo de ítem LR(1): (A, tuple(rhs), dot_pos, lookahead)
Item = Tuple[str, Tuple[str, ...], int, str]

# Tipo de acción LR(1)
Action = Tuple[str, Optional[Tuple[str, Tuple[str, ...]]], Optional[int]]

# ===============================================================
# FUNCIONES AUXILIARES — GRAMÁTICA, FIRST, FOLLOW
# ===============================================================

def terminals(grammar: Dict[str, List[List[str]]]) -> Set[str]:
    """Devuelve los terminales de la gramática."""
    nts = set(grammar.keys())
    ts = set()
    for prods in grammar.values():
        for rhs in prods:
            for s in rhs:
                if s == EPSILON:
                    continue
                if s not in nts:
                    ts.add(s)
    return ts


def nonterminals(grammar: Dict[str, List[List[str]]]) -> Set[str]:
    """Devuelve los no terminales de la gramática."""
    return set(grammar.keys())


def compute_first_seq(seq: List[str],
                      first: Dict[str, Set[str]]) -> Set[str]:
    """Calcula FIRST(α) para una secuencia α de símbolos."""
    result: Set[str] = set()
    for X in seq:
        Fs = first.get(X, set())
        result |= (Fs - {EPSILON})
        if EPSILON not in Fs:
            return result
    result.add(EPSILON)
    return result


# ===============================================================
# AUMENTACIÓN DE GRAMÁTICA
# ===============================================================

def augment_grammar(grammar: Dict[str, List[List[str]]],
                    start_symbol: str) -> Tuple[Dict[str, List[List[str]]], str]:
    """Crea S' -> S y devuelve (gramática_aumentada, nuevo_start)."""
    new_start = start_symbol + "'"
    while new_start in grammar:
        new_start += "'"
    g2 = {A: [prod[:] for prod in prods] for A, prods in grammar.items()}
    g2[new_start] = [[start_symbol]]
    return g2, new_start


# ===============================================================
# FIRST(α) AUXILIAR
# ===============================================================

def first_of_sequence(seq: List[str],
                      first: Dict[str, Set[str]]) -> Set[str]:
    """FIRST(α) para secuencia α."""
    if not seq:
        return {EPSILON}
    acc: Set[str] = set()
    nullable = True
    for s in seq:
        Fs = first[s] if s.isupper() else ({s} if s != EPSILON else {EPSILON})
        acc |= (Fs - {EPSILON})
        if EPSILON not in Fs:
            nullable = False
            break
    if nullable:
        acc.add(EPSILON)
    return acc


# ===============================================================
# CLOSURE LR(1)
# ===============================================================

def closure(items: Set[Item],
            grammar: Dict[str, List[List[str]]],
            first: Dict[str, Set[str]]) -> Set[Item]:
    """CLOSURE(I) para LR(1)."""
    I = set(items)
    changed = True
    while changed:
        changed = False
        for (A, rhs, dot, la) in list(I):
            rhs_list = list(rhs)
            if dot < len(rhs_list):
                B = rhs_list[dot]
                if B.isupper():  # B es no terminal
                    beta = rhs_list[dot+1:]
                    la_set = first_of_sequence(beta + [la], first)
                    for prod in grammar.get(B, []):
                        for b in la_set:
                            new_item: Item = (B, tuple(prod), 0, b)
                            if new_item not in I:
                                I.add(new_item)
                                changed = True
    return I


# ===============================================================
# GOTO LR(1)
# ===============================================================

def goto(items: Set[Item],
         X: str,
         grammar: Dict[str, List[List[str]]],
         first: Dict[str, Set[str]]) -> Set[Item]:
    """GOTO(I, X) para LR(1)."""
    moved: Set[Item] = set()
    for (A, rhs, dot, la) in items:
        rhs_list = list(rhs)
        if dot < len(rhs_list) and rhs_list[dot] == X:
            moved.add((A, tuple(rhs_list), dot + 1, la))
    if not moved:
        return set()
    return closure(moved, grammar, first)


# ===============================================================
# CIERRE INICIAL Y COLECCIÓN CANÓNICA
# ===============================================================

def initial_closure_from_augmented(g_aug: Dict[str, List[List[str]]],
                                   S_aug: str,
                                   first_aug: Dict[str, Set[str]]) -> Set[Item]:
    """I0 = CLOSURE({ [S' -> · S, $] })"""
    seed: Item = (S_aug, tuple(g_aug[S_aug][0]), 0, "$")
    return closure({seed}, g_aug, first_aug)


def _grammar_symbols(grammar: Dict[str, List[List[str]]]) -> Set[str]:
    symbols: Set[str] = set(grammar.keys())
    for prods in grammar.values():
        for rhs in prods:
            for s in rhs:
                symbols.add(s)
    symbols.discard(EPSILON)
    return symbols


def _state_index(states: List[Set[Item]], J: Set[Item]) -> int:
    for i, S in enumerate(states):
        if S == J:
            return i
    return -1


def canonical_collection_from_augmented(
    g_aug: Dict[str, List[List[str]]],
    S_aug: str,
    first_aug: Dict[str, Set[str]]
) -> Tuple[List[Set[Item]], Dict[Tuple[int, str], int]]:
    """Devuelve (states, transitions) de la colección canónica LR(1)."""
    I0 = initial_closure_from_augmented(g_aug, S_aug, first_aug)
    states: List[Set[Item]] = [I0]
    transitions: Dict[Tuple[int, str], int] = {}
    symbols = _grammar_symbols(g_aug)

    changed = True
    while changed:
        changed = False
        for i, I in list(enumerate(states)):
            for X in symbols:
                J = goto(I, X, g_aug, first_aug)
                if not J:
                    continue
                j = _state_index(states, J)
                if j == -1:
                    states.append(J)
                    j = len(states) - 1
                    changed = True
                transitions[(i, X)] = j
    return states, transitions


# ===============================================================
# REPRESENTACIÓN Y VISUALIZACIÓN (DOT, TABLAS)
# ===============================================================

def _with_dot(rhs: List[str], dot: int) -> str:
    parts = []
    for i, s in enumerate(rhs):
        if i == dot:
            parts.append("·")
        parts.append(s)
    if dot == len(rhs):
        parts.append("·")
    return " ".join(parts)


def items_to_rows(items: Set[Item]) -> List[dict]:
    """Convierte un conjunto de ítems en filas (para tabla HTML/Streamlit)."""
    rows = []
    for idx, (A, rhs, dot, la) in enumerate(sorted(items, key=lambda x: (x[0], x[1], x[2], x[3]))):
        prod_str = _with_dot(list(rhs), dot)
        rows.append({
            "No.": idx,
            "A": A,
            "Producción": f"{A} -> {prod_str}",
            "Lookahead": la
        })
    return rows


def transitions_to_rows(transitions: Dict[Tuple[int, str], int]) -> List[dict]:
    return [{"Desde": i, "Símbolo": X, "Hacia": j} for (i, X), j in sorted(transitions.items())]

def _item_str(A: str, rhs: Tuple[str, ...], dot: int, la: str) -> str:
    """Representación legible de un ítem: A -> α · β , la"""
    parts = []
    for i, s in enumerate(rhs):
        if i == dot:
            parts.append("·")
        parts.append(s)
    if dot == len(rhs):
        parts.append("·")
    prod = " ".join(parts) if parts else "·"
    return f"{A} -> {prod} , {la}"

# Tipo de arista: (origen, destino, símbolo o None para ε)
Edge = Tuple[Item, Item, Optional[str]]

# ==========================================================
# Construcción del AFN LR(1): cada ítem es un nodo
# y las aristas ε provienen de las reglas de CLOSURE.
# ==========================================================
def build_lr1_item_graph(g_aug, S_aug, first_aug):
    nodes = []
    edges = []
    node_map = {}

    states, transitions = canonical_collection_from_augmented(g_aug, S_aug, first_aug)

    id_counter = 0
    for st_items in states:
        for item in st_items:
            A, rhs, dot, la = item
            label = f"{A} → {' '.join(list(rhs[:dot]) + ['·'] + list(rhs[dot:]))}, {la}"
            node_map[(A, tuple(rhs), dot, la)] = id_counter
            nodes.append((id_counter, label))
            id_counter += 1

    for st_items in states:
        for item in st_items:
            A, rhs, dot, la = item
            src = node_map[(A, tuple(rhs), dot, la)]

            if dot < len(rhs):
                B = rhs[dot]
                if B in g_aug:
                    for prod in g_aug[B]:
                        first_beta_a = set()
                        beta_a = list(rhs[dot + 1:]) + [la]
                        for sym in beta_a:
                            first_beta_a |= first_aug.get(sym, {sym})
                            if "ε" not in first_aug.get(sym, {}):
                                break
                        if "ε" in first_beta_a:
                            first_beta_a.remove("ε")
                        for b in first_beta_a:
                            tgt = node_map.get((B, tuple(prod), 0, b))
                            if tgt is not None:
                                edges.append((src, tgt, "ε"))

            if dot < len(rhs):
                X = rhs[dot]
                nxt = (A, tuple(rhs), dot + 1, la)
                if nxt in node_map:
                    edges.append((src, node_map[nxt], X))

    return nodes, edges



# ==========================================================
# Visualización del AFN LR(1)
# ==========================================================
def to_dot_lr1_afn(nodes, edges, S_aug):
    """
    Genera un grafo DOT para visualizar el AFN LR(1).
    Cada nodo representa un ítem LR(1) y cada arista representa
    una transición (ε o símbolo).
    """
    dot = ['digraph AFN_LR1 {']
    dot.append('  rankdir=LR;')
    dot.append('  node [shape=box, style="rounded,filled", fontname="Consolas", fontsize=10, fillcolor="#f8f9fa"];')
    dot.append('  edge [fontname="Consolas", fontsize=9];')

    # --- Nodos
    for nid, label in sorted(nodes, key=lambda x: x[0]):
        safe_label = label.replace('"', '\\"')
        if S_aug in label:
            dot.append(f'  N{nid} [label="{safe_label}", fillcolor="#d1e7dd", style="rounded,filled,bold"];')
        else:
            dot.append(f'  N{nid} [label="{safe_label}"];')

    # --- Aristas
    for src, dst, sym in edges:
        color = "black"
        style = "solid"
        if sym == "ε":
            color = "#6c757d"  # gris
            style = "dashed"
        elif sym.islower() or sym in {",", ";", "id", "int", "float"}:
            color = "#198754"  # verde para terminales
        else:
            color = "#0d6efd"  # azul para no terminales

        dot.append(f'  N{src} -> N{dst} [label="{sym}", color="{color}", fontcolor="{color}", style="{style}"];')

    dot.append('}')
    return "\n".join(dot)

# ===============================================================
# FOLLOW AUXILIAR (para reducciones ε)
# ===============================================================

def follow_set_of(A, g_aug, first_aug, S_aug, visited=None):
    """
    Calcula FOLLOW(A) con control de recursión para evitar ciclos infinitos.
    Maneja producciones ε y referencias mutuas.
    """
    if visited is None:
        visited = set()
    if A in visited:
        return set()  # corta recursión

    visited.add(A)
    follow = set()

    if A == S_aug:
        follow.add("$")

    for B, prods in g_aug.items():
        for prod in prods:
            for i, X in enumerate(prod):
                if X == A:
                    beta = prod[i + 1:] if i + 1 < len(prod) else []
                    if beta:
                        first_beta = compute_first_seq(beta, first_aug)
                        follow.update(first_beta - {EPSILON})
                        if EPSILON in first_beta:
                            follow.update(follow_set_of(B, g_aug, first_aug, S_aug, visited))
                    else:
                        if B != A:
                            follow.update(follow_set_of(B, g_aug, first_aug, S_aug, visited))
    return follow


# ===============================================================
# CONSTRUCCIÓN DE TABLAS ACTION / GOTO
# ===============================================================

def build_lr1_tables(g_aug, S_aug, first_aug):
    """Construye las tablas ACTION y GOTO para el analizador LR(1)."""
    states, transitions = canonical_collection_from_augmented(g_aug, S_aug, first_aug)
    ACTION = {}
    GOTO = {}
    conflicts = []

    for i, items in enumerate(states):
        for (A, rhs, dot, a) in items:
            # SHIFT
            if dot < len(rhs) and rhs[dot] in terminals(g_aug):
                X = rhs[dot]
                j = transitions.get((i, X))
                if j is not None:
                    ACTION.setdefault((i, X), []).append(("shift", None, j))

            # REDUCE / ACCEPT
            elif dot == len(rhs):
                if A == S_aug:
                    ACTION.setdefault((i, a), []).append(("accept", None, None))
                else:
                    prod = (A, rhs)
                    ACTION.setdefault((i, a), []).append(("reduce", prod, None))

        # GOTO
        for X in nonterminals(g_aug):
            j = transitions.get((i, X))
            if j is not None:
                GOTO[(i, X)] = j

    # Añadimos reducciones ε explícitas
    for i, items in enumerate(states):
        for (A, rhs, dot, a) in items:
            if rhs == (EPSILON,):
                for look in follow_set_of(A, g_aug, first_aug, S_aug):
                    ACTION.setdefault((i, look), []).append(("reduce", (A, tuple()), None))

    # Detectar conflictos
    for key, acts in ACTION.items():
        kinds = {k for k, _, _ in acts}
        if len(kinds) > 1:
            conflicts.append(f"Conflicto en estado {key[0]} con símbolo {key[1]}: {acts}")

    return states, transitions, ACTION, GOTO, conflicts

def to_dot_lr1(states: List[Set[Item]],
               transitions: Dict[Tuple[int, str], int],
               S_aug: str) -> str:
    """
    Genera el grafo DOT del autómata LR(1).
    Cada nodo muestra los ítems y los estados de reducción/aceptación.
    """
    import html
    lines = [
        "digraph LR1 {",
        "  rankdir=LR;",
        '  node [shape=box, fontname="Courier"];'
    ]

    reduce_states: Set[int] = set()
    accept_states: Set[int] = set()

    for i, st_items in enumerate(states):
        has_reduce = any(dot == len(rhs) for (_, rhs, dot, _) in st_items)
        if has_reduce:
            reduce_states.add(i)
        if any(A == S_aug and dot == len(rhs) and la == "$" for (A, rhs, dot, la) in st_items):
            accept_states.add(i)

    for i, st_items in enumerate(states):
        ordered = sorted(st_items, key=lambda it: (it[0] != S_aug, it[0], it[1], it[2], it[3]))
        lines_html = [f"<B>I{i}</B>"]
        for (A, rhs, dot, la) in ordered:
            prod = " ".join(list(rhs[:dot]) + ["·"] + list(rhs[dot:]))
            txt = f"{A} -> {prod} , {la}"
            lines_html.append(html.escape(txt))
        label = "<" + "<BR ALIGN='LEFT'/>".join(lines_html) + ">"
        per = 1
        if i in reduce_states:
            per = 2
        if i in accept_states:
            per = 3
        lines.append(f'  {i} [label={label}, peripheries={per}];')

    def _esc(s: str) -> str:
        return s.replace('"', r'\"')

    for (i, X), j in sorted(transitions.items(), key=lambda t: (t[0][0], t[0][1])):
        lines.append(f'  {i} -> {j} [label="{_esc(X)}"];')

    lines.append("}")
    return "\n".join(lines)

# ===============================================================
# SIMULACIÓN DEL PARSER LR(1)
# ===============================================================

def simulate_lr1(tokens: List[str],
                 ACTION: Dict[Tuple[int, str], List[Action]],
                 GOTO: Dict[Tuple[int, str], int]) -> Tuple[bool, List[dict], str]:
    """Simula el análisis LR(1) (pila de estados y símbolos)."""
    stack_states: List[int] = [0]
    stack_symbols: List[str] = []
    ip = 0
    trace: List[dict] = []

    def _action_str(acts: List[Action]) -> str:
        parts = []
        for (k, prod, j) in acts:
            if k == "shift":
                parts.append(f"shift {j}")
            elif k == "reduce":
                A, beta = prod  # type: ignore
                beta_str = " ".join(beta) if beta else EPSILON
                parts.append(f"reduce {A} -> {beta_str}")
            else:
                parts.append("accept")
        return " / ".join(parts)

    while True:
        a = tokens[ip] if ip < len(tokens) else "$"
        s = stack_states[-1]
        acts = ACTION.get((s, a), [])
        trace.append({
            "Pila(estados)": str(stack_states),
            "Pila(símbolos)": " ".join(stack_symbols) if stack_symbols else "·",
            "Entrada": " ".join(tokens[ip:]),
            "Acción": _action_str(acts) if acts else "— (error)"
        })

        if not acts:
            return False, trace, f"Error sintáctico en estado {s} con lookahead '{a}'."

        chosen = acts[0]
        kind, prod, j = chosen

        if kind == "shift":
            stack_symbols.append(a)
            stack_states.append(j if j is not None else -1)
            ip += 1
        elif kind == "reduce":
            A, beta = prod  # type: ignore
            k = 0 if (len(beta) == 1 and beta[0] == EPSILON) else len(beta)
            for _ in range(k):
                stack_symbols.pop()
                stack_states.pop()
            t = stack_states[-1]
            goto_state = GOTO.get((t, A))
            if goto_state is None:
                return False, trace, f"Falta GOTO({t}, {A}) durante la reducción."
            stack_symbols.append(A)
            stack_states.append(goto_state)
        else:  # accept
            trace[-1]["Acción"] = "accept"
            return True, trace, ""
