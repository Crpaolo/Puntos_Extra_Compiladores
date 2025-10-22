import streamlit as st
import pandas as pd

from parser_lr1 import analizar_y_augment
from utils import limpiar_texto, tokenize_input , EPSILON
from automatas import (
    initial_closure_from_augmented,
    canonical_collection_from_augmented,
    items_to_rows,
    transitions_to_rows,
    to_dot_lr1,
    build_lr1_tables,
    simulate_lr1,
    to_dot_lr1_afn,
    build_lr1_item_graph
)


st.set_page_config(page_title="LR(1) - Colección, Tabla y Parser", page_icon="🧩", layout="wide")

# ---------- Session helpers ----------
def _store_results(**kwargs):
    st.session_state["lr1_results"] = kwargs

def _has_results() -> bool:
    return "lr1_results" in st.session_state

def _get_results():
    return st.session_state.get("lr1_results", {})

# ---------- App ----------
def app():
    st.title("Proyecto Compiladores: Analizador LR(1) (canónico)")
    st.caption("Ingresa una gramática en el formato: `S -> A a | b` (línea por producción). NT en MAYÚSCULAS.")

    with st.expander("Ejemplo rápido (copiar/pegar)", expanded=False):
        st.code(
            "Declaracion -> Tipo Var-list\n"
            "Tipo -> int | float\n"
            "Var-list -> id Var-tail\n"
            "Var-tail -> , Var-list | ε\n",
            language="none"
        )

    col1, col2 = st.columns([2, 1])
    with col1:
        gramatica_input = st.text_area("Gramática:", height=220, key="grammar_text")
    with col2:
        cadena_input = st.text_area("Cadena a analizar (tokens separados por espacio):",
                                    placeholder="int id , id", height=220)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("1) Analizar gramática", type="primary", use_container_width=True):
            if gramatica_input.strip() == "":
                st.warning("Por favor ingresa una gramática válida.")
            else:
                texto_limpio = limpiar_texto(gramatica_input)
                (
                    grammar, first, follow, start,
                    g_aug, first_aug, follow_aug, S_aug,
                    error
                ) = analizar_y_augment(texto_limpio)

                if error:
                    st.error(error)
                else:
                    # 1️⃣ Cierre inicial y colección
                    I0 = initial_closure_from_augmented(g_aug, S_aug, first_aug)
                    states, transitions = canonical_collection_from_augmented(g_aug, S_aug, first_aug)

                    # 2️⃣ Tablas LR(1)
                    states2, transitions2, ACTION, GOTO, conflicts = build_lr1_tables(g_aug, S_aug, first_aug)

                    # 3️⃣ AFN LR(1)
                    afn_nodes, afn_edges = build_lr1_item_graph(g_aug, S_aug, first_aug)

                    # 4️⃣ Guardar todo
                    _store_results(
                        grammar=grammar, first=first, follow=follow, start=start,
                        g_aug=g_aug, first_aug=first_aug, follow_aug=follow_aug, S_aug=S_aug,
                        I0=I0, states=states, transitions=transitions,
                        ACTION=ACTION, GOTO=GOTO, conflicts=conflicts,
                        afn_nodes=afn_nodes, afn_edges=afn_edges
                    )

                    st.success("Gramática analizada y AFN LR(1) generado correctamente ✅")


    with c2:
        if st.button("2) Construir tabla LR(1)", use_container_width=True):
            if not _has_results():
                st.warning("Primero analiza la gramática.")
            else:
                st.success("Tabla LR(1) construida (ver abajo).")

    with c3:
        if st.button("3) Simular parsing", use_container_width=True):
            if not _has_results():
                st.warning("Primero analiza la gramática.")
            else:
                res = _get_results()
                ACTION = res["ACTION"]
                GOTO = res["GOTO"]
                tokens = tokenize_input(cadena_input or "")
                ok, trace, err = simulate_lr1(tokens, ACTION, GOTO)
                st.session_state["trace"] = trace
                st.session_state["accepted"] = ok
                st.session_state["err"] = err

    if not _has_results():
        return

    res = _get_results()

    # ---- FIRST / FOLLOW (aumentada)
    st.subheader("FIRST y FOLLOW (gramática aumentada)")
    st.caption(f"Símbolo inicial aumentado: **{res['S_aug']}**")
    df_first_a = pd.DataFrame([{"No Terminal": nt, "FIRST": ", ".join(sorted(vals))}
                               for nt, vals in res["first_aug"].items()])
    df_follow_a = pd.DataFrame([{"No Terminal": nt, "FOLLOW": ", ".join(sorted(vals))}
                                for nt, vals in res["follow_aug"].items()])
    st.dataframe(pd.merge(df_first_a, df_follow_a, on="No Terminal", how="outer").fillna(""),
                 hide_index=True, use_container_width=True)
    
    # ---- CLOSURE(0)
    st.subheader("CLOSURE(0) — Ítems LR(1) con lookahead")
    st.dataframe(pd.DataFrame(items_to_rows(res["I0"])),
                 hide_index=True, use_container_width=True)

    # ---- Colección canónica (todos los estados)
    st.subheader("Colección canónica — Todos los estados (ítems)")
    all_rows = []
    for i, st_items in enumerate(res["states"]):
        rows = items_to_rows(st_items)
        for r in rows:
            r["Estado"] = f"I{i}"
        all_rows.extend(rows)
    df_all = pd.DataFrame(all_rows)[["Estado", "No.", "A", "Producción", "Lookahead"]]
    st.dataframe(df_all, hide_index=True, use_container_width=True)

    # ---- Transiciones
    with st.expander("Transiciones (i --X--> j)", expanded=False):
        st.dataframe(pd.DataFrame(transitions_to_rows(res["transitions"])),
                     hide_index=True, use_container_width=True)
    
    
    # ---- AFN LR(1) (ítems con ε) — nuevo
    st.subheader("AFN LR(1) — Ítems y ε-transiciones")
    st.caption("Cada nodo es un ítem LR(1); ε-aristas provienen del CLOSURE.")

    if "afn_nodes" in res and "afn_edges" in res:
        st.graphviz_chart(to_dot_lr1_afn(res["afn_nodes"], res["afn_edges"], res["S_aug"]),
                        use_container_width=True)
    else:
        st.info("⚙️ Aún no se ha generado el AFN LR(1). Presiona **Analizar gramática** para construirlo.")

    # ---- Grafo LR(1)
    st.subheader("AFD LR(1) — Visual")
    dot = to_dot_lr1(res["states"], res["transitions"], res["S_aug"])
    st.graphviz_chart(dot, use_container_width=True)

    # ---- Tabla LR(1): ACTION / GOTO + conflictos
    st.subheader("Tabla LR(1) — ACTION / GOTO")
    # ACTION a tabla legible
    action_rows = []
    for (i, a), acts in sorted(res["ACTION"].items(), key=lambda kv: (kv[0][0], kv[0][1])):
        action_rows.append({
            "Estado": i,
            "Símbolo": a,
            "Acción": " / ".join(
                ["shift " + str(j) if k == "shift"
                 else ("reduce " + p[0] + " -> " + (" ".join(p[1]) if p[1] else EPSILON)) if k == "reduce"
                 else "accept"
                 for (k, p, j) in acts]
            )
        })
    st.dataframe(pd.DataFrame(action_rows), hide_index=True, use_container_width=True)

    goto_rows = [{"Estado": i, "NoTerminal": A, "Ir a": j}
                 for (i, A), j in sorted(res["GOTO"].items(), key=lambda kv: (kv[0][0], kv[0][1]))]
    st.dataframe(pd.DataFrame(goto_rows), hide_index=True, use_container_width=True)

    if res["conflicts"]:
        st.error("⚠️ Conflictos detectados:")
        for c in res["conflicts"]:
            st.write("- " + c)
    else:
        st.success("Sin conflictos en ACTION/GOTO.")

    # ---- Simulación (si existe)
    if "trace" in st.session_state:
        st.subheader("Simulación LR(1) — Pila y acciones")
        ok = st.session_state["accepted"]
        trace = st.session_state["trace"]
        err = st.session_state["err"]
        st.dataframe(pd.DataFrame(trace), hide_index=True, use_container_width=True)
        if ok:
            st.success("✅ Cadena aceptada.")
        else:
            st.error(f"❌ Cadena rechazada. Detalle: {err}")
            

if __name__ == "__main__":
    app()
