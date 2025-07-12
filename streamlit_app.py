import streamlit as st
import numpy as np
from ruby_quest import RubyQuestRL


MAX_HUNGER = 100

# Initialisation du jeu
if 'game' not in st.session_state:
    st.session_state.game = RubyQuestRL()

game = st.session_state.game

st.title("🍄 Quête du Ruby 🐿️")
st.markdown("Collecte le **Ruby 🍄** avant que la faim ne t'achève !")

# Affichage de la grille
def display_grid(grid):
    for row in grid:
        st.text(" ".join(row))

# Colonnes pour l'affichage
col1, col2 = st.columns([2, 1])

with col1:
    display_grid(game.get_grid_for_render())

with col2:
    # Barre de progression de la faim
    st.progress(game.hunger / MAX_HUNGER, text=f"Faim : {game.hunger}/{MAX_HUNGER}")

    if game.has_ruby:
        st.success("✅ Ruby Collecté !")

    # Boutons d'action
    st.markdown("### Mouvements")
    col_up = st.columns(3)
    with col_up[1]:
        if st.button("⬆️ Haut"):
            game.step(0)
            st.rerun()

    col_mid = st.columns(3)
    with col_mid[0]:
        if st.button("⬅️ Gauche"):
            game.step(3)
            st.rerun()
    with col_mid[1]:
        if st.button("🛑 Reset"):
            st.session_state.game = RubyQuestRL()
            st.rerun()
    with col_mid[2]:
        if st.button("➡️ Droite"):
            game.step(1)
            st.rerun()

    col_down = st.columns(3)
    with col_down[1]:
        if st.button("⬇️ Bas"):
            game.step(2)
            st.rerun()

# Fin du jeu
if game.done:
    if game.has_ruby:
        st.balloons()
        st.success("🎉 Tu as trouvé le Ruby ! Victoire !")
    else:
        st.error("💀 Game Over !")
    if st.button("🔄 Recommencer"):
        st.session_state.game = RubyQuestRL()
        st.rerun()