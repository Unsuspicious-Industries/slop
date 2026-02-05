import argparse
from pathlib import Path

import streamlit as st
import numpy as np

from .interface_utils import (
    load_embeddings,
    load_labels,
    load_poles,
    load_trajectories,
    render_embedding_space,
    render_trajectories,
    render_flow_field,
    render_density,
)


def sidebar_inputs():
    st.sidebar.header("Données")
    hist_path = st.sidebar.text_input("Embeddings historiques (.npy)", "data/historical/embeddings/embeddings.npy")
    modern_path = st.sidebar.text_input("Embeddings modernes (.npy)", "data/generated/embeddings.npy")
    traj_dir = st.sidebar.text_input("Trajectoires (.npy dir)", "data/generated/trajectories")
    poles_path = st.sidebar.text_input("Pôles stéréotypés (.npy)", "outputs/stereotype_poles.npy")
    labels_path = st.sidebar.text_input("Labels / mask (.npy)", "outputs/labels.npy")
    resolution = st.sidebar.slider("Flow resolution", 10, 120, 50, step=5)
    radius = st.sidebar.slider("Flow radius", 0.1, 2.0, 0.5, step=0.1)
    return hist_path, modern_path, traj_dir, poles_path, labels_path, resolution, radius


def main():
    st.title("Racist Currents – Visual Explorer")
    st.markdown("Exploration rapide des relations entre embeddings, trajectoires et pôles stéréotypés.")

    hist_path, modern_path, traj_dir, poles_path, labels_path, resolution, radius = sidebar_inputs()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Charger embeddings"):
            try:
                historical = load_embeddings(Path(hist_path))
                modern = load_embeddings(Path(modern_path))
                fig = render_embedding_space(historical, modern)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur embeddings: {e}")

    with col2:
        if st.button("Charger densité stéréotypes"):
            try:
                embs = load_embeddings(Path(hist_path))
                labels = load_labels(Path(labels_path))
                mask = labels >= 0
                fig = render_density(embs, mask)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur densité: {e}")

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Trajectoires"):
            try:
                trajectories = load_trajectories(Path(traj_dir))
                poles = load_poles(Path(poles_path)) if Path(poles_path).exists() else None
                fig = render_trajectories(trajectories, poles)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur traj: {e}")

    with col4:
        if st.button("Flow Field"):
            try:
                trajectories = load_trajectories(Path(traj_dir))
                poles = load_poles(Path(poles_path)) if Path(poles_path).exists() else None
                fig = render_flow_field(trajectories, poles, resolution=resolution, radius=radius)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur flow: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    # Streamlit expects `streamlit run`, but keeping argparse for clarity if reused.
    main()
