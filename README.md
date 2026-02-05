Racist Currents
===============

Framework d'analyse des biais latents dans les modèles de diffusion et VLMs. Le but principal ici est de **produire des visualisations** qui révèlent comment les modèles modernes réactivent des motifs visuels hérités de l'imagerie coloniale. La collecte/encodage est possible mais doit rester minimale; la priorité est le tooling léger, hackable et détournable pour explorer rapidement l'espace latent et les courants biaisés.

## Contenu
- Extraction/agrégation d'embeddings multi-modèles (CLIP, DINOv2) pour servir les viz.
- Hooks de capture pour trajectoires de débruitage (Stable Diffusion, Flux) — optionnels, plug-and-play.
- Analyses prêtes pour la viz: clustering stéréotypé, distances, champs de flux, attracteurs.
- Visualisations essentielles: espace latent, trajectoires, champs vectoriels, heatmaps; scripts prêts (`scripts/`).

## Démarrage rapide (focus viz)
1) Créer un environnement Python 3.10+.
2) `pip install -r requirements.txt`.
3) Placer des embeddings / trajectoires existants (ou générer vite fait) dans `data/historical/embeddings/` et `data/generated/trajectories/`.
4) Configurer `config.yaml` si besoin.
5) Lancer des viz directement :
   - `python scripts/viz_embedding_space.py --historical data/historical/embeddings/embeddings.npy --modern data/generated/embeddings.npy --out outputs/figures/embedding_space.png`
   - `python scripts/viz_flow_fields.py --trajectories data/generated/trajectories --poles outputs/stereotype_poles.npy --out outputs/figures/flow_field.png`
   - `python scripts/viz_trajectories.py --trajectories data/generated/trajectories --poles outputs/stereotype_poles.npy --out outputs/figures/trajectories.png`
    - Interface live : `streamlit run src/visualization/interface_app.py`

## Arborescence
Voir `config.yaml` et les modules dans `src/` pour la structure détaillée. Les scripts sous `scripts/` produisent les visualisations à partir de données déjà présentes.
