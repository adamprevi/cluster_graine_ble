import streamlit as st
import numpy as np
import pandas as pd
import joblib


# ============================================================
#                  CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Classification des graines de blé",
    page_icon="🌾",
    layout="centered"
)


# ============================================================
#                  CHARGEMENT DU MODELE
# ============================================================

artefacts = joblib.load("modele_dbscan.joblib")

dbscan_model = artefacts["model"]
points_coeur = artefacts["points_coeur"]
labels_coeur = artefacts["labels_coeur"]
eps = artefacts["eps"]
min_samples = artefacts["min_samples"]
colonnes = artefacts["colonnes"]
valeurs_defaut = artefacts["valeurs_defaut"]


# ============================================================
#                  INTERPRETATION DES CLASSES
# ============================================================

noms_classes = {
    0: "Première catégorie",
    1: "Deuxième catégorie"
}


# ============================================================
#                  TITRE
# ============================================================

st.title("🌾 Classification des graines de blé")

st.write(
    "Application de classification basée sur "
    "l'algorithme DBSCAN."
)

st.info(
    f"Modèle utilisé : DBSCAN | "
    f"eps = {eps} | "
    f"min_samples = {min_samples}"
)


# ============================================================
#                  INFORMATIONS MODELE
# ============================================================

with st.expander("Informations sur le modèle"):

    st.write(
        f"**Nombre de points cœur :** "
        f"{len(points_coeur)}"
    )

    nombre_clusters = (
        len(set(dbscan_model.labels_))
        - (1 if -1 in dbscan_model.labels_ else 0)
    )

    st.write(
        f"**Nombre de clusters :** "
        f"{nombre_clusters}"
    )

    st.write(
        f"**Nombre de variables :** "
        f"{len(colonnes)}"
    )

    st.write("**Variables utilisées :**")

    st.write(colonnes)

    st.write("**Interprétation des classes :**")

    st.write("Classe 0 → Première catégorie")
    st.write("Classe 1 → Deuxième catégorie")
    st.write("Classe -1 → Graine atypique")


# ============================================================
#              FONCTION DE PREDICTION
# ============================================================

def predire_classe(nouvelle_graine):

    graine = np.array(
        nouvelle_graine,
        dtype=float
    ).reshape(1, -1)

    # Vérification du nombre de variables
    if graine.shape[1] != len(colonnes):

        raise ValueError(
            f"La graine doit contenir "
            f"{len(colonnes)} variables."
        )

    # Calcul des distances avec tous les points cœur
    distances = np.linalg.norm(
        points_coeur - graine,
        axis=1
    )

    # Point cœur le plus proche
    plus_proche = np.argmin(distances)

    # Distance minimale
    distance_min = distances[plus_proche]

    # Application de la règle DBSCAN
    if distance_min <= eps:

        classe = int(
            labels_coeur[plus_proche]
        )

    else:

        classe = -1

    return classe, distance_min


# ============================================================
#              FORMULAIRE DE SAISIE
# ============================================================

st.subheader(
    "Caractéristiques de la nouvelle graine"
)

valeurs = []

for col in colonnes:

    valeur = st.number_input(
        col,
        value=float(valeurs_defaut[col])
    )

    valeurs.append(valeur)


# ============================================================
#                  BOUTON PREDICTION
# ============================================================

if st.button(
    "🔍 Classifier la graine",
    use_container_width=True
):

    try:

        classe, distance = predire_classe(
            valeurs
        )

        st.divider()

        # ----------------------------------------------------
        # GRAINE ATYPIQUE
        # ----------------------------------------------------

        if classe == -1:

            st.error(
                "⚠️ Graine atypique"
            )

            st.write(
                "Cette graine ne se trouve pas "
                "dans le voisinage d'un point cœur "
                "du modèle DBSCAN."
            )

            st.write(
                f"Distance minimale : "
                f"**{distance:.3f}**"
            )

            st.write(
                f"Seuil DBSCAN (eps) : "
                f"**{eps:.3f}**"
            )

        # ----------------------------------------------------
        # GRAINE APPARTENANT A UNE CATEGORIE
        # ----------------------------------------------------

        else:

            nom_categorie = noms_classes.get(
                classe,
                f"Classe {classe}"
            )

            st.success(
                f"🌾 Cette graine appartient à la "
                f"**{nom_categorie}**."
            )

            st.write(
                f"Classe DBSCAN : **{classe}**"
            )

            st.write(
                f"Distance au point cœur le plus proche : "
                f"**{distance:.3f}**"
            )

            st.write(
                f"Seuil DBSCAN (eps) : "
                f"**{eps:.3f}**"
            )

            if distance <= eps:

                st.info(
                    "La distance est inférieure ou égale "
                    "à eps : la graine appartient à cette "
                    "catégorie."
                )


    except Exception as e:

        st.error(
            f"Erreur : {e}"
        )


# ============================================================
#                     PIED DE PAGE
# ============================================================

st.divider()

st.caption(
    "Projet No 2 — Clustering des graines de blé | "
    "DBSCAN"
)

st.caption(
    "Ousseynou DIOUF — Professeur : Abdoul Wahab DIALLO"
)
