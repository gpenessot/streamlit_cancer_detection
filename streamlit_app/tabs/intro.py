import streamlit as st


title = "Classification d’images de biopsie du col de l’utérus"
sidebar_name = "Introduction"


def run():

    st.image("https://storage.googleapis.com/kaggle-competitions/kaggle/11848/logos/header.png")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        Ces dernières années, les avancées de l’intelligence artificielle (IA) ont permis d’identifier de nouvelles pistes de recherche en travaillant sur les données de manière innovante. Ainsi, la médecine est un domaine qui a vu se développer des technologies d’IA à plusieurs niveaux, de manière exponentielle (diagnostics en imagerie, modélisations épidémiologiques, etc.).
        
Le diagnostic en imagerie est un champ d’application particulièrement intéressant dans lequel l’usage des réseaux de neurones convolutifs donnent des résultats très encourageants. Ces algorithmes peuvent ainsi s’appliquer sur des radiologies, des images IRM 3D mais aussi des lames histologiques qui sont à l’origine de notre projet. Ces dernières sont des images de tissus organiques (biopsies) dont l’étude des anomalies permet de diagnostiquer une pathologie.

L’objectif de notre projet est de classifier des images de biopsie de col de l’utérus pour détecter la présence d’un cancer et le cas échéant, d’en déterminer son grade. En effet, le cancer du col de l’utérus est l’un des rares cancers pour lequel le stade précurseur persiste de nombreuses années avant d’évoluer vers un authentique cancer invasif, ce qui peut offrir un temps suffisant pour le détecter et le traiter. Une détection précoce permet donc de sauver potentiellement des vies avant une dégradation plus avancée où les chances des patients se réduisent.

Ce projet a été possible grâce  à la société [**Ummon HealthTech**](https://www.ummonhealthtech.com/) qui nous a fourni un jeu de données d’histopathologie du col de l’utérus.
        """
    )
