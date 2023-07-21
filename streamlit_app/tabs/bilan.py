import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Bilan du projet"
sidebar_name = "Bilan"


def run():

    st.title(title)

    st.markdown(
        """
## Difficultés rencontrées lors du projet

Comme évoqué tout au long de notre rapport, nous avons rencontré plusieurs obstacles contribuant à la difficulté de ce projet :

- la grande complexité du domaine fonctionnel abordé
- la montée en compétence sur l&#39;apprentissage profond avec les outils et environnements associés
- la complexité de la tâche de classification due à sa divergence par rapport aux problématiques traitées habituellement par les CNNs
- la faible taille du jeu de données et son degré de qualité

Par ailleurs, sur un aspect pratique, les contraintes matérielles et de temps de calcul liées à l&#39;entraînement des modèles CNN impliquent des temps d&#39;attente très longs entre chaque essai, et entraînent une fragmentation du temps de travail. Dans un contexte de projet partagé avec nos activités professionnelles (et personnelles), cela limite la capacité à améliorer significativement la performance de nos modèles.

## Bilan et extensions possibles du projet

Ce projet a été exigeant mais extrêmement intéressant et formateur. Nous avons essayé d&#39;avoir une approche d&#39;analyse structurée pour comprendre le domaine puis proposer un modèle du meilleur niveau de performance possible en comprenant l&#39;impact de plusieurs facteurs sur les résultats obtenus.

Il existe de nombreuses pistes d&#39;améliorations que nous n&#39;avons pu explorer faute de temps.

- essayer d&#39;appliquer des espaces colorimétriques différents de l&#39;image initiale vers un encodage spécifique des couleurs liés au processus de coloration pour voir si cela permettrait un gain de performance
- entraîner notre meilleur modèle avec un descente de gradient customisée comme SAM pour essayer d&#39;améliorer la qualité de généralisation ([Sharpness Aware Minimization](https://paperswithcode.com/method/sharpness-aware-minimization))
- essayer des modèles composites avec système de voting sur plusieurs modèles complexes :
  - le gain d&#39;accuracy serait probablement potentiellement faible
  - mais cela pourrait donner une indication de niveau de confiance
- travailler sur une lame complète

  - Prendre une lame complète et la découper en n images pertinentes
  - Applique notre modèle pour avoir le grade sur chaque image
    - Avec histogramme pour étudier la répartition des grades sur une même lame
    - Déduire le grade global (grade maximal atteint)

  """)