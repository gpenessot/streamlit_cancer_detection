import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Choix du modèle &amp; optimisation"
sidebar_name = "Choix et optimisation du modèle"


def run():

    st.title(title)

    st.markdown("Méthodologiquement, après une première phase de découverte et d&#39;expérimentation de différents modèles et des différentes hypothèses évoquées ci-dessus, nous avons mis en place un protocole de test et de comparaison afin d&#39;évaluer l&#39;impact des différents critères sur la performance du modèle.")
    
    with st.expander("Choix techniques retenus"):
        st.markdown("""
Les modèles et les méthodes d&#39;entraînement/évaluation ont été codés en python sur la base de la librairie TF/Keras. Certains des membres de l&#39;équipe ont pu faire tourner les modèles en local sur des machines équipées en GPU, et d&#39;autres se sont appuyés sur Colab n&#39;ayant pas le matériel nécessaire.

Au niveau de la préparation des données, une ventilation 70/15/15 a été retenue entre les données d&#39;entraînement, de validation et de test. L&#39;augmentation de données a été réalisée avec la méthode ImageDataGenerator, et un batch size de 32 a été retenu dans la plupart des cas.

Pour l&#39;entraînement des modèles, un optimiseur Adam avec un learning rate de 5e-3 a été utilisé, une loss de crossentropy et une métrique d&#39;accuracy. Un maximum de 50 epoch a été retenu, mais des callback on été appliqués : un ReduceLROnPlateau avec une patience de 3 et un facteur de 0.3, et un EarlyStopping avec une patience de 8.

Dans les scénarii avec unfreeze et réentrainement de l&#39;ensemble du modèle, la seconde phase est identique à l&#39;exception du learning rate de départ de 5e-4.
""")
        
    st.markdown("""
## Critères analysés

Nous avons ainsi fait varier les critères suivants :
""")

    st.image(Image.open("assets/criteres.png"))
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Modèles", "Transfer Learning", "Taille", "Classification / Regression", "Augmentation de données"])

    
    with tab1:
        st.markdown("""Nous avons testé différents modèles standards, allant d&#39;un CNN simple jusqu&#39;à un EfficientNetB7 (avec une v1 sans dropout et une version v2 avec un dropout dans les couches de classification). Nous avons égalemewt retenu deux VGG et un Resnet50""")
        
    with tab2:
        st.markdown("""
En fonction des modèles, nous avons analysé 3 cas :

- &quot;from scratch&quot; : modèle sans transfer learning
- modèle avec transfer learning :
  - &quot;transfert&quot; : entraînement uniquement de la couche de classification
  - &quot;retrain&quot; : on repart du modèle précédent avec une première phase d&#39;entraînement des couches de classification (« transfert »), et on réentraîne l&#39;intégralité du modèle

Sur le modèle CustomCNN, seul le cas &quot;from scratch&quot; a naturellement été testé.

Le cas &quot;from scratch&quot; a également été étudié sur le VGG 16 (les modèles suivants étant trop profonds pour obtenir des résultats intéressants étant donné notre jeu de données réduit).

Sur la plupart des modèles, nous avons comparé les performances avec ou sans réentrainement.
""")

    with tab3:
        st.markdown("""
Comme évoqué précédemment, nous avons travaillé avec des images de taille de l&#39;ordre de 1350x1350. Cependant, les CNN courants sont calibrés pour des images de l&#39;ordre de 256x256 (et des modèles customisés acceptant en entrée du 1350x1350 auraient comporté un nombre de paramètres bien trop élevé à entraîner).

Cela étant, nous avons testé s&#39;il y avait un impact à réaliser l&#39;augmentation de données sur les images plein format (qui sont ensuite redimensionnées à l&#39;entrée du modèle) ou à réduire les images avant l&#39;augmentation de données (ce qui permet au passage un gain de la durée d&#39;entraînement).
""")

    with tab4:
        st.markdown("""
Étant donné le caractère ordinal de notre classification, nous avons également essayé d&#39;entraîner notre modèle avec une régression, puis de le réévaluer par classification.
""")
        
        
    with tab5:
        st.markdown("""
Etant donné notre jeu réduit d&#39;images d&#39;entraînement, et afin de réduire le surapprentissage, nous avons opté pour des augmentations de données.

Quatre scénarii ont été comparés :

- A0 : pas d&#39;augmentation de données
- A1 : flips + shifts (horizontaux et verticaux)
- A2 : A1 + rotations et zooms aléatoires
- A3 : rotation aléatoire + zoom calculé afin de ne pas avoir de &quot;blanc&quot; (consécutif à la rotation)
""")
        
    st.markdown("""
## Comparaison des résultats obtenus

Tableau général de l&#39;accuracy des modèles entraînés :
        """
    )

    st.image(Image.open("assets/resultats-general.png"))

    st.markdown("""
Pour chaque modèle, nous avons réalisé une évaluation sur un jeu de test distinct (mais identique à chaque fois) comportant 15% des données du jeu d&#39;origine.

Nous avons calculé pour chaque modèle l&#39;accuracy sur les 4 classes ainsi que le f1-score. A noter que ceux-ci sont systématiquement très proches.
""")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Modèle et transfer learning", "Taille de l'image", "Augmentation de données","Synthèse des critères"])
    
    with tab1:
     
        st.image(Image.open("assets/resultats-transfert.png"))

        st.markdown("""
    _Accuracy comparées entre les modèles pour une augmentation A2 et des images en 256x256_

    On note une amélioration des performances sur les modèles plus profonds, ainsi qu&#39;une très nette amélioration avec le réentraînement de l&#39;ensemble du modèle.
""")
    
    with tab2:
        st.image(Image.open("assets/resultats-taille.png"))
        st.markdown("A une exception près, on ne note pas d&#39;écart de performance entre les modèles entraînés avec les images plein format ou au format réduit.")

    with tab3:
        st.image(Image.open("assets/resultats-augmentation.png"))
        st.markdown("""
    _Images en 256x256_

Il est intéressant de noter que les performances des augmentations ne sont pas homogènes que l&#39;on soit en transfer learning avec ou sans réentrainement.

Dans le cas sans réentrainement, l&#39;impact de l&#39;augmentation de données est finalement faible voire contre-productif (pour le VGG 19 et l&#39;Efficient Net, les meilleures performances sont obtenues sans augmentation).

Par contre, avec réentrainement, l&#39;impact de l&#39;augmentation est significatif, et les meilleurs résultats sont obtenus avec le scénario A2.
""")
    with tab4:
        st.markdown("Globalement, on note une amélioration des performances avec des réseaux profonds pré-entraînés, avec réentrainement et une augmentation de données A2. L&#39;impact de la taille de l&#39;image en entrée est négligeable.")

    st.markdown("""
## Meilleur modèle trouvé

Notre meilleur modèle est ainsi un Efficient Net B2 sans dropout, avec réentraînement sur une augmentation A2
""")
    
    tab1, tab2, tab3 = st.tabs(["Courbe d'entrainement", "Performances sur le jeu de test","Notre modèle versus l'état de l'art"])
    
    with tab1:
        st.image(Image.open("assets/courbe-entrainement.png"))

        st.markdown("Au niveau de l&#39;entraînement du modèle, on note effectivement un saut de performance très conséquent au moment de l&#39;unfreeze de l&#39;ensemble du modèle (à l&#39;epoch 10), suivi d&#39;une convergence assez rapide sur le jeu de validation. A noter un overfit important, étant donné que l&#39;on attend quasiment une accuracy de 1 sur le jeu d&#39;entraînement (vs env. 0.85 sur le jeu de validation)")
        
    with tab2:    
        st.image(Image.open("assets/matrice-confusion.png"))
        st.markdown("On obtient une accuracy de **0.84** et un f1-score de 0.84 également. A noter que l&#39;accuracy binaire (0 vs 1/2/3) obtenue est de **0.9**.")

    with tab3:
        st.markdown("""   
Il est très compliqué de comparer les résultats de notre meilleur modèle par rapport à d&#39;autres modèles existants car peu de publications scientifiques existent sur cette thématique très précise du cancer de l&#39;utérus. Celui-ci est bien entendu largement étudié dans le monde, et a priori, surtout en Chine, mais la classification d&#39;image dans ce domaine est souvent liée à des images macroscopiques du col de l&#39;utérus ou encore et surtout à des images cytologiques (étalement de cellules suite à un frottis ou à une ponction tissulaire), ce qui est bien différent de notre problématique (étalement unicellulaire d&#39;un tissu complet suite à une biopsie) pour laquelle aucune étude n&#39;a pu être utilisée à des fins de comparaison dans ce travail. Voir ci-dessous pour des illustrations des articles les plus fréquemment rencontrés dans ce domaine.

Il n &#39;est pas rare d&#39;obtenir dans les études concernant les étalements cytologiques des accuracy allant de 82 à 86% pour les lésions de faible grade jusqu&#39;à 93-99% pour les lésions de haut grade. Les études les plus performantes montrent des résultats très impressionnants tels que, pour exemple : détection des atypies (= &quot;irrégularités&quot;) de l&#39;ordre de 100% et classification &quot;négative&quot; de lames classées &quot;lésions de haut grade&quot; de l&#39;ordre de 0%.

  """)