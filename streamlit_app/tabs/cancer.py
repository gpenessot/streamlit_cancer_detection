import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

STREAMLIT_CLOUD_ROOT_PATH='/app/streamlit_cancer_detection/streamlit_app/'
title = "Cancer du col de l'utérus et analyse histologique"
sidebar_name = "Cancer et analyse histologique"


def run():

    st.title(title)

    st.markdown(
        """
## La cellule

La _cellule_ est l&#39;unité fondamentale de la vie, le corps humain en comprend entre 50 et 100 millions de millions. On trouve dans notre organisme environ 200 types de cellules aux formes, tailles, et fonctions très diverses.

Une cellule humaine comporte trois régions principales :

- la _membrane plasmique_ : elle forme la limite extérieure de la cellule.
- le _cytoplasme_ : liquide intracellulaire dans lequel baignent des _organites_ (petites structures assurant certaines fonctions à l&#39;intérieur de la cellule).
- le _noyau_ : un organite qui régit toutes les activités de la cellule.
  - il contient les _gènes_ qui contiennent notre patrimoine génétique possédant entre autres les instructions nécessaires à l&#39;élaboration des protéines de l&#39;organisme
  - il traite une extraordinaire complexité en faisant à la fois le travail d&#39;un ordinateur, d&#39;un architecte, d&#39;un chef de chantier et d&#39;un conseil d&#39;administration
"""
    )

    st.image(Image.open(os.path.join(STREAMLIT_CLOUD_ROOT_PATH, "assets/cellule.jpg")))

    st.markdown("""
## Les tissus

Un _tissu_ est un ensemble de cellules qui ont une structure semblable et qui remplissent des fonctions identiques ou analogues.
"""
    )

    st.image(Image.open(os.path.join(STREAMLIT_CLOUD_ROOT_PATH, "assets/tissus.jpg")))

    st.markdown("""
Nous voyons que quatre tissus primaires s&#39;enchevêtrent pour former la trame du corps humain, nous nous concentrerons par la suite sur le _tissu épithélial_ présent, entre-autres, au niveau du col de l&#39;utérus.

Ci-dessous un schéma d&#39;un _épithélium_ (tissu épithélial) sain :

- la partie haute de l&#39;épithélium est la partie externe, en contact avec la lumière vaginale (et donc, avec le mucus vaginal et l&#39;air extérieur).
- l&#39;épithélium est l&#39;&quot;empilement&quot; de cellules de l&#39;imagedont la structure globale va différer en fonction de leur positionnement, de leur composition interne, de leur forme, etc
- tout en bas, l&#39;épithélium repose sur la « _lame basale_ » qui sépare l&#39;épithélium du tissu conjonctif sous-jacent (en dessous). Ce dernier contient les vaisseaux sanguins, entre-autres, par lesquels sont acheminés les nutriments essentiels à son fonctionnement.
"""
    )

    st.image(Image.open(os.path.join(STREAMLIT_CLOUD_ROOT_PATH, "assets/Epithelium.jpg")))

    st.markdown("""
## Processus d&#39;obtention d&#39;image pour analyse des tissus

Parmi les différentes techniques de dépistage du cancer de l&#39;utérus, la _biopsie_ (prélèvement d&#39;un fragment de tissu présumé cancéreux) intra-utérine est recommandée pour caractériser précisément le _grade_ du cancer (nous expliquons plus loin comment le déterminer).

Le tissu prélevé est ensuite étalé sur une _lame_ en une couche unicellulaire.

Ces lames sont ensuite séchées et colorées. Le processus de coloration est un processus complexe de plusieurs étapes faisant appel à l&#39;humain. Le temps de trempage dans les solutions est un facteur important car ils vont impacter le degré de coloration donc la gamme et l&#39;intensité de couleur finales.

Les 2 colorants les plus importants utilisés lors de ce processus sont :

- l&#39;_hématoxyline_ : couleur bleu ou bleu-noir
- l&#39;_éosine_ : couleur orange-rosée
"""
    )

    st.image(Image.open(os.path.join(STREAMLIT_CLOUD_ROOT_PATH, "assets/coloration.png")))

    st.markdown("""
Ci-dessus :

- un exemple de séquence de coloration
- des lames trempant dans de l&#39;hématoxyline durant une de ces étapes.

Les lames sont ensuite photographiées et stockées dans des images de très large dimension : couramment aux alentours de 100.000 \* 100.000 pixels.
"""
    )

    st.image(Image.open(os.path.join(STREAMLIT_CLOUD_ROOT_PATH, "assets/cervix_lame_complete.jpg"), width=400))

    st.markdown("""
## Analyse du grade de cancer

Afin de déterminer la présence ou non d&#39;un cancer et le stade, le cas échéant, de celui-ci, on analyse des zones précises de tissus des lames photographiées. On appelle _histologie_ la discipline de la médecine qui étudie les tissus biologiques.

Attention à ne pas confondre deux notions distinctes :

- Le _stade_ d&#39;un cancer correspond à un état d&#39;avancement « géographique ».
- Le _grade_ correspond à l&#39;agressivité du tissu cancéreux.
"""
    )

    st.image(Image.open(os.path.join(STREAMLIT_CLOUD_ROOT_PATH,"assets/rappel_1.png")))

    st.markdown("""
Ci-dessus, une image en noir et blanc pour clarifier a quoi correspondent les 3 principales différentes zones épithéliales et leur intérêt respectif dans la gradation finale (+ ou -):

- les noyaux (zone foncées) - organites intracellulaires (+)
- les cytoplasmes (zones grisées) - &quot;vide&quot; intracellulaire (+)
- les zones vides sans éléments (zones blanches) - &quot;vide&quot; extracellulaire (-)

Pour déterminer le grade, nous devons :

- Repérer la zone d&#39;intérêt :
  - l&#39;épithélium
  - la lame basale (qui le sépare du tissu de support ou tissu conjonctif)
- Regarder le pourcentage de l&#39;épithélium en partant de la base (au-dessus de la lame basale) qui contient des cellules atypiques ou dont l&#39;agencement est atypique.
- Le grade retenu dans ce travail sera :
  - 0 si pas de cancer
  - entre 1 et 3 si détection d&#39;un cancer en fonction du ratio de cellule atypique dans l&#39;épithélium

Ci-dessous, des exemples d&#39;images labellisées (grossissement x40) :

- (A) **Grade 1**  : Cellules atypiques dans le tiers inférieur
- (B) **Grade 2**  : Cellules atypiques dans les deux tiers inférieurs de l&#39;épithélium.
- (C) **Grade 3**  : L&#39;entièreté de l&#39;épithélium est remplie de cellules atypiques,
- (D) **Grade « 0 »**  : Aucune cellule atypique, épithélium normal, pas de cancer
"""
    )

    st.image(Image.open(os.path.join(STREAMLIT_CLOUD_ROOT_PATH,"assets/rappel_4.png")))

    st.markdown("""
Il faut savoir que le système de notation des cancers est complexe et qu&#39;il varie en fonction du type de cancer, l&#39;approche ici décrite n&#39;est pas applicable en l&#39;état à tous les cancers.

## Complexité du domaine fonctionnel

Comme on peut le voir, le contexte fonctionnel de notre projet est très complexe car c&#39;est en fait un domaine de spécialité de la médecine. 
Ce travail a été fait en partenariat avec la société « Ummon HealthTech », société spécialisée dans l&#39;amélioration des soins médicaux par une amélioration de la détection des cancers, qui nous a fourni le jeu de données ainsi qu'une formation fonctionnelle pour introduire ces différents concepts.

Monter en compétence sur ce domaine a demandé un très grand investissement de notre part.

En termes d&#39;arrière-plan sur ce domaine, un seul participant du projet avait des connaissances préalables en médecine (vétérinaire), mais même pour lui, cela restait un domaine de spécialité sur lequel il n&#39;a pas été formé en détail et dont il n&#39;avait pas toutes les clés.

Que ce soit donc tant du côté scientifique (médecine) que du côté technique (apprentissage profond), nous partions donc de 0 et avons nécessité un long ramp-up. Ce projet était en effet en décalage complet avec nos métiers quotidiens sur tous ces aspects.
        """
    )


    
