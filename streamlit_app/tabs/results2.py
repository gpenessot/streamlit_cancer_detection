import streamlit as st
import pandas as pd
import numpy as np
from os.path import join 

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

st.set_option('deprecation.showPyplotGlobalUse', False)


import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image


title = "Prédictions du modèle"
sidebar_name = "Prédictions du modèle"

df_test = pd.read_csv("../assets/df_test.csv", dtype=str)
df_test["name"] = df_test.filename.str.split("/", expand=True)[2]
df_test.set_index("name", inplace=True)


@st.cache
def chargement_modele():
    return load_model( 'assets/EfficientNetB7v1-retrain-small-4-A2-model.h5')


#test_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1)
#
#test_generator = test_data_generator.flow_from_dataframe(df_test,
#                                                        class_mode="sparse",
#                                                        shuffle=True,
#                                                    )

#
#def afficher_images(generators):
#  plt.figure(figsize=(15,10))
#
#  for i, generator in enumerate(generators):
#    img, label = generator.next()
#    for j in range(18):
#      plt.subplot(3,6,i*6 + j +1)
#      plt.imshow(img[j]/255)
#      plt.title( str(int(label[j])))
#      plt.axis("off")
      
def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
     [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
     last_conv_layer_output, preds = grad_model(img_array)
     if pred_index is None:
         pred_index = tf.argmax(preds[0])
     class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    # display(Image(cam_path))
    st.image(cam_path)
    
def afficher_resultats(model,file,y):
    img_array = preprocess_input(get_img_array(file, size=(256, 256)))
    preds = model.predict(img_array)
    y_pred = np.argmax(preds[0])
    
    if str(y) == str(y_pred) : 
        st.success(f"Classe de l'image : {y} | Classe prédite : {y_pred}")
        st.balloons()
    else:
        st.error(f"Classe de l'image : {y} | Classe prédite : {y_pred}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Image")
        st.image(file)    
    
    with col2:
        st.subheader("Prédictions")
        plt.pie(preds[0], labels=["0","1","2","3"], autopct='%1.0f%%')
        st.pyplot( )
    
    
    with col3:
        st.subheader("Gradcam")
        
        with st.spinner(text="Calcul du gradcam"):
            last_conv_layer_name = "top_conv"
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            save_and_display_gradcam(file, heatmap, cam_path=join('assets/gradcams/gradcam_'+file.split('/')[-1]))
    

def run(): 

    st.title(title)

    
    st.markdown("### Exemples de classifications")
    st.markdown("""
    Des images gradcam ont pu être générées afin d&#39;analyser les zones sur lesquelles notre meilleur modèle construit son choix de classification finale.

Certaines images gradcam montrent des éléments qui nous semblent, à notre niveau de compréhension, pertinents. D&#39;autres le sont beaucoup moins. Les images bien classées sont très régulièrement le fait d&#39;une zone ciblée pertinente pour le réseau. Très subjectivement, il est amusant de constater que le réseau semble avoir les mêmes difficultés que notre œil humain non averti et, s&#39;il sait repérer rapidement les cas &quot;classiques&quot;, il perd son latin devant les cas qui nous semblent, à nous également, plus complexes.
    """)
    
    tab1, tab2, tab3 = st.tabs(["Exemples de bonnes classifications", "Exemples de mauvaises classifications", "Simulations sur les images test"])
    
    with tab1:

        col1, col2 = st.columns(2)
        col2.success("Classe réelle : 2, classe prédite : 2")
        col1.image("assets/gradcam1.png")
        col2.write("Ici, le modèle a parfaitement ciblé l&#39;épaisseur de l&#39;épithélium et bien classé l&#39;image.")   

        col1, col2 = st.columns(2)
        col1.image("assets/gradcam2.png")
        col2.success("Classe réelle : 1, classe prédite : 1")
        col2.write("A priori, l’épithélium semble ciblé, au moins partiellement. La petite fenêtre cible de la lame (pour rappel, les images sont des extraits de plus grandes images) ne donne pas d’indication sur le contexte général (orientation par rapport à la lumière utérine, etc). Un œil d’expert serait nécessaire dans ce cas pour juger du résultat gradcam.")   

        col1, col2 = st.columns(2)
        col2.success("Classe réelle : 3, classe prédite : 3")
        col1.image("assets/gradcam3.png")
        col2.write("Ce cas est particulièrement intéressant car la structure globale de l’image n’est pas classique. Les circonvolutions visibles sur l’image sont sans doute liées à la présence d’une masse tumorale (état très avancé du cancer). Pour notre œil non averti, il est très difficile de déduire de la pertinence du choix de zone du réseau ici, mais nous constatons que la classification est correcte.")   

    with tab2:

        col1, col2 = st.columns(2)
        col2.error("Classe réelle : 0, classe prédite : 1")
        col1.image("assets/gradcam4.png")
        col2.write("Ici, la zone choisie par le réseau est judicieuse puisqu’il s’agit bien, approximativement, de la trame épithéliale. Il est à déplorer que l’entièreté de l’épaisseur épithéliale n’ait pas été prise en compte.")   

        col1, col2 = st.columns(2)
        col2.error("Classe réelle : 0, classe prédite : 3")
        col1.image("assets/gradcam5.png")
        col2.write("Mauvaise surprise pour cette patiente qui pourrait se croire condamnée alors qu’aucune lésion n’est à déplorer sur cette image. Il est difficile de trouver une cohérence dans les “choix” du réseau sur cette image gradcam.") 

        col1, col2 = st.columns(2)
        col2.error("Classe réelle : 2, classe prédite : 0")
        col1.image("assets/gradcam6.png")
        col2.write("Il nous semble que le réseau avait pourtant bien ciblé la zone à explorer. Les cellules paraissent peu différenciées, le grade 2 ne semblait, ici, pourtant pas “compliqué” à attribuer. ") 

        col1, col2 = st.columns(2)
        col2.error("Classe réelle : 2, classe prédite : 0")
        col1.image("assets/gradcam7.png")
        col2.write("Clairement, le réseau a pris en compte de manière trop importante cette zone blanche, qui ne comporte aucune information pertinente dans le cadre de la classification de cancer.") 

        col1, col2 = st.columns(2)
        col2.error("Classe réelle : 2, classe prédite : 0")
        col1.image("assets/gradcam8.png")
        col2.write("Même remarque ici, où la zone blanche devrait être totalement délaissée par le réseau.") 
    
    with tab3:

        model = chargement_modele()

        option = st.selectbox('Selectionner une image',sorted(df_test.index))

        file = df_test.loc[option,"filename"]
        y = df_test.loc[option,"class"]

        afficher_resultats(model,file,y)

