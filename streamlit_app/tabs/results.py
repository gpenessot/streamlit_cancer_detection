import streamlit as st
import pandas as pd
import numpy as np
from os.path import join 

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image


title = "Résultats"
sidebar_name = "Résultats"


df_test = pd.read_csv("assets/df_test.csv", dtype=str)
df_test["name"] = df_test.filename.str.split("/", expand=True)[2]
df_test.set_index("name", inplace=True)


@st.cache
def chargement_modele():
    return load_model( 'assets/EfficientNetB7v1-retrain-small-4-A2-model.h5')


def run(): 

    st.title(title)

    st.markdown("Evaluation du modèle sur les images test")
    
    model = chargement_modele()
    
    test_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1)

    test_generator = test_data_generator.flow_from_dataframe(df_test,
                                                        class_mode="sparse",
                                                        shuffle=False,
                                                    )
    
    predict = model.predict_generator(test_generator)  
    y_true = test_generator.classes 
    y_pred = np.argmax(predict, axis = 1)
    
    from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, accuracy_score, f1_score    
    
    cm = confusion_matrix(y_true, y_pred)
    st.write(cm)

    cr = classification_report(y_true, y_pred)
    st.write(cr)
    
    
#
#
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
#      
#def get_img_array(img_path, size):
#    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
#    array = tf.keras.preprocessing.image.img_to_array(img)
#    array = np.expand_dims(array, axis=0)
#    return array
#
#
#def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#    grad_model = tf.keras.models.Model(
#        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#    )
#
#    with tf.GradientTape() as tape:
#        last_conv_layer_output, preds = grad_model(img_array)
#        if pred_index is None:
#            pred_index = tf.argmax(preds[0])
#        class_channel = preds[:, pred_index]
#
#    grads = tape.gradient(class_channel, last_conv_layer_output)
#
#    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#
#    last_conv_layer_output = last_conv_layer_output[0]
#    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
#    heatmap = tf.squeeze(heatmap)
#
#    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#    return heatmap.numpy()
#
#def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
#    # Load the original image
#    img = tf.keras.preprocessing.image.load_img(img_path)
#    img = tf.keras.preprocessing.image.img_to_array(img)
#
#    # Rescale heatmap to a range 0-255
#    heatmap = np.uint8(255 * heatmap)
#
#    # Use jet colormap to colorize heatmap
#    jet = cm.get_cmap("jet")
#
#    # Use RGB values of the colormap
#    jet_colors = jet(np.arange(256))[:, :3]
#    jet_heatmap = jet_colors[heatmap]
#
#    # Create an image with RGB colorized heatmap
#    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
#    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
#    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
#
#    # Superimpose the heatmap on original image
#    superimposed_img = jet_heatmap * alpha + img
#    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
#
#    # Save the superimposed image
#    superimposed_img.save(cam_path)
#
#    # Display Grad CAM
#    # display(Image(cam_path))
#    st.image(cam_path)
#
#def run(): 
#
#    st.title(title)
#
#    st.markdown(
#        """
#        This is the third sample tab.
#        """
#    )
#    
#   
#    st.set_option('deprecation.showPyplotGlobalUse', False)
#    afficher_images((test_generator,))
#    st.pyplot()
#
#    # st.dataframe(df_test)
#    
#    for i, row in df_test.sample(n=1).iterrows():
#        st.text(row.filename)
#        st.image(row.filename)
#        
#    model = load_model( 'assets/EfficientNetB7v1-retrain-small-4-A2-model.h5')
#    last_conv_layer_name = "top_conv"
#
#        
#    data_visu = {'filename' : ['C19_B037_S21_6.jpeg', 'C02_B001_S21_3.jpeg', 'C05_B012_S11_0.jpeg', 'C17_B021_S17_6.jpeg', 
#                           'C07_B103_S21_0.jpeg', 'C01_B205_S01_4.jpeg', 'C12_B194_S12_4.jpeg', 
#                           'C04_B027_S04_1.jpeg', 'C12_B562_S12_1.jpeg', 'C01_B110_S01_1.jpeg', 'C06_B045_S21_1.jpeg',
#                           'C13_B241_S11_1.jpeg',  'C02_B221_S21_3.jpeg', 'C06_B027_S21_1.jpeg'],
#             'class' : [3, 1,3, 3,  1, 0, 3,  0, 1, 0, 0, 2,  0, 2]}
#
#    visu = pd.DataFrame(data_visu)
#    visu['filename']="assets/annotated_regions_test/"+visu['filename']
#    
#    for file in visu['filename'].values:
#        try:
#            img_array = preprocess_input(get_img_array(file, size=(256, 256)))
#            model.layers[-1].activation = None
#            preds = model.predict(img_array)
#            print(f"Image : {file.split('/')[-1]} | Classe prédite : {np.argmax(preds[0])} | Classe réelle : {visu[visu['filename']==file]['class'].values[0]}")
#            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
#            save_and_display_gradcam(file, heatmap, cam_path=join('assets/gradcams/gradcam_'+file.split('/')[-1]))
#        except:
#            st.error(file)
