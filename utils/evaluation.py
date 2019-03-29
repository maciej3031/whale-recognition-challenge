import os
import random
from itertools import combinations, product

import keras.backend as K
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.preprocessing import image
from scipy.spatial.distance import euclidean, cdist
from sklearn.manifold import TSNE
from termcolor import colored
from tqdm import tqdm

from config import DATA_DIR
from utils.utils import load_image, preprocess_input, prepare_tensor_from_image, plot_side_by_side
import seaborn as sns


class Evaluation:
    def __init__(self, train_conf):
        self.input_shape = train_conf.input_shape
        self.model_name = train_conf.model_name
        self.encoder = train_conf.encoder
        self.keras_model = train_conf.keras_model

        
class VisualizeEmbeddings(Evaluation):
    def get_embeddings(self):
        all_unique_whales = pd.read_csv(os.path.join(DATA_DIR, 'validation_all.csv'))
        selected_whales_ids = all_unique_whales.groupby('Id')
        selected_whales_ids_with_most_occurances = list(selected_whales_ids.count().sort_values('Image', ascending=False)[:15].index)
        
        selected_whales = all_unique_whales[all_unique_whales['Id'].isin(selected_whales_ids_with_most_occurances)]
        selected_whales = selected_whales[selected_whales['Id'] != 'new_whale']
        
        imgs = []
        classes = []
        classes_names = {}
        for image_name, class_name in selected_whales.values:
            img = load_image(image_name, input_shape=self.input_shape, mode='train')
            imgs.append(img)
            if class_name not in classes_names.keys():
                new_class = len(classes_names.keys()) + 1
                classes_names[class_name] = new_class
                classes.append(new_class)
            else:
                classes.append(classes_names[class_name])
                
            if len(classes_names.keys()) == 20:
                break

        tensor = np.stack(imgs)
        preprocessed_tensor = preprocess_input(tensor, self.model_name)
        embeddings = self.encoder.predict(preprocessed_tensor)
        return embeddings, np.array(classes), classes_names


    @staticmethod
    def reduce_features_dimensions(preprocessed_features, tsne_perplexity=15):
        tsne = TSNE(n_components=2, verbose=0, perplexity=tsne_perplexity)
        tsne_results = tsne.fit_transform(preprocessed_features)
        return tsne_results

    def visualize(self, tsne_perplexity=15):
        embeddings, classes, classes_names = self.get_embeddings()
        tsne_results = self.reduce_features_dimensions(embeddings, tsne_perplexity)

        for name, val in classes_names.items():
            mask = classes == val
            plt.plot(tsne_results[:, 0][mask], tsne_results[:, 1][mask], 'o', label=name)
        plt.legend()
        fig = plt.figure(1)
        fig.set_size_inches(14, 12)

        
class Scores(Evaluation):
    def get_scores(self, positive, negative):
        scores = []
        for array in [positive, negative]:
            A_imgs = []
            B_imgs = []
            info = []
            for (image_A_name, image_B_name, label) in tqdm(array.values):
                img_A = load_image(image_A_name, input_shape=self.input_shape, mode='train')
                img_B = load_image(image_B_name, input_shape=self.input_shape, mode='train')
                A_imgs.append(img_A)
                B_imgs.append(img_B)
                
                info.append((image_A_name, image_B_name, label))

            A_tensor = np.stack(A_imgs)
            B_tensor = np.stack(B_imgs)
    
            A_embedding = self.encoder.predict(A_tensor)
            B_embedding = self.encoder.predict(B_tensor)
            distances = cdist(A_embedding, B_embedding, metric='euclidean').diagonal()

            scores.extend(zip(distances, info))

        positive_scores, negative_scores = scores[:len(positive)], scores[len(positive):]
        return positive_scores, negative_scores

    def get_normalized_scores(self, clip_negative):
        df = pd.read_csv(os.path.join(DATA_DIR, 'validation_pairs.csv'))

        positive = df.loc[df['the_same'] == True]
        negative = df.loc[df['the_same'] == False]
        
        negative = negative.sample(frac=1)[:clip_negative]

        print(len(positive))
        print(len(negative))

        _positive_scores, _negative_scores = self.get_scores(positive, negative)

        print(len(_positive_scores))
        print(len(_negative_scores))

        _positive_scores = [((2 - row[0]) / 2, *row[1:]) for row in _positive_scores]
        _negative_scores = [((2 - row[0]) / 2, *row[1:]) for row in _negative_scores]
        return _positive_scores, _negative_scores        
        

class HeatMaps(Evaluation):
    def get_heatmap_encoder_learnt(self, img):
        img_tensor = prepare_tensor_from_image(img, self.model_name)
        _ = self.encoder.predict(img_tensor)

        distances = self.encoder.get_output_at(0)
        last_conv_layer = self.encoder.get_layer('block14_sepconv2_act')

        grads = K.gradients(distances, last_conv_layer.output)[0]

        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([self.encoder.get_input_at(0)], [pooled_grads, last_conv_layer.get_output_at(0)[0]])
        pooled_grads_value, conv_layer_output_value = iterate([img_tensor])

        for channel in range(512):
            conv_layer_output_value[:, :, channel] *= pooled_grads_value[channel]

        heatmap = np.abs(np.mean(conv_layer_output_value, axis=-1))
        heatmap /= np.max(heatmap)

        return heatmap

    def show_visualizations(self, scores, number_of_samples=None):
        random.shuffle(scores)
        counter = 0
        for row in scores:
            score, (image_A_name, image_B_name, label) = row

            if number_of_samples is not None and counter >= number_of_samples:
                break
            
            img_A = load_image(image_A_name, input_shape=self.input_shape, mode='train')
            img_B = load_image(image_B_name, input_shape=self.input_shape, mode='train')

            heatmap_A = self.get_heatmap_encoder_learnt(img_A)
            heatmap_B = self.get_heatmap_encoder_learnt(img_B)

            plot_side_by_side([heatmap_A * 255, img_A, img_B, heatmap_B * 255],
                              titles=['heatmap B',
                                      image_A_name + "\nSCORE: {}".format(score),
                                      image_B_name + "\nThe same? : {}".format(label),
                                      'heatmap B'])
            counter += 1
