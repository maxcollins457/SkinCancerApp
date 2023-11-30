from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
from keras.models import clone_model

from config import ML_MODELS

class GradCAM:
    def __init__(self, model_name):
        original_model = ML_MODELS[model_name]['model']
        self.model = clone_model(original_model)
        self.model.layers[-1].activation = None
        self.last_conv_layer_name = ML_MODELS[model_name]['last_conv_layer_name']

    def get_img_array(self, img_path):
        img = Image.open(img_path)
        array = keras.utils.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array

    def make_gradcam_heatmap(self, img_path, pred_index=None):
        img_array = self.get_img_array(img_path)
        grad_model = keras.models.Model(
            self.model.inputs, [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
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
        # Rescale heatmap to a range 0-255
        heatmap = np.nan_to_num(heatmap)
        return np.uint(255 * heatmap)

    def write_gradcam_to_buffer(self, buffer, img_path):
        img = Image.open(img_path)
        img_gray = img.convert("L")
        img_array = keras.utils.img_to_array(img_gray)

        # Generate class activation heatmap
        heatmap = self.make_gradcam_heatmap(img_path)

        # Use jet colormap to colorize heatmap
        jet = mpl.cm.get_cmap("magma")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)

        # Superimpose the heatmap on the original image
        superimposed_img = jet_heatmap + img
        superimposed_img = keras.utils.array_to_img(superimposed_img)

        # Display the resized image with a colorbar legend
        plt.imshow(superimposed_img)  # Display as grayscale
        plt.colorbar(label="Activation Level")
        plt.axis("off")
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
