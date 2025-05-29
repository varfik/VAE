import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
import matplotlib
matplotlib.use('Agg')  # Устанавливаем бэкенд
import matplotlib.pyplot as plt

# Определяем кастомные компоненты
@register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@register_keras_serializable()
class KLDivergenceLayer(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=-1
        )
        self.add_loss(tf.reduce_mean(kl_loss))
        return z_mean

# Загрузка моделей
custom_objects = {
    'KLDivergenceLayer': KLDivergenceLayer,
    'sampling': sampling
}

encoder = load_model("encoder_final.h5", custom_objects=custom_objects)
decoder = load_model("decoder_final.h5", custom_objects=custom_objects)

# Обработка данных
(x_train, y_train), _ = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Создаем директорию для сохранения
os.makedirs("vae_output", exist_ok=True)

# Вычисляем и сохраняем средние изображения
for class_id, class_name in enumerate(class_names):
    # Получаем латентные векторы
    class_images = x_train[y_train.flatten() == class_id]
    latent_vectors = encoder.predict(class_images, batch_size=128)
    mean_latent = np.mean(latent_vectors, axis=0)
    
    # Генерируем и сохраняем изображение
    generated_image = decoder.predict(mean_latent.reshape(1, -1))[0]
    plt.figure()
    plt.imshow(generated_image)
    plt.title(class_name)
    plt.axis('off')
    plt.savefig(f"vae_output/{class_name}.png")
    plt.close()
    
    print(f"Saved: vae_output/{class_name}.png")