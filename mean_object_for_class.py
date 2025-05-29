import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
import matplotlib.pyplot as plt

# Определяем и регистрируем функцию sampling
@register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Определяем KLDivergenceLayer с регистрацией
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

# Загрузка данных CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0

# Загрузка моделей с указанием всех custom_objects
custom_objects = {
    'KLDivergenceLayer': KLDivergenceLayer,
    'sampling': sampling
}

encoder = load_model("encoder_final.h5", custom_objects=custom_objects)

# Список классов CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Вычисляем средние латентные векторы
class_latent_means = {}
for class_id in range(10):
    class_images = x_train[y_train.flatten() == class_id]
    latent_vectors = encoder.predict(class_images, batch_size=128)
    mean_latent = np.mean(latent_vectors, axis=0)
    class_latent_means[class_names[class_id]] = mean_latent
    print(f"Class: {class_names[class_id]}, Mean Latent Shape: {mean_latent.shape}")

# Визуализация
decoder = load_model("decoder_final.h5", custom_objects=custom_objects)

plt.figure(figsize=(15, 8))
for i, (class_name, mean_latent) in enumerate(class_latent_means.items()):
    generated_image = decoder.predict(mean_latent.reshape(1, -1))[0]
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated_image)
    plt.title(class_name)
    plt.axis('off')

plt.tight_layout()
plt.show()