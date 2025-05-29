import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
from keras.layers import Input, Dense, Reshape, Concatenate, Flatten, Lambda, Reshape, Layer
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
tfd = tfp.distributions

def get_encoder_network(x, num_filters):
    x = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = MaxPooling2D()(x)
    return x

def get_decoder_network(x, num_filters):
    x = UpSampling2D()(x)
    x = Conv2D(num_filters, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(num_filters, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    return x

def get_vae(height, width, batch_size, latent_dim,
            is_variational=True,
            start_filters=8, nb_capacity=3,
            optimizer=Adam(learning_rate=0.001)):

    # ВХОД ##
    inputs = Input((height, width, 3))
    x = inputs

    # ЭНКОДЕР ##
    for i in range(nb_capacity + 1):
        x = get_encoder_network(x, start_filters * (2 ** i))

    shape_spatial = x.shape[1:]
    x_flat = Flatten()(x)
    
    if not is_variational:
        z = Dense(latent_dim)(x_flat)
        z_mean, z_log_var = None, None
    else:
        z_mean = Dense(latent_dim)(x_flat)
        z_log_var = Dense(latent_dim)(x_flat)

        # Слой для выборки z с использованием reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, name="sampling_layer")([z_mean, z_log_var])

        # Слой для вычисления KL-дивергенции
        class KLDivergenceLayer(Layer):
            def call(self, inputs):
                z_mean, z_log_var = inputs
                kl_loss = -0.5 * tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=-1
                )
                self.add_loss(tf.reduce_mean(kl_loss))
                return z_mean  # Возвращаем что-то, но не используется

        kl_layer = KLDivergenceLayer(name="kl_divergence_layer")
        kl_layer([z_mean, z_log_var])  # Добавляем KL-дивергенцию в модель

    ## ДЕКОДЕР ##
    decoder_input = Input(shape=(latent_dim,))
    x_dec = Dense(np.prod(shape_spatial), activation='relu')(decoder_input)
    x_dec = Reshape(shape_spatial)(x_dec)

    for i in range(nb_capacity, -1, -1):
        x_dec = get_decoder_network(x_dec, start_filters * (2 ** i))

    decoder_output = Conv2D(3, 3, activation='sigmoid', padding='same')(x_dec)
    decoder = Model(decoder_input, decoder_output, name="decoder")

    # Выход VAE
    vae_output = decoder(z)
    vae = Model(inputs, vae_output, name="vae")

    # Компиляция с MSE (KL уже учтена через add_loss)
    vae.compile(optimizer=optimizer, loss='mse')

    return vae, Model(inputs, z, name="encoder"), decoder

# гиперпараметры
VARIATIONAL = True
HEIGHT = 32
WIDTH = 32
BATCH_SIZE = 128
LATENT_DIM = 16
START_FILTERS = 32
CAPACITY = 2
CONDITIONING = True
OPTIMIZER = Adam(learning_rate=0.001)

vae, encoder, decoder = get_vae(is_variational=VARIATIONAL,
                                   height=HEIGHT,
                                   width=WIDTH,
                                   batch_size=BATCH_SIZE,
                                   latent_dim=LATENT_DIM,
                                   start_filters=START_FILTERS,
                                   nb_capacity=CAPACITY,
                                   optimizer=OPTIMIZER)

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Загрузка данных
(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0  # нормализация [0, 1]
x_test = x_test.astype('float32') / 255.0

# Параметры обучения
EPOCHS = 5
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1

# Коллбэки
callbacks = [
    ModelCheckpoint("best_vae.h5", save_best_only=True, monitor='val_loss'),
    EarlyStopping(patience=5, restore_best_weights=True)
]

# Обучение модели
history = vae.fit(
    x_train,
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks
)

# Оценка модели
test_loss = vae.evaluate(
    x_test,
    x_test,
    batch_size=BATCH_SIZE
)
print(f"Test loss: {test_loss:.4f}")

# Визуализация обучения
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()