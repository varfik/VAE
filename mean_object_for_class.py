import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10

# Загрузка данных CIFAR-10 (с метками классов)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0  # нормализация [0, 1]

# Загрузка encoder (убедитесь, что путь правильный)
encoder = load_model("encoder_final.h5", custom_objects={"KLDivergenceLayer": KLDivergenceLayer})

# Список классов CIFAR-10 (всего 10)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Создаем словарь для хранения средних латентных векторов
class_latent_means = {}

# Для каждого класса вычисляем средний латентный вектор
for class_id in range(10):
    # Выбираем все изображения данного класса
    class_images = x_train[y_train.flatten() == class_id]
    
    # Кодируем их в латентное пространство
    latent_vectors = encoder.predict(class_images, batch_size=128)
    
    # Вычисляем среднее по всем векторам
    mean_latent = np.mean(latent_vectors, axis=0)
    
    # Сохраняем результат
    class_latent_means[class_names[class_id]] = mean_latent

    print(f"Class: {class_names[class_id]}, Mean Latent Shape: {mean_latent.shape}")

import matplotlib.pyplot as plt

# Загружаем decoder
decoder = load_model("decoder_final.h5")

# Визуализируем средние изображения для каждого класса
plt.figure(figsize=(15, 8))
for i, (class_name, mean_latent) in enumerate(class_latent_means.items()):
    # Декодируем средний вектор в изображение
    generated_image = decoder.predict(mean_latent.reshape(1, -1))[0]
    
    # Рисуем
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated_image)
    plt.title(class_name)
    plt.axis('off')

plt.tight_layout()
plt.show()