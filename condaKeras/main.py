import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

#нормальный формат выходных значений
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# функции активации
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),  #Первый слой должен преобразовывать изображение 28x28 пикселей в вектор из 784 элементов
    Dense(128, activation='relu'),   #функция активации скрытого слоя
    Dense(10, activation='softmax')      #итоговые нейроны
])

print(model.summary())      # вывод структуры НС в консоль

model.compile(optimizer='adam',      #критерий качества
             loss='categorical_crossentropy',
             metrics=['accuracy'])


model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)   #запуск обучения

model.evaluate(x_test, y_test_cat)  #проверка работы сети на тестовом множестве

n = 15
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)     #прогоняем его по сети
print(res)
print( "Цифра на картинке -", np.argmax(res))  #выделяем из вектора выходных значений макс значение

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

