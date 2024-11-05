# -*- coding: utf-8 -*-
"""
@author: Wishnu H
"""

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 1. Memuat model VGG16 tanpa top layer
def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(148, 148, 3))
    
    # Membangun model
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # Mengkompilasi model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 2. Cek apakah model sudah ada
try:
    model = load_model('vgg16_cats_dogs_model.h5')
    print("Model berhasil dimuat.")
except Exception as e:
    print("Model tidak ditemukan, membuat model baru.")
    model = create_model()

# 3. Mengatur augmentasi data
train_datagen = ImageDataGenerator(rescale=1./255, 
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

# 4. Mengatur generator untuk data pelatihan
train_generator = train_datagen.flow_from_directory(
    'E:\\LATIHAN\\deeplearning\\Data\\Cats vs Dogs\\data\\train',  # Ganti dengan path ke direktori data pelatihan Anda
    target_size=(148, 148),
    batch_size=32,
    class_mode='binary')

# 5. Melatih model jika baru dibuat
if not hasattr(model, 'layers'):
    model.fit(train_generator, epochs=20)
    # Simpan model setelah pelatihan
    model.save('E:\\LATIHAN\\deeplearning\\vgg16_cats_dogs_model.h5')

# 6. Prediksi dengan gambar baru
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(148, 148))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Menambah dimensi batch
    img_array /= 255.0  # Normalisasi

    prediction = model.predict(img_array)
    
    # Menentukan label prediksi
    if prediction[0][0] > 0.5:
        class_label = "Anjing"
    else:
        class_label = "Kucing"
    
    # Menampilkan hasil prediksi
    print(f"Prediksi: {class_label}")
    
    # Menampilkan gambar yang diprediksi
    plt.imshow(img)
    plt.axis('off')  # Matikan sumbu
    plt.show()

# 7. Contoh penggunaan fungsi prediksi
img_path = 'E:\\LATIHAN\\deeplearning\\Data\\Cats vs Dogs\\data\\test\\dogs\\dog.1139.jpg'  # Ganti dengan path gambar Anda
predict_image(img_path)
