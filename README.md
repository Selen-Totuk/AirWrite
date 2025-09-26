AirWrite – Havada Parmakla Yazı ve Basit CNN Tahmin / AirWrite – Finger Tracking & Simple CNN Prediction

Türkçe

Bu proje, kamerayı kullanarak parmak ucunu takip eden, havada yazdığın çizgileri ekranda gösteren ve isteğe bağlı olarak basit bir CNN modeline verip karakter tahmini yapan bir Python uygulamasıdır.

Kullanılan Teknolojiler
Kamera ve görüntü işleme → OpenCV
El ve parmak takibi → MediaPipe
Model eğitimi → TensorFlow / Keras
Sayısal hesaplamalar → NumPy

Kurulum
Python 3.8+ kurulu olmalı.

Gerekli kütüphaneleri yükle:
pip install opencv-python mediapipe tensorflow numpy
TensorFlow büyük olduğu için CPU-only sürümünü de kullanabilirsin:
pip install tensorflow-cpu

air_write.py dosyasını masaüstüne veya istediğin klasöre koy.

Kullanım
PowerShell’de dosyanın bulunduğu klasöre geç:
cd <dosyanın bulunduğu klasör>
Programı çalıştır:
python air_write.py


Klavye tuşları:

0–9 → Çizimi dataset klasörüne kaydeder
c → Canvas’ı temizler
s → Çizimi saved klasörüne kaydeder
t → Model eğitimi (TensorFlow kuruluysa çalışır)
p → Eğitilmiş model ile tahmin yapar
q → Programı kapatır

English
This project is a Python application that tracks your fingertip via camera, displays lines you draw in the air, and optionally feeds them to a simple CNN model for character prediction.

Technologies Used: OpenCV, MediaPipe, TensorFlow/Keras, NumPy
Installation:
pip install opencv-python mediapipe tensorflow numpy
Usage:
cd <folder-containing-air_write.py>
python air_write.py
Keyboard controls:
0–9 → Save drawing to dataset folder
c → Clear canvas
s → Save to saved folder
t → Train model (TensorFlow required)
p → Predict with trained model

q → Quit
