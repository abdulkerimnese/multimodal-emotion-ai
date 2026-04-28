# 🎭 Multimodal Fusion AI: Çok Modlu Duygu Tanıma Sistemi

Bu proje, bireylerin anlık duygusal durumlarını yalnızca tek bir veri tipine bağlı kalmadan, hem yüz ifadeleri (görüntü) hem de yazılı ifadeleri (metin) üzerinden eşzamanlı olarak analiz eden uçtan uca (end-to-end) bir yapay zeka web uygulamasıdır.

## 🚀 Proje Özeti
Proje kapsamında iki farklı yapay zeka mimarisi eğitilmiş ve "Karar Seviyesinde Birleştirme" (Late Fusion) yöntemiyle tek bir beyinde toplanmıştır:
* **Görsel Algı (CNN):** FER2013 veri seti kullanılarak eğitilen Convolutional Neural Network modeli, insan yüzündeki 7 temel duyguyu sınıflandırır.
* **Metin Algısı (NLP):** Sentiment140 veri seti kullanılarak eğitilen Doğal Dil İşleme modeli (TF-IDF + Logistic Regression), metindeki duygu kutupluluğunu (Pozitif/Negatif) tespit eder.
* **Akıllı Kırpma:** Görüntüler modelden önce **Mediapipe Face Mesh** teknolojisiyle taranır, arka plan gürültüsü silinerek sadece yüz bölgesine odaklanılır.

## 🛠️ Kullanılan Teknolojiler
* **Modelleme:** TensorFlow/Keras, Scikit-Learn
* **Bilgisayarlı Görme (CV):** OpenCV, Mediapipe
* **Arayüz & Görselleştirme:** Streamlit, Plotly
* **Veri İşleme:** Pandas, NumPy

## 📊 Performans ve Sonuçlar
* **CNN Doğruluğu:** Test veri seti üzerinde **%63.76** accuracy elde edilmiştir. (EarlyStopping ile 31. Epoch'ta optimize edilmiştir).
* **Fusion Başarısı:** Sistem özellikle sarkazm (iğneleme) içeren "ters köşe" durumlarda (Örn: Kızgın bir yüz ifadesiyle "Harika, kahvem döküldü!" yazılması) tek bir modüle aldanmayıp ağırlıklı ortalama mantığıyla doğru duyguyu yakalamayı başarmıştır.

## 💻 Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1. Depoyu klonlayın:
```bash
git clone [https://github.com/KULLANICI_ADIN/multimodal-emotion-ai.git](https://github.com/KULLANICI_ADIN/multimodal-emotion-ai.git)
cd multimodal-emotion-ai