# Telco-Customer-Churn-Prediction---Artificial-Neural-Network-Project
# Telco Müşteri Kaybı Tahmini - Yapay Sinir Ağı Projesi

## 📌 Proje Özeti

Bu projede, Telco (telekomünikasyon) müşterilerinin hizmetten ayrılıp ayrılmayacağını tahmin etmek amacıyla bir **yapay sinir ağı (YSA)** modeli geliştirilmiştir. Veri seti, Kaliforniya'daki 7043 müşterinin demografik bilgileri ve hizmet geçmişini içermektedir.

Projenin ana amacı, doğru bir YSA modeli ile müşteri kaybı (churn) ihtimalini tahmin edebilmektir.

---

## 🛠 Kullanılan Teknolojiler ve Kütüphaneler

- Python
  - `Pandas` – Veri analizi ve ön işleme
  - `NumPy` – Sayısal işlemler
  - `scikit-learn` – Modelleme, değerlendirme ve ölçekleme
  - `Matplotlib`, `Seaborn` – Görselleştirme
  - `TensorFlow / Keras` – Yapay sinir ağı modellemesi
  - `NetworkX` – Ağ yapısının görselleştirilmesi

---

## 📂 Veri Seti

- Gözlem sayısı: 7043
- Değişken sayısı (ön işleme sonrası): 32
- Hedef değişken: `Churn` (0 - Kalmış, 1 - Terk etmiş)

---

## 🧪 Veri Ön İşleme Adımları

- Eksik değer analizi ve median ile doldurma
- Kategorik veriler için:
  - Label Encoding (binary değişkenlerde)
  - One-Hot Encoding (çoklu kategorilerde)
- `CustomerID` gibi model için anlamsız değişkenler çıkarıldı
- Özellikler `StandardScaler` ile ölçeklendirildi

---

## 🧠 Modelleme Yöntemleri

### 1. Eğitim verisi = Test verisi
- Epochs: 100
- Batch size: 32
- Gizli katman sayısı: 5
- Aktivasyon: `relu` (gizli), `sigmoid` (çıkış)

### 2. %66 Eğitim / %34 Test (5 farklı split)
- Epochs: 30
- Dropout: 0.5
- Gizli katman sayısı: 4

### 3. 5-Fold Cross Validation
- Epochs: 50
- Batch size: 16
- Dropout: 0.3
- Gizli katman sayısı: 5

### 4. 10-Fold Cross Validation
- Epochs: 50
- Dropout: 0.5
- Gizli katman sayısı: 4

Tüm modellerde optimizasyon için **Adam**, çıktı aktivasyon fonksiyonu için **sigmoid** kullanılmıştır.

---

## 📊 Performans Ölçütleri

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Performans sonuçları, her yönteme ait detaylı şekilde analiz edilmiştir. Ayrıca confusion matrix görselleri ile desteklenmiştir.

---

## 📚 Kaynakça

- Scikit-learn Documentation: https://scikit-learn.org/stable/
- Python for Data Analysis – W. McKinney
- Python Data Science Handbook – J. VanderPlas
- Cross-Validation - M.B. Durna, Medium

---

## 👩‍💻 Hazırlayan

**Yeliz İrfan**  
Konya Teknik Üniversitesi - Bilgisayar Mühendisliği  

