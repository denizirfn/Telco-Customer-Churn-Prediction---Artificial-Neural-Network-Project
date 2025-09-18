📝 Proje Özeti

Bu proje, Kaliforniya’daki 7043 müşteriye ev telefonu ve internet hizmeti sağlayan hayali bir telekom şirketinin müşteri kaybını (churn) tahmin etmeyi amaçlar.

Veri seti, hangi müşterilerin hizmetten ayrıldığını, kaldığını veya kaydolduğunu içerir.

Hedef, şirketten ayrılacak müşterileri tahmin edebilecek Yapay Sinir Ağı (YSA) modeli geliştirmektir.

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

<img width="403" height="247" alt="image" src="https://github.com/user-attachments/assets/63ef17d3-34a4-4143-b8ef-43bb1ff135e6" /><img width="422" height="243" alt="image" src="https://github.com/user-attachments/assets/d72b1048-9b76-4c34-b3ee-5a8aef5daa2d" />


---

## 🧠 Modelleme Yöntemleri
Yapay sinir ağı modeli farklı yöntemlerle eğitildi:
| Yöntem                                | Epoch | Batch | Dropout | Gizli Katman |
| ------------------------------------- | ----: | ----: | ------: | -----------: |
| Eğitim = Test                         |   100 |    32 |       – |            5 |
| %66-%34 Train/Test (5 rastgele bölme) |    30 |    32 |     0.5 |            4 |
| 5-Fold Cross Validation               |    50 |    16 |     0.3 |            5 |
| 10-Fold Cross Validation              |    50 |    16 |     0.5 |            4 |


Aktivasyon: Gizli katmanlarda ReLU, çıkış katmanında Sigmoid
Optimizasyon: Adam (lr=0.001)

---

📈 Performans Ölçütleri

Model başarıları; Accuracy, Precision, Recall, F1-Skoru ve Konfüzyon Matrisi ile değerlendirildi.

Accuracy: Doğru tahminlerin toplam tahminlere oranı

Precision: Pozitif tahminlerin doğruluk oranı

Recall: Gerçek pozitiflerin yakalanma oranı

F1-Skoru: Precision ve Recall’un harmonik ortalaması

---
🔑 Araştırma Sonuçları

En iyi ağ topolojisi, 1 giriş katmanı, 4–5 gizli katman ve 1 çıkış katmanından oluşmuştur.

Tüm yöntemler arasında en yüksek doğruluk, cross-validation yaklaşımlarında gözlenmiştir.

Uygun dropout ve epoch değerleri, aşırı öğrenmeyi (overfitting) önemli ölçüde azaltmıştır.

Eğitim seti aynı zamanda test seti olarak kullanıldığında elde edilen başarı değeri:
<img width="763" height="319" alt="image" src="https://github.com/user-attachments/assets/fa9d6769-67e0-421c-9de1-4d2a058dab39" />


Eğitim seti aynı zamanda test seti olarak kullanıldığında elde edilen konfüzyon matrisi:
<img width="598" height="443" alt="image" src="https://github.com/user-attachments/assets/78e8d5a3-bb29-48ad-836e-8d618448fb2d" />


Eğitim test rassal olarak %66-34 bölünen başarı değeri (5 farklı rassal ayırma ile):
<img width="808" height="329" alt="image" src="https://github.com/user-attachments/assets/58412b6b-5935-4f3d-84b5-f2f8c977e6af" />


Eğitim test rassal olarak %66-34 kullanıldığında elde edilen konfüzyon matrisi görselleri:
<img width="451" height="350" alt="image" src="https://github.com/user-attachments/assets/da5570a7-eee3-47ef-bb74-4aa111a00e4e" />
<img width="466" height="349" alt="image" src="https://github.com/user-attachments/assets/424fbe5b-8168-4e44-a079-ae0e0d86dd12" />
<img width="472" height="349" alt="image" src="https://github.com/user-attachments/assets/7df0ba9d-8962-4ddf-a479-d514dbd5689e" />
<img width="469" height="346" alt="image" src="https://github.com/user-attachments/assets/03cb198d-3aa5-4b92-b112-bfed1b988ab4" />
<img width="474" height="357" alt="image" src="https://github.com/user-attachments/assets/ae226b74-d03b-49c1-a99a-38fd0c0f7590" />

5 katlı çapraz doğrulama sonucu başarı değerleri:
<img width="536" height="158" alt="image" src="https://github.com/user-attachments/assets/65cd2df4-3c53-4c4f-9f81-c5fbce71237e" />


5 katlı çapraz doğrulama sonucu elde edilen konfüzyon matrisi:
<img width="509" height="423" alt="image" src="https://github.com/user-attachments/assets/bb260f73-7b81-48c9-860c-c49f89bc3a87" />


10 katlı çapraz doğrulama sonucu başarı değerleri:
<img width="609" height="136" alt="image" src="https://github.com/user-attachments/assets/86bb0de7-58b2-4cdd-9200-15185370bb2d" />


10 katlı çapraz doğrulama sonucu elde edilen konfüzyon matrisi:
<img width="523" height="437" alt="image" src="https://github.com/user-attachments/assets/ca9f825c-a22e-4cf6-b554-f1f1a2a13637" />

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

