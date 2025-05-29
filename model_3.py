
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score,f1_score
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from telco_preprocessing  import load

# Verileri yükleme
X, y = load()

# Sınıf ağırlıklarını hesapla
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)
# build_model fonksiyonunu parametre alacak şekilde düzenleyelim
def build_model(input_dim, learning_rate, dropout_rate):
    model = Sequential([
        Input(shape=(input_dim,)),  # İlk katmanda Input kullanıyoruz
        Dense(64, activation="relu"),
        Dropout(dropout_rate),
        Dense(32, activation="relu"),
        Dropout(dropout_rate),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model



# KerasClassifier ile modeli sarmalayalım
input_dim = X_train.shape[1]
model3 = KerasClassifier(model=build_model, input_dim=input_dim, learning_rate=0.001, dropout_rate=0.3, epochs=50, batch_size=16, verbose=0)

# Sınıf ağırlıklarını ayarlayalım
model3.set_params(class_weight=dict(enumerate(class_weights)))

# Cross-validation işlemi
cv_results = cross_validate(model3, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

# Sonuçları yazdırma
print(f"Ortalama Test Accuracy: {cv_results['test_accuracy'].mean()*100:.2f}%")
print(f"Ortalama Test F1 Score: {cv_results['test_f1'].mean()*100:.2f}%")
print(f"Ortalama Test ROC AUC Score: {cv_results['test_roc_auc'].mean()*100:.2f}%")

# Modeli yeniden eğitip test verisi üzerinde tahminler yapalım
model3.fit(X_train, y_train)

# Test verisi üzerinde tahminler yapma
y_pred = model3.predict(X_test)

# Confusion matrix hesaplama
cm = confusion_matrix(y_test, y_pred)

# Konfüzyon matrisini görselleştirme
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negatif", "Pozitif"], yticklabels=["Negatif", "Pozitif"])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Değer")
plt.title("Konfüzyon Matrisi")
plt.show()