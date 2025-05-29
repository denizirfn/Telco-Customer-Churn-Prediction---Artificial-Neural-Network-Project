import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score,f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam,Adagrad,Nadam
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model
from telco_preprocessing  import load

## Yapay Sinir Ağı Modeli (%66-%34 eğitim test ayırarak (5 farklı rassal ayırma ile))

# Veri Yükleme
X, y = load()

# Parametreler
num_splits = 5
epochs = 30
batch_size = 32

# Sonuçları depolamak için liste
results = []

for i in range(num_splits):
    print(f"Split {i + 1}")

    # Eğitim ve test verisini ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=i)



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


    input_dim = X_train.shape[1]

    model2 = build_model(input_dim=input_dim, learning_rate=0.001, dropout_rate=0.5)

    # Modeli eğitme
    history = model2.fit(
        X_train, y_train,
        epochs=epochs, batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0
    )

    # Tahmin yapma
    y_prob = model2.predict(X_test, verbose=0).flatten()  # Tahmin olasılıkları
    y_pred = (y_prob > 0.5).astype(int)  # Sınıf etiketleri

    # Performans metrikleri
    accuracy = model2.evaluate(X_test, y_test, verbose=0)[1]
    auc_score = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    results.append({'split': i + 1, 'accuracy': accuracy, 'roc_auc': auc_score,'f1_score':f1})


    # Konfüzyon Matrisi
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Sonuçları özetleme
results_df = pd.DataFrame(results)
print("\n=== Final Results ===")
print(results_df)

# Ortalama sonuçları yazdırma
print("\nAverage Results:")
print(results_df.mean(numeric_only=True))