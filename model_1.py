
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam,Adagrad,Nadam
from tensorflow.keras.layers import Input
import networkx as nx
from telco_preprocessing  import load

## Yapay Sinir Ağı Modeli (Eğitim setini aynı zamanda test verisi olarak kullanarak)
X,y=load()
# Yapay Sinir Ağı Modeli

def build_model(input_dim, learning_rate, dropout_rate):
    model = Sequential([
        Input(shape=(input_dim,)),  # İlk katmanda Input kullanıyoruz
        Dense(128, activation="relu"),
        Dropout(dropout_rate),
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

input_dim = X.shape[1]
model1 = build_model(input_dim=input_dim, learning_rate=0.001, dropout_rate=0.0)

# Model Eğitimi
history= model1.fit(X,y, epochs=100, batch_size=32, verbose=1)

# Model Performansı
y_pred = (model1.predict(X) > 0.5).astype(int)
print(classification_report(y, y_pred))
import matplotlib.pyplot as plt



def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.title('Confusion Matrix')
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

import networkx as nx
import matplotlib.pyplot as plt


def visualize_network_networkx(input_size, hidden_sizes, output_size):
    G = nx.DiGraph()  # Yönlü bir graph oluştur

    # Katmanlar ve düğümler
    layers = [range(input_size)]  # İlk katman (giriş katmanı)
    for hidden_size in hidden_sizes:
        layers.append(range(max(layers[-1]) + 1, max(layers[-1]) + 1 + hidden_size))  # Gizli katmanlar
    layers.append(range(max(layers[-1]) + 1, max(layers[-1]) + 1 + output_size))  # Çıkış katmanı

    # Ağ yapısını oluştur
    for l in range(len(layers) - 1):
        for node in layers[l]:  # Mevcut katmanın düğümleri
            for next_node in layers[l + 1]:  # Sonraki katmanın düğümleri
                G.add_edge(node, next_node)

    # Düğüm pozisyonları
    pos = {}
    layer_gap = 3  # Katmanlar arası boşluk
    for i, layer in enumerate(layers):
        y_offset = (max(input_size, *hidden_sizes, output_size) - len(layer)) / 2
        for j, node in enumerate(layer):
            pos[node] = (i * layer_gap, j + y_offset)  # X ekseninde katman sırası, Y ekseninde düğüm pozisyonu

    # Düğümleri ve kenarları çiz
    plt.figure(figsize=(14, 7))
    nx.draw_networkx_edges(G, pos, alpha=0.05, edge_color="gray")  # Kenarlar
    nx.draw_networkx_nodes(G, pos, nodelist=layers[0], node_color="green", node_size=15, label="Input")  # Giriş
    for i, hidden_layer in enumerate(layers[1:-1], 1):
        nx.draw_networkx_nodes(G, pos, nodelist=hidden_layer, node_color="blue", node_size=15,
                               label="Hidden" if i == 1 else None)  # Gizli
    nx.draw_networkx_nodes(G, pos, nodelist=layers[-1], node_color="red", node_size=15, label="Output")  # Çıkış

    # Grafik özellikleri
    plt.axis("off")
    plt.legend(scatterpoints=1)
    plt.show()


# Örnek kullanım
visualize_network_networkx(30,[128,64,32,16,8], 1)



