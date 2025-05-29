import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler


#ayarlamalar
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

def load():
    # VERİ ANALİZİ
    df = pd.read_csv('datasets/Telco-Customer-Churn.csv')
    df.head(50)
    df.info()
    df.shape


    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print(df.isnull().sum())
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    print(df.isnull().sum())

    # label encoding
    binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    binary_mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    df[binary_columns] = df[binary_columns].replace(binary_mapping)
    # one-hat encoding
    categorical_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                           'Contract', 'PaymentMethod']
    data_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    print(data_encoded.shape)  # shape arttı

    data_cleaned = data_encoded.drop(columns=['customerID'])  # cardinalitesi yüksek kullanma

    X = data_cleaned.drop(columns=['Churn'])  # bağımsız değşkeni sil
    y = data_cleaned['Churn']  # y=hedef değişken

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # girişler standart yap
    print(X_scaled.shape)  # 30 giriş ,çıkış 1
    X = pd.DataFrame(X_scaled, columns=X.columns)
    return X,y
load()