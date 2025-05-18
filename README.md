Kan Mikrobiyom Verilerine Dayalı Kanser Tanısı
Bu repo, Erciyes Üniversitesi BS444 Makine Öğrenmesine Giriş dersi midterm projesi olarak hazırlanmıştır. Projede, kan mikrobiyom verilerini kullanarak dört farklı kanser türünü (meme, kolon, akciğer ve prostat kanseri) sınıflandırmak için makine öğrenmesi modelleri geliştirilmiştir.
Proje Özeti
Bu projede:

355 hastadan toplanan kan mikrobiyom verileri kullanılmıştır
Veri normalizasyonu yapılarak öznitelikler ön işlemeden geçirilmiştir
Random Forest ve LightGBM algoritmaları karşılaştırılmıştır
Hiperparametre optimizasyonu için RandomizedSearchCV kullanılmıştır
5 katlı çapraz doğrulama ile model performansı değerlendirilmiştir
Duyarlılık (Sensitivity) ve Özgüllük (Specificity) metrikleri ile performans ölçülmüştür

Veri Seti ve Ön İşleme
Veri Seti Genel Bakış
Çalışmamızda kullanılan veri seti şunları içermektedir:

355 kişiden alınan kan örnekleri
4 farklı kanser türü: Meme kanseri (107 örnek, %30.14), Kolon kanseri (109 örnek, %30.70), Akciğer kanseri (18 örnek, %5.07) ve Prostat kanseri (121 örnek, %34.08)
1836 farklı mikroorganizma türüne ait DNA parçası sayımlarını içeren özellikler

Veri Temizliği ve Kontroller
Veri setinde herhangi bir eksik değer bulunmamaktadır:
Eksik değer kontrolü:
X veri setindeki eksik değer toplamı: 0
y veri setindeki eksik değer toplamı: 0
Veri Normalizasyonu
Ödevde belirtildiği gibi, her örnek için DNA parçacık sayılarını toplam sayıya bölerek normalizasyon uyguladık:
pythonrow_sums = X.sum(axis=1)
X_normalized = X.div(row_sums, axis=0)
Metodoloji
Veri Bölme
Her kanser türü için veri setini eğitim (%80) ve test (%20) setlerine ayırdık. Sınıf dağılımının korunmasını sağlamak için tabakalı örnekleme kullandık:
pythonX_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
Cross-Validation Stratejisi
Model performansının güvenilir bir şekilde değerlendirilmesi için 5 katlı tabakalı çapraz doğrulama kullandık:
pythoncv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
Model Seçimi ve Hiperparametre Optimizasyonu
İki farklı makine öğrenimi algoritması karşılaştırdık:
Random Forest
RandomizedSearchCV kullanarak kapsamlı bir hiperparametre optimizasyonu gerçekleştirdik:
pythonrf_param_dist = {
    'n_estimators': [100, 200, 400, 800],
    'max_depth': [2, 4, 8, 16, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}
LightGBM
LightGBM için de benzer bir hiperparametre optimizasyonu gerçekleştirdik:
pythonlgb_param_dist = {
    'n_estimators': randint(50, 500),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'num_leaves': randint(15,128),
    'min_child_samples': randint(10, 40),
    'subsample': uniform(0.4, 0.6),  
    'colsample_bytree': uniform(0.4, 0.6), 
    'reg_alpha': uniform(0, 1.0),
    'reg_lambda': uniform(0, 1.0)
}
Sonuçlar ve Analiz
Model Performansları
Her kanser türü için her iki modelin de performans sonuçları grafiklerle gösterilmiştir.
Kanser Türlerine Göre Performans Analizi

Kolon Kanseri: Her iki model için de en başarılı sınıflandırma sonuçları.
Akciğer Kanseri: En düşük duyarlılık değerlerine sahip, bunun muhtemel nedeni veri setindeki örnek sayısının azlığı (sadece 18 örnek).
Meme ve Prostat Kanseri: LightGBM, bu kanser türleri için Random Forest'a göre daha yüksek duyarlılık göstermiştir.

Gereksinimler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.stats import randint, uniform
Kurulum
Gerekli kütüphaneleri Conda ile kurabilirsiniz:
conda install pandas numpy matplotlib seaborn scikit-learn
conda install -c conda-forge lightgbm
conda install -c anaconda scipy
