
# Kan Mikrobiyom Verilerine Dayalı Kanser Tanısı

Bu repo, **Erciyes Üniversitesi BS444 - Makine Öğrenmesine Giriş** dersi kapsamındaki midterm projesi olarak hazırlanmıştır. Projenin amacı, **kan mikrobiyom verilerini** kullanarak **dört farklı kanser türünü** (meme, kolon, akciğer ve prostat) sınıflandırmak için makine öğrenmesi modelleri geliştirmektir.

---

## 🔍 Proje Özeti

Bu projede aşağıdaki adımlar gerçekleştirilmiştir:

- 355 hastadan toplanan kan mikrobiyom verileri kullanılmıştır.
- Veriler normalize edilerek öznitelikler ön işlenmiştir.
- **Random Forest** ve **LightGBM** algoritmaları karşılaştırılmıştır.
- Hiperparametre optimizasyonu için `RandomizedSearchCV` kullanılmıştır.
- 5 katlı **tabakalı çapraz doğrulama** ile model performansı değerlendirilmiştir.
- Performans ölçümünde **duyarlılık (Sensitivity)** ve **özgüllük (Specificity)** metrikleri kullanılmıştır.

---

## 🧬 Veri Seti ve Ön İşleme

### Veri Seti Özeti

- Toplam: **355 kan örneği**
- Kanser Türleri:
  - **Meme Kanseri**: 107 örnek (%30.14)
  - **Kolon Kanseri**: 109 örnek (%30.70)
  - **Akciğer Kanseri**: 18 örnek (%5.07)
  - **Prostat Kanseri**: 121 örnek (%34.08)
- Özellikler: **1836 farklı mikroorganizma türüne** ait DNA parçası sayımları

### Eksik Değer Kontrolü

Veri setinde eksik değer **bulunmamaktadır**:
```python
X.isnull().sum().sum()  # 0
y.isnull().sum().sum()  # 0
```

### Normalizasyon

Her örnek için DNA parça sayıları, toplam sayıya bölünerek normalize edilmiştir:
```python
row_sums = X.sum(axis=1)
X_normalized = X.div(row_sums, axis=0)
```

---

## 🧪 Metodoloji

### Veri Bölme

- Eğitim ve test setleri: **%80 eğitim / %20 test**
- Tabakalı örnekleme ile sınıf dağılımı korunmuştur:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)
```

### Çapraz Doğrulama

5 katlı **Stratified K-Fold** kullanılmıştır:
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## ⚙️ Modelleme ve Hiperparametre Optimizasyonu

### Random Forest

Hiperparametre aralığı:
```python
rf_param_dist = {
    'n_estimators': [100, 200, 400, 800],
    'max_depth': [2, 4, 8, 16, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}
```

### LightGBM

Hiperparametre aralığı:
```python
lgb_param_dist = {
    'n_estimators': randint(50, 500),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'num_leaves': randint(15, 128),
    'min_child_samples': randint(10, 40),
    'subsample': uniform(0.4, 0.6),
    'colsample_bytree': uniform(0.4, 0.6),
    'reg_alpha': uniform(0, 1.0),
    'reg_lambda': uniform(0, 1.0)
}
```

---

## 📊 Sonuçlar ve Performans Analizi

Her iki modelin performansı, her kanser türü için ayrı ayrı değerlendirilmiştir:

- **Kolon Kanseri**: En yüksek sınıflandırma başarısı
- **Akciğer Kanseri**: En düşük duyarlılık – örnek sayısının azlığı (18) etkili olabilir
- **Meme & Prostat Kanseri**: LightGBM, Random Forest'a kıyasla daha iyi sonuçlar vermiştir

Performans metrikleri:
- **Sensitivity / Specificity**
- **Accuracy**


---

## 🧰 Gereksinimler

Aşağıdaki Python kütüphaneleri kullanılmıştır:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.stats import randint, uniform
```

---

## ⚙️ Kurulum

Gerekli paketleri Conda ile aşağıdaki şekilde kurabilirsiniz:

```bash
conda install pandas numpy matplotlib seaborn scikit-learn
conda install -c conda-forge lightgbm
conda install -c anaconda scipy
```
