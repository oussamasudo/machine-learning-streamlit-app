# 🔧 Documentation Technique - ML Application

Documentation technique complète pour les développeurs et contributeurs.

---

## 📋 Table des matières

1. [Architecture générale](#architecture)
2. [Module data_loader.py](#data-loader)
3. [Module preprocessing.py](#preprocessing)
4. [Module models.py](#models)
5. [Module evaluation.py](#evaluation)
6. [Module app.py](#app)
7. [API Reference](#api)
8. [Bonnes pratiques](#bonnes-pratiques)

---

## 🏗️ Architecture générale {#architecture}

### Stack technique

- **Framework UI** : Streamlit 1.x
- **ML/Data** : scikit-learn, pandas, numpy
- **Visualisation** : matplotlib, seaborn
- **Serialization** : pickle, openpyxl

### Pattern de conception

**MLModelManager** : Classe centrale utilisant le pattern **Facade**
- Encapsule la complexité de scikit-learn
- Interface unifiée pour tous les algorithmes
- Gestion du cycle de vie complet (train→predict→evaluate)

---

## 📥 Module data_loader.py {#data-loader}

### Fonction principale

```python
def load_data(uploaded_file) -> pd.DataFrame | None
```

**Paramètres** :
- `uploaded_file` : UploadedFile (Streamlit) - Fichier uploadé

**Retour** :
- `DataFrame` : Données chargées
- `None` : En cas d'erreur

**Formats supportés** :
- CSV (`.csv`) → `pd.read_csv()`
- Excel (`.xlsx`, `.xls`) → `pd.read_excel()`

**Gestion d'erreurs** :
- Format non supporté → Affiche erreur Streamlit
- Erreur de parsing → Capture exception et affiche message

### Exemple d'usage

```python
from data_loader import load_data

uploaded = st.file_uploader("Upload CSV")
if uploaded:
    df = load_data(uploaded)
    if df is not None:
        st.dataframe(df)
```

---

## ⚙️ Module preprocessing.py {#preprocessing}

### 1. handle_missing_values()

```python
def handle_missing_values(df: pd.DataFrame, method: str = "moyenne") -> pd.DataFrame
```

**Méthodes disponibles** :
- `"supprimer les lignes"` → `df.dropna()`
- `"moyenne"` → `fillna(mean())` pour numériques, `mode()` pour catégorielles
- `"médiane"` → `fillna(median())` pour numériques
- `"valeur fixe"` → 0 pour numériques, "Missing" pour catégorielles

**Logique** :
```python
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == "object":
            # Traitement catégoriel
        else:
            # Traitement numérique
```

### 2. encode_categorical()

```python
def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]
```

**Retour** :
- `df` : DataFrame encodé
- `encoders` : Dict {col_name: LabelEncoder} pour reverse transform

**Implémentation** :
```python
from sklearn.preprocessing import LabelEncoder

encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
```

### 3. normalize_features()

```python
def normalize_features(
    df: pd.DataFrame, 
    method: str = 'StandardScaler', 
    exclude_target: bool = True
) -> pd.DataFrame
```

**Méthodes** :
- `StandardScaler` : μ=0, σ=1
- `MinMaxScaler` : [0, 1]
- `RobustScaler` : Basé sur quantiles (robuste aux outliers)

**Important** : Exclut la colonne 'target' par défaut

---

## 🤖 Module models.py {#models}

### Classe MLModelManager

#### Attributs d'instance

```python
class MLModelManager:
    def __init__(self):
        self.model = None              # Modèle sklearn
        self.model_type = None         # 'regression', 'classification', 'clustering'
        self.scaler = StandardScaler() # Pour normalisation
        self.X_train = None            # Features d'entraînement
        self.X_test = None             # Features de test
        self.y_train = None            # Target d'entraînement
        self.y_test = None             # Target de test
        self.predictions = None        # Prédictions
        self.is_fitted = False         # Flag d'entraînement
```

#### Méthodes principales

##### prepare_data()

```python
def prepare_data(
    self, 
    X: ArrayLike, 
    y: ArrayLike, 
    test_size: float = 0.2, 
    random_state: int = 42, 
    scale: bool = True
) -> tuple
```

**Workflow** :
1. `train_test_split()` → Division données
2. Si `scale=True` : `StandardScaler.fit_transform()` sur X_train, `transform()` sur X_test
3. Stocke dans attributs d'instance

##### train()

```python
def train(self, X=None, y=None) -> Model
```

**⚠️ Point clé** : **Cette méthode entraîne réellement le modèle**

```python
# Ligne critique (321 dans models.py)
self.model.fit(X_train, y_train)  # ← ENTRAÎNEMENT
self.is_fitted = True
```

##### predict()

```python
def predict(self, X=None) -> np.ndarray
```

**Vérifications** :
```python
if not self.is_fitted:
    raise ValueError("Modèle non entraîné")
```

#### Méthodes de création de modèles

**Pattern** : Toutes suivent le même schéma

```python
def <algorithme>_<type>(self, params=None):
    params = params or {}
    default_params = {...}  # Paramètres par défaut
    default_params.update(params)
    
    self.model = SklearnModel(**default_params)
    self.model_type = 'regression' | 'classification' | 'clustering'
    return self.model
```

**Exemples** :
- `random_forest_classifier(params)`
- `svm_regressor(params)`
- `kmeans_clustering(n_clusters, params)`

#### Méthodes d'évaluation

##### evaluate_regression()

```python
def evaluate_regression(
    self, 
    y_true=None, 
    y_pred=None
) -> dict
```

**Retour** :
```python
{
    'MSE': float,
    'RMSE': float,
    'MAE': float,
    'R2': float
}
```

##### evaluate_classification()

```python
def evaluate_classification(
    self,
    y_true=None,
    y_pred=None,
    average='weighted'
) -> dict
```

**Retour** :
```python
{
    'Accuracy': float,
    'Precision': float,
    'Recall': float,
    'F1-Score': float,
    'Confusion Matrix': np.ndarray,
    'Classification Report': str
}
```

#### Sauvegarde/Chargement

```python
def save_model(self, filepath: str) -> None
def load_model(self, filepath: str) -> None
```

**Format** : Pickle avec dictionnaire contenant modèle + scaler + métadonnées

---

## 📊 Module evaluation.py {#evaluation}

### Fonctions d'évaluation

#### evaluate_model()

```python
def evaluate_model(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    model_type: str = 'regression',
    **kwargs
) -> dict
```

**Router function** : Délègue à la fonction spécialisée selon `model_type`

### Fonctions de visualisation

Toutes retournent `matplotlib.figure.Figure`

#### plot_regression_results()

```python
def plot_regression_results(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: str = "Résultats de régression"
) -> Figure
```

**Subplots** :
- Scatter plot : y_true vs y_pred + ligne identité
- Résidus vs prédictions
- Histogramme des résidus

#### plot_confusion_matrix()

```python
def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: list = None,
    title: str = "Matrice de confusion"
) -> Figure
```

**Rendu** : Heatmap seaborn avec annotations

---

## 🖥️ Module app.py {#app}

### Architecture Streamlit

#### Session State

```python
if "df" not in st.session_state:
    st.session_state.df = None
if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "model_manager" not in st.session_state:
    st.session_state.model_manager = None
if "trained" not in st.session_state:
    st.session_state.trained = False
```

**Usage** : Persistance entre reruns

#### Navigation

```python
page = st.radio("Navigation", [
    "Chargement des données",
    "Exploration",
    "Prétraitement",
    "Modélisation",
    "Évaluation",
    "Export"
])

if page == "Chargement des données":
    # Code de la page
elif page == "Exploration":
    # ...
```

### CSS personnalisé

```python
st.markdown("""
<style>
    /* Styles pour thème bleu clair */
    .stApp { background: #F8F9FA; }
    /* ... 280 lignes de CSS ... */
</style>
""", unsafe_allow_html=True)
```

### Gestion du workflow

**Vérifications** :
```python
if st.session_state.df is None:
    st.warning("Chargez d'abord des données")
else:
    # Logique de la page
```

---

## 📚 API Reference {#api}

### Fonctions utilitaires (models.py)

#### get_available_models()

```python
def get_available_models() -> dict
```

**Retour** :
```python
{
    'Régression': ['Liste', 'de', 'modèles'],
    'Classification': [...],
    'Clustering': [...]
}
```

#### create_model()

```python
def create_model(
    model_name: str,
    params: dict = None
) -> MLModelManager
```

**Factory function** : Crée et retourne un manager avec le modèle initialisé

**Mapping interne** :
```python
model_mapping = {
    'Régression Linéaire Simple': manager.linear_regression_simple,
    'Random Forest': manager.random_forest_classifier,
    # ...
}
```

---

## ✅ Bonnes pratiques {#bonnes-pratiques}

### 1. Gestion d'erreurs

```python
try:
    # Code potentiellement problématique
    df = load_data(file)
except Exception as e:
    st.error(f"Erreur : {str(e)}")
    return None
```

### 2. Normalisation

**TOUJOURS normaliser pour** :
- SVM
- KNN  
- Réseaux de neurones
- Modèles basés sur distances

**Optionnel pour** :
- Arbres de décision
- Random Forest
- Naïve Bayes

### 3. Validation croisée

```python
manager = create_model('Random Forest')
results = manager.cross_validate(X, y, cv=5)
print(f"Score moyen : {results['mean']:.3f} ± {results['std']:.3f}")
```

### 4. Sauvegarde de modèles

```python
# Entraînement
manager.train()

# Sauvegarde
manager.save_model('mon_modele.pkl')

# Rechargement ultérieur
new_manager = MLModelManager()
new_manager.load_model('mon_modele.pkl')
predictions = new_manager.predict(X_new)
```

### 5. Gestion de la mémoire

Pour gros datasets :
```python
# Sampling
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42)
```

### 6. Encodage robuste

```python
# Vérifier le type avant encodage
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
```

---

## 🐛 Débogage courant

### Problème : Accuracy 0%

**Cause** : Variable cible mal choisie (probablement un ID)

**Solution** :
```python
# Vérifier cardinalité
if len(np.unique(y)) > 50:
    st.warning("Trop de classes uniques, vérifiez la variable cible")
```

### Problème : RMSE très élevé

**Cause** : Échelle de la target non normalisée

**Solution** : Normaliser également y, puis dénormaliser les prédictions

### Problème : Modèle lent

**Causes possibles** :
1. Dataset trop grand → Sampling
2. SVM sur >10k échantillons → Essayer Random Forest
3. Trop de validations croisées → Réduire cv

---

**Documentation mise à jour le** : 2026-01-03
