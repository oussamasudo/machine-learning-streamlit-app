# 📖 Tutoriels - Algorithmes de Machine Learning

Ce document contient des tutoriels détaillés pour chaque algorithme implémenté dans l'application.

---

## 📋 Table des matières

1. [Régression Linéaire](#regression-lineaire)
2. [Random Forest](#random-forest)
3. [SVM (Support Vector Machine)](#svm)
4. [K-Means Clustering](#kmeans)
5. [Réseaux de Neurones](#reseaux-neurones)
6. [Arbre de Décision](#arbre-decision)
7. [KNN (K-Nearest Neighbors)](#knn)
8. [Naïve Bayes](#naive-bayes)

---

## 1. Régression Linéaire {#regression-lineaire}

### 📚 **Théorie**

La régression linéaire

 modélise la relation entre une variable dépendante (y) et une ou plusieurs variables indépendantes (X) par une équation linéaire.

**Formule** : `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε`

### 🎯 **Quand l'utiliser ?**

- Relations linéaires entre variables
- Prédiction de valeurs continues
- Compréhension de l'impact des features

### ✅ **Avantages**
- Simple et interprétable
- Rapide à entraîner
- Peu de paramètres

### ❌ **Inconvénients**
- Suppose une relation linéaire
- Sensible aux outliers
- Ne capture pas les interactions complexes

### 🔧 **Tutoriel pratique**

#### Étape 1 : Préparation des données
```
Dataset recommandé : Diabetes (dataset exemple)
Variable cible : target (progression de la maladie)
Features : Toutes les autres colonnes
```

#### Étape 2 : Prétraitement
1. Vérifiez les valeurs manquantes → Aucune normalement
2. Normalisez avec **StandardScaler**
3. Taille du test : 20%

#### Étape 3 : Configuration
- Algorithme : **Régression Linéaire Simple**
- Pas de paramètres à ajuster
- Normalisation : **Activée** ✅

#### Étape 4 : Interprétation
- **R²** : Proche de 1 = bon modèle (ex: 0.52 = 52% de variance expliquée)
- **RMSE** : Plus petit = meilleures prédictions
- **Graphique résidus** : Points dispersés aléatoirement = bon modèle

### 📊 **Exemple de résultats**

```
R² : 0.52
RMSE : 54.2
MAE : 43.1
```

**Interprétation** : Le modèle explique 52% de la variance. Amélioration possible avec régression polynomiale ou Random Forest.

---

## 2. Random Forest {#random-forest}

### 📚 **Théorie**

Random Forest est un ensemble d'arbres de décision qui vote pour la prédiction finale.

**Principe** : "Wisdom of the crowd" - Plusieurs arbres faibles font un modèle fort.

### 🎯 **Quand l'utiliser ?**

- Données tabulaires avec features mixtes
- Classification ou régression
- Robustesse nécessaire (peu sensible au surapprentissage)
- Importance des features requise

### ✅ **Avantages**
- Très performant "out of the box"
- Gère bien les valeurs manquantes
- Importance des features automatique
- Peu de tuning nécessaire

### ❌ **Inconvénients**  
- Moins interprétable qu'un arbre unique
- Plus lent que modèles linéaires
- Mémoire importante pour grands datasets

### 🔧 **Tutoriel pratique**

#### Étape 1 : Préparation
```
Dataset : Iris (classification)
Variable cible : target (espèce de fleur)
Features : sepal length, sepal width, petal length, petal width
```

#### Étape 2 : Configuration
- Algorithme : **Random Forest** (classification)
- **Nombre d'arbres** : 100 (défaut, augmenter si sous-apprentissage)
- **Profondeur max** : 10 (limiter si surapprentissage)
- Normalisation : Optionnelle pour Random Forest

#### Étape 3 : Entraînement
- Lancez l'entraînement
- Temps : Quelques secondes

#### Étape 4 : Évaluation
- **Accuracy** : Devrait être > 95% sur Iris
- **Matrice de confusion** : Peu d'erreurs
- **Importance des features** : Petal length et width dominent

### 📊 **Exemple de résultats**

```
Accuracy : 97%
Precision : 97%
F1-Score : 97%

Importance des features :
- petal length : 0.45
- petal width : 0.42
- sepal length : 0.08
- sepal width : 0.05
```

**Interprétation** : Excellente performance ! Les pétales sont plus discriminants que les sépales.

---

## 3. SVM (Support Vector Machine) {#svm}

### 📚 **Théorie**

SVM trouve l'hyperplan optimal qui sépare les classes avec la marge maximale.

**Astuce** : Le "kernel trick" permet de traiter des frontières non-linéaires.

### 🎯 **Quand l'utiliser ?**

- Classification binaire ou multi-classes
- Données en haute dimension
- Frontières de décision complexes
- Dataset de petite/moyenne taille

### ✅ **Avantages**
- Très efficace en haute dimension
- Marge maximale = généralisation robuste
- Kernel trick pour non-linéarité
- Bon pour texte, images

### ❌ **Inconvénients**
- Lent sur gros datasets (>10k échantillons)
- Sensible au choix du kernel et de C
- Nécessite normalisation
- Moins interprétable

### 🔧 **Tutoriel pratique**

#### Configuration recommandée
- **Kernel** : 
  - `rbf` (défaut) → Bonne option générale
  - `linear` → Si données linéairement séparables
  - `poly` → Rarement utilisé
- **C** : Régularisation
  - C = 0.1 → Forte régularisation (simple)
  - C = 1.0 → Équilibré (défaut)
  - C = 10.0 → Faible régularisation (complexe)

#### Étape 1 : Prétraitement **OBLIGATOIRE**
```
⚠️ CRITIQUE : SVM nécessite normalisation !
→ Utilisez StandardScaler
```

#### Étape 2 : Tuning
1. Commencez avec `C=1.0` et `kernel=rbf`
2. Si sous-apprentissage → Augmentez C ou changez kernel
3. Si surapprentissage → Diminuez C

### 📊 **Exemple de résultats**

```
Avec kernel='rbf', C=1.0 :
Accuracy : 95%

Avec kernel='linear', C=1.0 :
Accuracy : 93%
```

---

## 4. K-Means Clustering {#kmeans}

### 📚 **Théorie**

K-Means groupe les points similaires en K clusters en minimisant la distance intra-cluster.

**Algorithme** :
1. Initialiser K centres aléatoirement
2. Assigner chaque point au centre le plus proche
3. Recalculer les centres
4. Répéter jusqu'à convergence

### 🎯 **Quand l'utiliser ?**

- Segmentation de clients
- Compression d'images
- Détection d'anomalies
- Exploration de données non-étiquetées

### 🔧 **Tutoriel pratique**

#### Étape 1 : Choix du nombre de clusters

**Méthode du coude (Elbow method)** :
1. Testez K = 2, 3, 4, 5, 6...
2. Tracez Inertia vs K
3. Cherchez le "coude" (cassure)

#### Étape 2 : Configuration
- **Nombre de clusters** : Selon méthode du coude
- Normalisation : **Recommandée** (distances importantes)

#### Étape 3 : Évaluation
- **Silhouette Score** : [-1, 1]
  - > 0.5 : Bonne séparation
  - 0.2 - 0.5 : Acceptable
  - < 0.2 : Mauvais clustering
- **Inertia** : Plus petit = meilleur (mais attention au surapprentissage)

### 📊 **Exemple**

```
Pour 3 clusters sur Iris :
Silhouette Score : 0.55 (Bon !)
Inertia : 78.85

Visualisation PCA : 3 groupes distincts visibles
```

---

## 5. Réseaux de Neurones {#reseaux-neurones}

### 📚 **Théorie**

Modèle inspiré du cerveau avec couches de neurones interconnectés.

**Architecture** : Input → Hidden Layers → Output

### 🎯 **Quand l'utiliser ?**

- Problèmes complexes non-linéaires
- Beaucoup de données disponibles
- Features complexes (images, texte après embedding)

### ✅ **Avantages**
- Très flexible
- Capture interactions complexes
- Performant avec beaucoup de données

### ❌ **Inconvénients**
- Nécessite beaucoup de données
- Long à entraîner
- Difficile à interpréter
- Beaucoup d'hyperparamètres

### 🔧 **Tutoriel pratique**

#### Configuration
- **Couche 1** : 100 neurones (défaut)
- **Couche 2** : 50 neurones
- **Activation** : ReLU (défaut, bon choix général)
- **Solver** : Adam (optimiseur adaptatif)
- **Max iterations** : 500 (augmenter si pas convergé)

#### Conseils
1. **Normalisation** : OBLIGATOIRE (StandardScaler)
2. **Taille dataset** : >1000 échantillons recommandé
3. **Overfitting** : Réduire nb de neurones ou ajouter régularisation

### 📊 **Courbe d'apprentissage**

Consultez "Courbe d'apprentissage" dans l'évaluation :
- Descendante → Bon apprentissage
- Plateau → Convergé
- Fluctuante → Réduire learning rate

---

## 6. Arbre de Décision {#arbre-decision}

### 📚 **Théorie**

Structure d'arbre avec tests sur les features pour prendre des décisions.

**Principe** : Diviser récursivement les données pour maximiser la pureté.

### 🎯 **Quand l'utiliser ?**

- Interprétabilité cruciale
- Règles de décision explicites nécessaires
- Features catégorielles
- Prototype rapide

### ✅ **Avantages**
- Très interprétable (visualisation d'arbre)
- Pas de normalisation nécessaire
- Gère features mixtes
- Rapide

### ❌ **Inconvénients**
- Prone au surapprentissage
- Instable (petits changements → gros impact)
- Moins performant que ensemble methods

### 🔧 **Contrôle du surapprentissage**

- **Profondeur max** : Limiter à 5-10
- **Min samples split** : Augmenter à 10-20
- **Min samples leaf** : Augmenter à 5-10

### 📊 **Visualisation**

Utilisez "Visualisation de l'arbre" dans l'évaluation pour :
- Comprendre les décisions
- Identifier features importantes
- Expliquer aux non-techniques

---

## 7. KNN (K-Nearest Neighbors) {#knn}

### 📚 **Théorie**

Classifie un point selon la majorité de ses K voisins les plus proches.

**Principe** : "Qui se ressemble s'assemble"

### 🎯 **Quand l'utiliser ?**

- Petits datasets
- Frontières de décision irrégulières
- Recommandations (similarité)

### 🔧 **Choix de K**

- **K petit (1-3)** : Frontières complexes, risque d'overfitting
- **K moyen (5-10)** : Équilibré (recommandé)
- **K grand (>15)** : Lisse, risque d'underfitting

**Règle** : K = √n (n = nombre d'échantillons)

### ⚠️ **Points importants**

- Normalisation **OBLIGATOIRE** (distances)
- Lent sur gros datasets (calcule toutes les distances)
- Curse of dimensionality (>20 features)

---

## 8. Naïve Bayes {#naive-bayes}

### 📚 **Théorie**

Basé sur le théorème de Bayes avec l'hypothèse d'indépendance des features.

**Formule** : P(y|X) = P(X|y) × P(y) / P(X)

### 🎯 **Quand l'utiliser ?**

- Classification de texte (spam, sentiment)
- Petits datasets
- Features indépendantes
- Rapidité requise

### 🔧 **Variantes**

- **Gaussien** : Features continues (distribution normale)
- **Multinomial** : Comptages (mots dans texte)
- **Bernoulli** : Features binaires (présence/absence)

### ✅ **Avantages**
- Très rapide
- Fonctionne bien avec peu de données
- Simple
- Bon pour texte

---

## 💡 Conseils généraux

### Choix d'algorithme

1. **Régression** : Linéaire → Polynomial → Random Forest
2. **Classification** : Logistic → Random Forest → SVM
3. **Clustering** : K-Means

### Workflow recommandé

1. **Baseline** : Modèle simple (Linear, Logistic)
2. **Amélioration** : Random Forest
3. **Fine-tuning** : SVM ou Neural Network si nécessaire

### Éviter le surapprentissage

- Validation croisée
- Régularisation (Ridge, Lasso, L2)
- Simplifier le modèle (réduire profondeur, neurones)
- Plus de données

---

**Bonne pratique du Machine Learning ! 🚀**
