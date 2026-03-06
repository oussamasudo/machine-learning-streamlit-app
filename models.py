"""
Module de gestion des modèles de Machine Learning
Contient tous les algorithmes et fonctionnalités nécessaires pour l'application
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)

# Importation des modèles
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier, MLPRegressor

import pickle
import warnings
warnings.filterwarnings('ignore')


class MLModelManager:
    """
    Classe principale pour gérer tous les modèles de Machine Learning
    """
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        self.is_fitted = False
        
    def prepare_data(self, X, y, test_size=0.2, random_state=42, scale=True):
        """
        Prépare les données pour l'entraînement
        
        Args:
            X: Features
            y: Target
            test_size: Taille du jeu de test
            random_state: Graine aléatoire
            scale: Normaliser les données ou non
        """
        # Division des données
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Normalisation
        if scale:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    # ==================== RÉGRESSION LINÉAIRE ====================
    
    def linear_regression_simple(self, params=None):
        """Régression linéaire simple"""
        self.model = LinearRegression()
        self.model_type = 'regression'
        return self.model
    
    def linear_regression_multiple(self, params=None):
        """Régression linéaire multiple"""
        self.model = LinearRegression()
        self.model_type = 'regression'
        return self.model
    
    def polynomial_regression(self, degree=2, params=None):
        """Régression polynomiale"""
        self.poly_features = PolynomialFeatures(degree=degree)
        self.model = LinearRegression()
        self.model_type = 'regression'
        self.is_polynomial = True
        return self.model
    
    def ridge_regression(self, alpha=1.0, params=None):
        """Régression Ridge (L2)"""
        params = params or {}
        self.model = Ridge(alpha=alpha, **params)
        self.model_type = 'regression'
        return self.model
    
    def lasso_regression(self, alpha=1.0, params=None):
        """Régression Lasso (L1)"""
        params = params or {}
        self.model = Lasso(alpha=alpha, **params)
        self.model_type = 'regression'
        return self.model
    
    # ==================== RÉGRESSION LOGISTIQUE ====================
    
    def logistic_regression(self, params=None):
        """Régression logistique pour la classification"""
        params = params or {}
        default_params = {
            'max_iter': 1000,
            'random_state': 42
        }
        default_params.update(params)
        self.model = LogisticRegression(**default_params)
        self.model_type = 'classification'
        return self.model
    
    # ==================== ARBRES DE DÉCISION ====================
    
    def decision_tree_classifier(self, params=None):
        """Arbre de décision pour la classification"""
        params = params or {}
        default_params = {
            'random_state': 42,
            'max_depth': None,
            'min_samples_split': 2
        }
        default_params.update(params)
        self.model = DecisionTreeClassifier(**default_params)
        self.model_type = 'classification'
        return self.model
    
    def decision_tree_regressor(self, params=None):
        """Arbre de décision pour la régression"""
        params = params or {}
        default_params = {
            'random_state': 42,
            'max_depth': None
        }
        default_params.update(params)
        self.model = DecisionTreeRegressor(**default_params)
        self.model_type = 'regression'
        return self.model
    
    # ==================== RANDOM FOREST ====================
    
    def random_forest_classifier(self, params=None):
        """Random Forest pour la classification"""
        params = params or {}
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': None
        }
        default_params.update(params)
        self.model = RandomForestClassifier(**default_params)
        self.model_type = 'classification'
        return self.model
    
    def random_forest_regressor(self, params=None):
        """Random Forest pour la régression"""
        params = params or {}
        default_params = {
            'n_estimators': 100,
            'random_state': 42
        }
        default_params.update(params)
        self.model = RandomForestRegressor(**default_params)
        self.model_type = 'regression'
        return self.model
    
    # ==================== GRADIENT BOOSTING ====================
    
    def gradient_boosting_classifier(self, params=None):
        """Gradient Boosting pour la classification"""
        params = params or {}
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
        default_params.update(params)
        self.model = GradientBoostingClassifier(**default_params)
        self.model_type = 'classification'
        return self.model
    
    def gradient_boosting_regressor(self, params=None):
        """Gradient Boosting pour la régression"""
        params = params or {}
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
        default_params.update(params)
        self.model = GradientBoostingRegressor(**default_params)
        self.model_type = 'regression'
        return self.model
    
    # ==================== SUPPORT VECTOR MACHINE ====================
    
    def svm_classifier(self, params=None):
        """SVM pour la classification"""
        params = params or {}
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'random_state': 42
        }
        default_params.update(params)
        self.model = SVC(**default_params)
        self.model_type = 'classification'
        return self.model
    
    def svm_regressor(self, params=None):
        """SVM pour la régression"""
        params = params or {}
        default_params = {
            'kernel': 'rbf',
            'C': 1.0
        }
        default_params.update(params)
        self.model = SVR(**default_params)
        self.model_type = 'regression'
        return self.model
    
    # ==================== K-NEAREST NEIGHBORS ====================
    
    def knn_classifier(self, params=None):
        """KNN pour la classification"""
        params = params or {}
        default_params = {
            'n_neighbors': 5,
            'metric': 'minkowski'
        }
        default_params.update(params)
        self.model = KNeighborsClassifier(**default_params)
        self.model_type = 'classification'
        return self.model
    
    def knn_regressor(self, params=None):
        """KNN pour la régression"""
        params = params or {}
        default_params = {
            'n_neighbors': 5
        }
        default_params.update(params)
        self.model = KNeighborsRegressor(**default_params)
        self.model_type = 'regression'
        return self.model
    
    # ==================== NAÏVE BAYES ====================
    
    def naive_bayes_gaussian(self, params=None):
        """Naïve Bayes Gaussien"""
        params = params or {}
        self.model = GaussianNB(**params)
        self.model_type = 'classification'
        return self.model
    
    def naive_bayes_multinomial(self, params=None):
        """Naïve Bayes Multinomial"""
        params = params or {}
        default_params = {'alpha': 1.0}
        default_params.update(params)
        self.model = MultinomialNB(**default_params)
        self.model_type = 'classification'
        return self.model
    
    def naive_bayes_bernoulli(self, params=None):
        """Naïve Bayes Bernoulli"""
        params = params or {}
        default_params = {'alpha': 1.0}
        default_params.update(params)
        self.model = BernoulliNB(**default_params)
        self.model_type = 'classification'
        return self.model
    
    # ==================== K-MEANS CLUSTERING ====================
    
    def kmeans_clustering(self, n_clusters=3, params=None):
        """K-Means pour le clustering"""
        params = params or {}
        default_params = {
            'n_clusters': n_clusters,
            'random_state': 42,
            'n_init': 10
        }
        default_params.update(params)
        self.model = KMeans(**default_params)
        self.model_type = 'clustering'
        return self.model
    
    # ==================== RÉSEAUX DE NEURONES ====================
    
    def neural_network_classifier(self, params=None):
        """Réseau de neurones pour la classification"""
        params = params or {}
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 500,
            'random_state': 42
        }
        default_params.update(params)
        self.model = MLPClassifier(**default_params)
        self.model_type = 'classification'
        return self.model
    
    def neural_network_regressor(self, params=None):
        """Réseau de neurones pour la régression"""
        params = params or {}
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 500,
            'random_state': 42
        }
        default_params.update(params)
        self.model = MLPRegressor(**default_params)
        self.model_type = 'regression'
        return self.model
    
    # ==================== ENTRAÎNEMENT ET PRÉDICTION ====================
    
    def train(self, X=None, y=None):
        """
        Entraîne le modèle
        
        Args:
            X: Features d'entraînement (optionnel si déjà préparées)
            y: Target d'entraînement (optionnel si déjà préparées)
        """
        if self.model is None:
            raise ValueError("Aucun modèle n'a été initialisé. Appelez d'abord une méthode de création de modèle.")
        
        X_train = X if X is not None else self.X_train
        y_train = y if y is not None else self.y_train
        
        # Gestion spéciale pour la régression polynomiale
        if hasattr(self, 'is_polynomial') and self.is_polynomial:
            X_train = self.poly_features.fit_transform(X_train)
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self.model
    
    def predict(self, X=None):
        """
        Fait des prédictions
        
        Args:
            X: Features pour la prédiction (utilise X_test par défaut)
        """
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné. Appelez d'abord la méthode train().")
        
        X_pred = X if X is not None else self.X_test
        
        # Gestion spéciale pour la régression polynomiale
        if hasattr(self, 'is_polynomial') and self.is_polynomial:
            X_pred = self.poly_features.transform(X_pred)
        
        self.predictions = self.model.predict(X_pred)
        return self.predictions
    
    # ==================== ÉVALUATION ====================
    
    def evaluate_regression(self, y_true=None, y_pred=None):
        """
        Évalue un modèle de régression
        
        Returns:
            dict: Métriques d'évaluation
        """
        y_true = y_true if y_true is not None else self.y_test
        y_pred = y_pred if y_pred is not None else self.predictions
        
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        return metrics
    
    def evaluate_classification(self, y_true=None, y_pred=None, average='weighted'):
        """
        Évalue un modèle de classification
        
        Returns:
            dict: Métriques d'évaluation
        """
        y_true = y_true if y_true is not None else self.y_test
        y_pred = y_pred if y_pred is not None else self.predictions
        
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average=average, zero_division=0),
            'Confusion Matrix': confusion_matrix(y_true, y_pred),
            'Classification Report': classification_report(y_true, y_pred, zero_division=0)
        }
        return metrics
    
    def evaluate_clustering(self, X=None, labels=None):
        """
        Évalue un modèle de clustering
        
        Returns:
            dict: Métriques d'évaluation
        """
        X = X if X is not None else self.X_test
        labels = labels if labels is not None else self.predictions
        
        metrics = {
            'Inertia': self.model.inertia_,
            'Silhouette Score': silhouette_score(X, labels)
        }
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Validation croisée
        
        Args:
            X: Features
            y: Target
            cv: Nombre de folds
            
        Returns:
            dict: Résultats de la validation croisée
        """
        if self.model_type == 'regression':
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        elif self.model_type == 'classification':
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        else:
            raise ValueError("La validation croisée n'est pas disponible pour le clustering")
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    def grid_search(self, X, y, param_grid, cv=5):
        """
        Recherche des meilleurs hyperparamètres
        
        Args:
            X: Features
            y: Target
            param_grid: Grille de paramètres
            cv: Nombre de folds
            
        Returns:
            dict: Meilleurs paramètres et score
        """
        if self.model_type == 'regression':
            scoring = 'r2'
        elif self.model_type == 'classification':
            scoring = 'accuracy'
        else:
            raise ValueError("GridSearch n'est pas disponible pour le clustering")
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    # ==================== SAUVEGARDE ET CHARGEMENT ====================
    
    def save_model(self, filepath):
        """
        Sauvegarde le modèle
        
        Args:
            filepath: Chemin du fichier
        """
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas entraîné")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'scaler': self.scaler,
        }
        
        if hasattr(self, 'is_polynomial'):
            model_data['poly_features'] = self.poly_features
            model_data['is_polynomial'] = self.is_polynomial
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """
        Charge un modèle sauvegardé
        
        Args:
            filepath: Chemin du fichier
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.scaler = model_data['scaler']
        self.is_fitted = True
        
        if 'is_polynomial' in model_data:
            self.is_polynomial = model_data['is_polynomial']
            self.poly_features = model_data['poly_features']
    
    # ==================== INFORMATIONS SUR LE MODÈLE ====================
    
    def get_feature_importance(self):
        """
        Obtient l'importance des features (pour les modèles qui le supportent)
        
        Returns:
            array: Importance des features
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            coef = np.abs(self.model.coef_)
            if coef.ndim > 1:
                return np.mean(coef, axis=0)
            return coef
        else:
            return None
    
    def get_model_params(self):
        """
        Obtient les paramètres du modèle
        
        Returns:
            dict: Paramètres du modèle
        """
        if self.model is not None:
            return self.model.get_params()
        return None


# ==================== FONCTIONS UTILITAIRES ====================

def get_available_models():
    """
    Retourne la liste des modèles disponibles
    
    Returns:
        dict: Dictionnaire des modèles par catégorie
    """
    return {
        'Régression': [
            'Régression Linéaire Simple',
            'Régression Linéaire Multiple',
            'Régression Polynomiale',
            'Régression Ridge',
            'Régression Lasso',
            'Arbre de Décision (Régression)',
            'Random Forest (Régression)',
            'Gradient Boosting (Régression)',
            'SVM (Régression)',
            'KNN (Régression)',
            'Réseau de Neurones (Régression)'
        ],
        'Classification': [
            'Régression Logistique',
            'Arbre de Décision',
            'Random Forest',
            'Gradient Boosting',
            'SVM',
            'KNN',
            'Naïve Bayes Gaussien',
            'Réseau de Neurones'
        ],
        'Clustering': [
            'K-Means'
        ]
    }


def create_model(model_name, params=None):
    """
    Crée un modèle selon son nom
    
    Args:
        model_name: Nom du modèle
        params: Paramètres du modèle
        
    Returns:
        MLModelManager: Instance du gestionnaire avec le modèle initialisé
    """
    manager = MLModelManager()
    
    model_mapping = {
        'Régression Linéaire Simple': manager.linear_regression_simple,
        'Régression Linéaire Multiple': manager.linear_regression_multiple,
        'Régression Polynomiale': manager.polynomial_regression,
        'Régression Ridge': manager.ridge_regression,
        'Régression Lasso': manager.lasso_regression,
        'Régression Logistique': manager.logistic_regression,
        'Arbre de Décision': manager.decision_tree_classifier,
        'Arbre de Décision (Régression)': manager.decision_tree_regressor,
        'Random Forest': manager.random_forest_classifier,
        'Random Forest (Régression)': manager.random_forest_regressor,
        'SVM': manager.svm_classifier,
        'SVM (Régression)': manager.svm_regressor,
        'KNN': manager.knn_classifier,
        'KNN (Régression)': manager.knn_regressor,
        'Naïve Bayes Gaussien': manager.naive_bayes_gaussian,
        'Naïve Bayes Multinomial': manager.naive_bayes_multinomial,
        'Naïve Bayes Bernoulli': manager.naive_bayes_bernoulli,
        'K-Means': manager.kmeans_clustering,
        'Gradient Boosting': manager.gradient_boosting_classifier,
        'Gradient Boosting (Régression)': manager.gradient_boosting_regressor,
        'Réseau de Neurones': manager.neural_network_classifier,
        'Réseau de Neurones (Régression)': manager.neural_network_regressor
    }
    
    if model_name in model_mapping:
        if params:
            model_mapping[model_name](params=params)
        else:
            model_mapping[model_name]()
    else:
        raise ValueError(f"Modèle '{model_name}' non reconnu")
    
    return manager