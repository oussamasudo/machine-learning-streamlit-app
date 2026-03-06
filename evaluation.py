"""
Module d'évaluation des modèles de Machine Learning
Contient les fonctions pour évaluer les performances des modèles
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    # Métriques de régression
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    
    # Métriques de classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    
    # Métriques de clustering
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true, y_pred, model_type='regression', **kwargs):
    """
    Fonction principale d'évaluation des modèles
    
    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites
        model_type: Type de modèle ('regression', 'classification', 'clustering')
        **kwargs: Arguments supplémentaires
        
    Returns:
        dict: Dictionnaire contenant les métriques d'évaluation
    """
    if model_type == 'regression':
        return evaluate_regression(y_true, y_pred)
    elif model_type == 'classification':
        return evaluate_classification(y_true, y_pred, **kwargs)
    elif model_type == 'clustering':
        X = kwargs.get('X', None)
        if X is None:
            raise ValueError("X est requis pour l'évaluation du clustering")
        return evaluate_clustering(X, y_pred)
    else:
        raise ValueError(f"Type de modèle non reconnu: {model_type}")


def evaluate_regression(y_true, y_pred):
    """
    Évalue un modèle de régression
    
    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites
        
    Returns:
        dict: Métriques de régression
    """
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
    }
    
    # MAPE seulement si pas de zéros dans y_true
    if not np.any(y_true == 0):
        try:
            metrics['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
        except:
            pass
    
    # Erreur relative moyenne
    metrics['Mean_Relative_Error'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)))
    
    return metrics


def evaluate_classification(y_true, y_pred, average='weighted', y_proba=None):
    """
    Évalue un modèle de classification
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        average: Méthode de moyennage ('binary', 'weighted', 'macro', 'micro')
        y_proba: Probabilités prédites (optionnel, pour ROC-AUC)
        
    Returns:
        dict: Métriques de classification
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'F1_Score': f1_score(y_true, y_pred, average=average, zero_division=0),
        'Confusion_Matrix': confusion_matrix(y_true, y_pred),
        'Classification_Report': classification_report(y_true, y_pred, zero_division=0)
    }
    
    # ROC-AUC si probabilités fournies
    if y_proba is not None:
        try:
            # Pour classification binaire
            if len(np.unique(y_true)) == 2:
                if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
                    metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba)
            # Pour classification multiclasse
            else:
                metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba, 
                                                   multi_class='ovr', average=average)
        except:
            pass
    
    return metrics


def evaluate_clustering(X, labels):
    """
    Évalue un modèle de clustering (K-Means, etc.)

    Args:
        X: array-like ou DataFrame (features)
        labels: array-like (labels de clusters)

    Returns:
        dict: métriques de clustering
    """

    metrics = {}

    # ---------- Vérifications ----------
    if X is None or labels is None:
        return {
            "Error": "X ou labels est None. Vérifiez l'entraînement du modèle."
        }

    # Convertir DataFrame → numpy array
    if isinstance(X, pd.DataFrame):
        X = X.select_dtypes(include=[np.number]).values

    # Vérification dimensions
    if len(X) != len(labels):
        return {
            "Error": "X et labels n'ont pas la même taille."
        }

    n_clusters = len(np.unique(labels))
    metrics["N_Clusters"] = n_clusters

    # Distribution des clusters
    unique, counts = np.unique(labels, return_counts=True)
    metrics["Cluster_Sizes"] = dict(zip(unique.tolist(), counts.tolist()))

    # ---------- Métriques ----------
    # Silhouette Score (≥ 2 clusters requis)
    if n_clusters > 1:
        try:
            metrics["Silhouette_Score"] = silhouette_score(X, labels)
        except ValueError:
            metrics["Silhouette_Score"] = None
    else:
        metrics["Silhouette_Score"] = None

    # Davies-Bouldin Index (≥ 2 clusters requis)
    if n_clusters > 1:
        try:
            metrics["Davies_Bouldin_Index"] = davies_bouldin_score(X, labels)
        except ValueError:
            metrics["Davies_Bouldin_Index"] = None
    else:
        metrics["Davies_Bouldin_Index"] = None

    # Calinski-Harabasz Score (≥ 2 clusters requis)
    if n_clusters > 1:
        try:
            metrics["Calinski_Harabasz_Score"] = calinski_harabasz_score(X, labels)
        except ValueError:
            metrics["Calinski_Harabasz_Score"] = None
    else:
        metrics["Calinski_Harabasz_Score"] = None

    return metrics


def compare_models(results_dict):
    """
    Compare les résultats de plusieurs modèles
    
    Args:
        results_dict: Dictionnaire {nom_modele: métriques}
        
    Returns:
        DataFrame: Tableau comparatif des modèles
    """
    comparison_data = []
    
    for model_name, metrics in results_dict.items():
        row = {'Model': model_name}
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                row[metric_name] = value
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df


def plot_regression_results(y_true, y_pred, title="Résultats de régression"):
    """
    Visualise les résultats d'un modèle de régression
    
    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites
        title: Titre du graphique
        
    Returns:
        fig: Figure matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Prédictions vs Réel
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
    axes[0, 0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Valeurs réelles')
    axes[0, 0].set_ylabel('Prédictions')
    axes[0, 0].set_title('Prédictions vs Valeurs réelles')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Résidus
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Prédictions')
    axes[0, 1].set_ylabel('Résidus')
    axes[0, 1].set_title('Graphique des résidus')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution des résidus
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Résidus')
    axes[1, 0].set_ylabel('Fréquence')
    axes[1, 0].set_title('Distribution des résidus')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Matrice de confusion"):
    """
    Visualise la matrice de confusion
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        labels: Noms des classes (optionnel)
        title: Titre du graphique
        
    Returns:
        fig: Figure matplotlib
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Prédictions')
    ax.set_ylabel('Valeurs réelles')
    ax.set_title(title)
    
    return fig


def plot_roc_curve(y_true, y_proba, title="Courbe ROC"):
    """
    Visualise la courbe ROC
    
    Args:
        y_true: Labels réels
        y_proba: Probabilités prédites
        title: Titre du graphique
        
    Returns:
        fig: Figure matplotlib
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Aléatoire')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taux de faux positifs')
    ax.set_ylabel('Taux de vrais positifs')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_feature_importance(feature_names, importance_values, top_n=10, 
                           title="Importance des features"):
    """
    Visualise l'importance des features
    
    Args:
        feature_names: Noms des features
        importance_values: Valeurs d'importance
        top_n: Nombre de features à afficher
        title: Titre du graphique
        
    Returns:
        fig: Figure matplotlib
    """
    # Créer un DataFrame et trier
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values.flatten() if len(importance_values.shape) > 1 else importance_values
    }).sort_values('Importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df['Feature'], df['Importance'])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    return fig


def plot_learning_curve(train_scores, val_scores, title="Courbe d'apprentissage"):
    """
    Visualise la courbe d'apprentissage
    
    Args:
        train_scores: Scores d'entraînement
        val_scores: Scores de validation
        title: Titre du graphique
        
    Returns:
        fig: Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_scores) + 1)
    
    ax.plot(epochs, train_scores, 'b-', label='Score d\'entraînement', linewidth=2)
    ax.plot(epochs, val_scores, 'r-', label='Score de validation', linewidth=2)
    
    ax.set_xlabel('Époques')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def get_classification_metrics_summary(y_true, y_pred, class_names=None):
    """
    Génère un résumé détaillé des métriques de classification
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        class_names: Noms des classes (optionnel)
        
    Returns:
        str: Résumé formaté
    """
    metrics = evaluate_classification(y_true, y_pred)
    
    summary = "=" * 50 + "\n"
    summary += "RÉSUMÉ DES MÉTRIQUES DE CLASSIFICATION\n"
    summary += "=" * 50 + "\n\n"
    
    summary += f"Accuracy:  {metrics['Accuracy']:.4f}\n"
    summary += f"Precision: {metrics['Precision']:.4f}\n"
    summary += f"Recall:    {metrics['Recall']:.4f}\n"
    summary += f"F1-Score:  {metrics['F1_Score']:.4f}\n"
    
    if 'ROC_AUC' in metrics:
        summary += f"ROC-AUC:   {metrics['ROC_AUC']:.4f}\n"
    
    summary += "\n" + "=" * 50 + "\n"
    summary += "RAPPORT DE CLASSIFICATION DÉTAILLÉ\n"
    summary += "=" * 50 + "\n"
    summary += metrics['Classification_Report']
    
    return summary


def get_regression_metrics_summary(y_true, y_pred):
    """
    Génère un résumé détaillé des métriques de régression
    
    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites
        
    Returns:
        str: Résumé formaté
    """
    metrics = evaluate_regression(y_true, y_pred)
    
    summary = "=" * 50 + "\n"
    summary += "RÉSUMÉ DES MÉTRIQUES DE RÉGRESSION\n"
    summary += "=" * 50 + "\n\n"
    
    summary += f"R² Score:          {metrics['R2']:.4f}\n"
    summary += f"MSE:               {metrics['MSE']:.4f}\n"
    summary += f"RMSE:              {metrics['RMSE']:.4f}\n"
    summary += f"MAE:               {metrics['MAE']:.4f}\n"
    
    if 'MAPE' in metrics:
        summary += f"MAPE:              {metrics['MAPE']:.4f}\n"
    
    summary += f"Erreur Relative:   {metrics['Mean_Relative_Error']:.4f}\n"
    
    summary += "\n" + "=" * 50 + "\n"
    
    return summary


def cross_validation_summary(cv_scores):
    """
    Génère un résumé de la validation croisée
    
    Args:
        cv_scores: Array des scores de validation croisée
        
    Returns:
        dict: Statistiques de validation croisée
    """
    return {
        'Mean_Score': np.mean(cv_scores),
        'Std_Score': np.std(cv_scores),
        'Min_Score': np.min(cv_scores),
        'Max_Score': np.max(cv_scores),
        'All_Scores': cv_scores.tolist()
    }


def calculate_prediction_intervals(y_pred, residuals, confidence=0.95):
    """
    Calcule les intervalles de prédiction
    
    Args:
        y_pred: Prédictions
        residuals: Résidus (y_true - y_pred)
        confidence: Niveau de confiance (défaut: 0.95)
        
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    from scipy import stats
    
    # Calculer l'écart-type des résidus
    std_residuals = np.std(residuals)
    
    # Valeur critique de la distribution t
    dof = len(residuals) - 1
    t_val = stats.t.ppf((1 + confidence) / 2, dof)
    
    # Calculer les bornes
    margin = t_val * std_residuals
    lower_bound = y_pred - margin
    upper_bound = y_pred + margin
    
    return lower_bound, upper_bound