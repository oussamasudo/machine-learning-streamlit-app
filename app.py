import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from models import MLModelManager, get_available_models, create_model

# Import des modules du projet
from data_loader import load_data
from preprocessing import handle_missing_values, normalize_features
from models import MLModelManager, get_available_models, create_model
from evaluation import evaluate_model

# Configuration
st.set_page_config(page_title="ML Application", layout="wide", page_icon="🤖")

# ===== CSS PERSONNALISÉ - THÈME BLEU CLAIR PROFESSIONNEL =====
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Variables CSS - Palette Bleue Claire */
    :root {
        --primary-blue: #4A90E2;
        --light-blue: #B3D9F2;
        --sidebar-blue: #ADD8E6;
        --dark-text: #2c3e50;
        --light-bg: #F5F7FA;
    }
    
    /* Style général - Fond blanc/gris très clair */
    .stApp {
        background: #F8F9FA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Titres simples et élégants */
    h1 {
        color: #2c3e50 !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    h2 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        font-size: 1.6rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: #34495e !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
    }
    
    /* Texte général */
    p, label, span, div {
        color: #2c3e50 !important;
    }
    
    /* Sidebar bleue claire */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #C8E6F5 0%, #ADD8E6 100%);
        border-right: 1px solid #9EC8DC;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.7);
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.4rem;
        border: 1px solid transparent;
        transition: all 0.3s ease;
        color: #2c3e50 !important;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.9);
        border-color: #4A90E2;
        transform: translateX(5px);
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
        background: #4A90E2;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
    }
    
    /* Boutons bleus professionnels */
    .stButton > button {
        background: #4A90E2;
        color: white !important;
        border: none;
        border-radius: 0.5rem;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(74, 144, 226, 0.3);
    }
    
    .stButton > button:hover {
        background: #357ABD;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4);
    }
    
    /* Métriques bleues */
    [data-testid="stMetricValue"] {
        color: #4A90E2 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #5a6c7d !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    /* Cartes propres */
    [data-testid="stExpander"] {
        background: white;
        border: 1px solid #e1e8ed;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stExpander"]:hover {
        border-color: #4A90E2;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.15);
    }
    
    /* Tabs bleues */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        border-bottom: 2px solid #e1e8ed;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #5a6c7d;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        border-radius: 0.3rem 0.3rem 0 0;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #F5F7FA;
        color: #2c3e50;
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #4A90E2 !important;
        border-bottom: 3px solid #4A90E2;
        font-weight: 600;
    }
    
    /* DataFrames propres */
    [data-testid="stDataFrame"] {
        background: white;
        border: 1px solid #e1e8ed;
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div {
        background: white !important;
        border: 1px solid #D1D9E0 !important;
        border-radius: 0.4rem !important;
        color: #2c3e50 !important;
        padding: 0.6rem 1rem !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #4A90E2 !important;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1) !important;
    }
    
    /* Sliders bleus */
    .stSlider > div > div > div {
        background: #4A90E2;
    }
    
    /* File uploader avec zone bleue claire */
    [data-testid="stFileUploader"] {
        background: #E8F4FA;
        border: 2px dashed #4A90E2;
        border-radius: 0.5rem;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: #D6EDF7;
        border-color: #357ABD;
    }
    
    /* Messages d'alerte professionnels */
    .stSuccess {
        background: #d4edda !important;
        border-left: 4px solid #28a745 !important;
        border-radius: 0.3rem !important;
        color: #155724 !important;
        padding: 1rem !important;
    }
    
    .stWarning {
        background: #fff3cd !important;
        border-left: 4px solid #ffc107 !important;
        border-radius: 0.3rem !important;
        color: #856404 !important;
        padding: 1rem !important;
    }
    
    .stError {
        background: #f8d7da !important;
        border-left: 4px solid #dc3545 !important;
        border-radius: 0.3rem !important;
        color: #721c24 !important;
        padding: 1rem !important;
    }
    
    .stInfo {
        background: #d1ecf1 !important;
        border-left: 4px solid #17a2b8 !important;
        border-radius: 0.3rem !important;
        color: #0c5460 !important;
        padding: 1rem !important;
    }
    
    /* Animations subtiles */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .element-container {
        animation: fadeIn 0.3s ease-out;
    }
    
    /* Scrollbar bleue */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F5F7FA;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #B3D9F2;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4A90E2;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: #4A90E2;
    }
</style>
""", unsafe_allow_html=True)

st.image("logo.jpg", use_container_width=True)


# Initialisation session state
if "df" not in st.session_state:
    st.session_state.df = None
if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "model_manager" not in st.session_state:
    st.session_state.model_manager = None
if "trained" not in st.session_state:
    st.session_state.trained = False

from streamlit_option_menu import option_menu

with st.sidebar:
    page = option_menu(
        menu_title="Navigation",
        options=[
            "Chargement des données",
            "Exploration",
            "Prétraitement",
            "Modélisation",
            "Évaluation",
            "Export",
            "Documentation"
        ],
        icons=[
            "cloud-upload",
            "bar-chart",
            "sliders",
            "cpu",
            "check-circle",
            "cloud-download",
            "book"

        ],
        menu_icon="list",
        default_index=3,
        styles={
            "container": {
                "padding": "10px",
                "background-color": "#9fd3f2"
            },
            "icon": {
                "color": "#0f172a",
                "font-size": "20px"
            },
            "nav-link": {
                "font-size": "16px",
                "margin": "6px 0",
                "border-radius": "10px",
                "color": "#0f172a"
            },
            "nav-link-selected": {
                "background-color": "#2563eb",
                "color": "white",
                "font-weight": "600"
            }
        }
    )



# ==================== CHARGEMENT ====================
if page == "Chargement des données":
    st.header("Chargement des données")
    
    source = st.radio("Source", ["Fichier local", "URL", "Dataset exemple"])
    
    if source == "Fichier local":
        uploaded_file = st.file_uploader(
            "Charger un fichier CSV ou Excel",
            type=["csv", "xlsx", "xls"]
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                st.session_state.df_original = df.copy()
                st.success("Dataset chargé avec succès")
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    
    elif source == "URL":
        url = st.text_input("URL du fichier CSV")
        if st.button("Charger"):
            try:
                df = pd.read_csv(url)
                st.session_state.df = df
                st.session_state.df_original = df.copy()
                st.success("Dataset chargé")
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    
    else:
        dataset = st.selectbox("Choisir un dataset", ["Iris", "Diabetes"])
        if st.button("Charger"):
            try:
                if dataset == "Iris":
                    from sklearn.datasets import load_iris
                    data = load_iris()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df["target"] = data.target
                else:
                    from sklearn.datasets import load_diabetes
                    data = load_diabetes()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df["target"] = data.target
                
                st.session_state.df = df
                st.session_state.df_original = df.copy()
                st.success("Dataset chargé")
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    
    # ==================== AFFICHAGE DES DONNÉES ====================
    if st.session_state.df is not None:
        st.subheader("Aperçu des données")
        st.dataframe(st.session_state.df.head(20))

        st.subheader("Types de données")
        types_df = st.session_state.df.dtypes.astype(str).reset_index()
        types_df.columns = ["Colonne", "Type de donnée"]

        st.dataframe(types_df, use_container_width=True)


        st.subheader("Statistiques descriptives")
        st.write(st.session_state.df.describe())


# ==================== EXPLORATION ====================
elif page == "Exploration":
    if st.session_state.df is None:
        st.warning("Veuillez charger un dataset")
    else:
        st.header("Exploration des données")
        df = st.session_state.df
        
        tab1, tab2, tab3 = st.tabs(["Statistiques", "Distributions", "Corrélations"])
        
        with tab1:
            st.dataframe(df.describe())
            st.write(f"Valeurs manquantes: {df.isnull().sum().sum()}")
            st.write(f"Doublons: {df.duplicated().sum()}")
        
        with tab2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Sélectionner une colonne", numeric_cols)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.hist(df[col].dropna(), bins=30, edgecolor='black')
                ax1.set_title(f"Histogramme - {col}")
                ax1.set_xlabel(col)
                
                ax2.boxplot(df[col].dropna())
                ax2.set_title(f"Box Plot - {col}")
                ax2.set_ylabel(col)
                
                st.pyplot(fig)
                plt.close()
        
        with tab3:
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
                st.pyplot(fig)
                plt.close()

# ==================== PRÉTRAITEMENT ====================
elif page == "Prétraitement":
    if st.session_state.df is None:
        st.warning("Veuillez charger un dataset")
    else:
        st.header("Prétraitement des données")
        df = st.session_state.df.copy()
        
        # Valeurs manquantes
        st.subheader("1. Valeurs manquantes")
        missing = df.isnull().sum().sum()
        st.write(f"Total: {missing}")
        
        if missing > 0:
            method = st.selectbox("Méthode", [
                "Supprimer les lignes",
                "Moyenne",
                "Médiane",
                "Valeur fixe"
            ])
            
            if st.button("Appliquer"):
                df = handle_missing_values(df, method=method.lower())
                st.session_state.df = df
                st.success("Valeurs manquantes traitées")
        
        # Doublons
        st.subheader("2. Doublons")
        duplicates = df.duplicated().sum()
        st.write(f"Total: {duplicates}")
        
        if duplicates > 0 and st.button("Supprimer doublons"):
            df = df.drop_duplicates()
            st.session_state.df = df
            st.success("Doublons supprimés")
        
        # Encodage
        st.subheader("3. Encodage des variables catégorielles")
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if cat_cols:
            st.write(f"Colonnes: {', '.join(cat_cols)}")
            encoding = st.radio("Méthode", ["Label Encoding", "One-Hot Encoding"])
            
            if st.button("Encoder"):
                if encoding == "Label Encoding":
                    from sklearn.preprocessing import LabelEncoder
                    for col in cat_cols:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                else:
                    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
                
                st.session_state.df = df
                st.success("Encodage effectué")
        
        # Normalisation
        st.subheader("4. Normalisation")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            norm_method = st.selectbox("Méthode", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
            
            if st.button("Normaliser les données"):
                df = normalize_features(df, method=norm_method)
                st.session_state.df = df
                st.success("Normalisation effectuée")
        
        # Aperçu
        st.subheader("Aperçu")
        st.dataframe(df.head(10))
        
        if st.button("Réinitialiser"):
            st.session_state.df = st.session_state.df_original.copy()
            st.success("Données réinitialisées")
# ==================== MODÉLISATION ====================
elif page == "Modélisation":
    if st.session_state.df is None:
        st.warning("Veuillez charger un dataset")
    else:
        st.header("Modélisation Machine Learning")
        df = st.session_state.df.copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            target = st.selectbox("Variable cible", df.columns)
            problem_type = st.radio("Type", ["Classification", "Régression", "Clustering"])
        
        with col2:
            available_features = [col for col in df.columns if col != target]
            features = st.multiselect("Features", available_features, default=available_features)
            test_size = st.slider("Taille test (%)", 10, 50, 20) / 100
        
        if not features:
            st.warning("Sélectionnez au moins une feature")
        else:
            # Sélection du modèle
            available_models = get_available_models()
            
            if problem_type == "Régression":
                models = available_models['Régression']
            elif problem_type == "Classification":
                models = available_models['Classification']
            else:
                models = available_models['Clustering']
            
            selected_model = st.selectbox("Algorithme", models)
            
            # Paramètres
            with st.expander("Paramètres avancés"):
                params = {}
                
                if "Random Forest" in selected_model:
                    params['n_estimators'] = st.slider("Nombre d'arbres", 50, 500, 100)
                    params['max_depth'] = st.slider("Profondeur max", 3, 30, 10)
                elif "Arbre de Décision" in selected_model:
                    max_depth = st.slider("Profondeur maximale de l'arbre", 1, 30, 5)
                    params['max_depth'] = None if max_depth == 30 else max_depth
                    params['min_samples_split'] = st.slider("Échantillons min pour diviser", 2, 20, 2)
                    params['min_samples_leaf'] = st.slider("Échantillons min par feuille", 1, 10, 1)
                elif "KNN" in selected_model:
                    params['n_neighbors'] = st.slider("Nombre de voisins", 1, 20, 5)
                elif "K-Means" in selected_model:
                    params['n_clusters'] = st.slider("Nombre de clusters", 2, 10, 3)
                elif "SVM" in selected_model:
                    params['C'] = st.slider("C", 0.1, 10.0, 1.0)
                    params['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
                elif "Réseau de Neurones" in selected_model:
                    layer1 = st.slider("Neurones couche 1", 10, 200, 100)
                    layer2 = st.slider("Neurones couche 2", 10, 200, 50)
                    params['hidden_layer_sizes'] = (layer1, layer2)
                
                scale = st.checkbox("Normaliser", value=True)
            
            # Entraînement
            col1, col2 = st.columns(2)
            
            with col1:
                train_btn = st.button("Entraîner")
            with col2:
                cv_btn = st.button("Validation croisée")
            
            if train_btn:
                try:
                    # Préparation
                    X = df[features]
                    
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                    
                    if problem_type != "Clustering":
                        y = df[target]
                        if y.dtype == 'object':
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                    
                    # Création du modèle
                    manager = create_model(selected_model, params)
                    
                    if problem_type != "Clustering":
                        manager.prepare_data(X, y, test_size=test_size, scale=scale)
                        manager.train()
                        manager.predict()
                        
                        # Check for high cardinality in classification
                        if problem_type == "Classification" and len(np.unique(y)) > 50:
                            st.warning(f"⚠️ Attention: La variable cible contient {len(np.unique(y))} classes uniques. Êtes-vous sûr qu'il s'agit d'un problème de classification? (Vérifiez si vous n'avez pas sélectionné une colonne d'ID)")

                        st.session_state.model_manager = manager
                        st.session_state.trained = True
                        st.session_state.feature_names = features
                        
                        st.success("Modèle entraîné avec succès")
                        
                        # Métriques rapides
                        if problem_type == "Régression":
                            metrics = manager.evaluate_regression()
                            col1, col2, col3 = st.columns(3)
                            col1.metric("R²", f"{metrics['R2']:.4f}")
                            col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
                            col3.metric("MAE", f"{metrics['MAE']:.4f}")
                        else:
                            metrics = manager.evaluate_classification()
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                            col2.metric("Precision", f"{metrics['Precision']:.4f}")
                            col3.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
                    
                    else:
                        manager.model.fit(X)
                        predictions = manager.model.predict(X)
                        manager.predictions = predictions
                        manager.X_test = X.values # Store for evaluation
                        
                        st.session_state.model_manager = manager
                        st.session_state.trained = True
                        
                        metrics = manager.evaluate_clustering(X, predictions)
                        st.success("Clustering effectué")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Inertia", f"{metrics['Inertia']:.4f}")
                        col2.metric("Silhouette", f"{metrics['Silhouette Score']:.4f}")
                
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
            
            if cv_btn and problem_type != "Clustering":
                try:
                    X = df[features]
                    y = df[target]
                    
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                    
                    if y.dtype == 'object':
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                    
                    manager = create_model(selected_model, params)
                    results = manager.cross_validate(X, y, cv=5)
                    
                    st.success("Validation croisée terminée")
                    col1, col2 = st.columns(2)
                    col1.metric("Score moyen", f"{results['mean']:.4f}")
                    col2.metric("Écart-type", f"{results['std']:.4f}")
                
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")

# ==================== ÉVALUATION ====================
elif page == "Évaluation":
    if not st.session_state.trained:
        st.warning("Veuillez entraîner un modèle")
    else:
        st.header("Évaluation du modèle")
        manager = st.session_state.model_manager
        
        if manager.model_type == "regression":
            metrics = manager.evaluate_regression()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{metrics['R2']:.4f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
            col3.metric("MAE", f"{metrics['MAE']:.4f}")
            col4.metric("MSE", f"{metrics['MSE']:.4f}")
            
            # Graphiques
            tab1, tab2 = st.tabs(["Prédictions vs Réel", "Résidus"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(manager.y_test, manager.predictions, alpha=0.6)
                ax.plot([manager.y_test.min(), manager.y_test.max()],
                       [manager.y_test.min(), manager.y_test.max()], 'r--', lw=2)
                ax.set_xlabel("Valeurs réelles")
                ax.set_ylabel("Prédictions")
                st.pyplot(fig)
                plt.close()
            
            with tab2:
                residuals = manager.y_test - manager.predictions
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(manager.predictions, residuals, alpha=0.6)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_xlabel("Prédictions")
                ax.set_ylabel("Résidus")
                st.pyplot(fig)
                plt.close()
        
        elif manager.model_type == "classification":
            metrics = manager.evaluate_classification()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['Precision']:.4f}")
            col3.metric("Recall", f"{metrics['Recall']:.4f}")
            col4.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
            
            # Matrice de confusion
            st.subheader("Matrice de confusion")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Prédictions")
            ax.set_ylabel("Réel")
            st.pyplot(fig)
            plt.close()
            
            st.text("Rapport de classification:")
            st.text(metrics['Classification Report'])
        
        else:
            metrics = manager.evaluate_clustering(manager.X_test, manager.predictions)
            
            col1, col2 = st.columns(2)
            col1.metric("Inertia", f"{metrics['Inertia']:.4f}")
            col2.metric("Silhouette Score", f"{metrics['Silhouette Score']:.4f}")
            
            # Visualisation
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(manager.X_test)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                               c=manager.predictions, cmap='viridis', alpha=0.6)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plt.colorbar(scatter)
            st.pyplot(fig)
            plt.close()
        
        # Importance des features
        importance = manager.get_feature_importance()
        if importance is not None:
            st.subheader("Importance des variables")
            fig, ax = plt.subplots(figsize=(8, 6))
            if 'feature_names' in st.session_state:
                features = st.session_state.feature_names
            else:
                 features = st.session_state.df.columns.drop([st.session_state.df.columns[-1]])[:len(importance)]
            
            # Ensure lengths match before plotting
            if len(features) == len(importance):
                 ax.barh(features, importance)
            else:
                 st.warning(f"Impossible d'afficher l'importance des variables: mismatch ({len(features)} features vs {len(importance)} values)")
            ax.set_xlabel("Importance")
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()

        # Visualisation de l'Arbre de Décision
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
        if isinstance(manager.model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            st.subheader("Visualisation de l'Arbre")
            with st.expander("Voir l'arbre de décision"):
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(manager.model, 
                         feature_names=st.session_state.get('feature_names', None),
                         filled=True, 
                         rounded=True,
                         fontsize=10,
                         ax=ax)
                st.pyplot(fig)
                plt.close()

        # Visualisation SVM - Frontière de décision (2D seulement)
        from sklearn.svm import SVC, SVR
        if isinstance(manager.model, SVC) and manager.model_type == "classification":
            if manager.X_test.shape[1] == 2:
                st.subheader("Frontière de décision SVM")
                with st.expander("Voir la frontière de décision (2D)"):
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Créer une grille
                    h = 0.02
                    x_min, x_max = manager.X_test[:, 0].min() - 1, manager.X_test[:, 0].max() + 1
                    y_min, y_max = manager.X_test[:, 1].min() - 1, manager.X_test[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                    
                    # Prédire sur la grille
                    Z = manager.model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    
                    # Tracer
                    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
                    scatter = ax.scatter(manager.X_test[:, 0], manager.X_test[:, 1], 
                                       c=manager.y_test, cmap='viridis', edgecolors='black', s=50)
                    ax.set_xlabel("Feature 1")
                    ax.set_ylabel("Feature 2")
                    ax.set_title("Frontière de décision SVM")
                    plt.colorbar(scatter, ax=ax)
                    st.pyplot(fig)
                    plt.close()
            else:
                st.info("💡 La visualisation de la frontière de décision SVM nécessite exactement 2 features. Utilisez PCA ou sélectionnez 2 features pour voir cette visualisation.")

        # Visualisation KNN
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        if isinstance(manager.model, (KNeighborsClassifier, KNeighborsRegressor)):
            if manager.X_test.shape[1] == 2 and manager.model_type == "classification":
                st.subheader("Visualisation K-NN")
                with st.expander("Voir les régions de décision"):
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    h = 0.02
                    x_min, x_max = manager.X_test[:, 0].min() - 1, manager.X_test[:, 0].max() + 1
                    y_min, y_max = manager.X_test[:, 1].min() - 1, manager.X_test[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                    
                    Z = manager.model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    
                    ax.contourf(xx, yy, Z, alpha=0.3, cmap='Spectral')
                    scatter = ax.scatter(manager.X_test[:, 0], manager.X_test[:, 1], 
                                       c=manager.y_test, cmap='Spectral', edgecolors='black', s=50)
                    ax.set_xlabel("Feature 1")
                    ax.set_ylabel("Feature 2")
                    ax.set_title(f"Régions de décision K-NN (k={manager.model.n_neighbors})")
                    plt.colorbar(scatter, ax=ax)
                    st.pyplot(fig)
                    plt.close()

        # Visualisation Régression Logistique / Linéaire - Coefficients
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
        if isinstance(manager.model, (LogisticRegression, LinearRegression, Ridge, Lasso)):
            if hasattr(manager.model, 'coef_'):
                st.subheader("Coefficients du modèle")
                with st.expander("Voir les coefficients"):
                    coef = manager.model.coef_
                    if coef.ndim > 1:
                        coef = np.mean(np.abs(coef), axis=0)
                    else:
                        coef = np.abs(coef)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    features = st.session_state.get('feature_names', [f"Feature {i}" for i in range(len(coef))])
                    
                    # Ensure lengths match
                    if len(features) != len(coef):
                        features = [f"Feature {i}" for i in range(len(coef))]
                    
                    ax.barh(features, coef, color='steelblue')
                    ax.set_xlabel("Valeur absolue du coefficient")
                    ax.set_title("Impact des variables sur la prédiction")
                    ax.invert_yaxis()
                    st.pyplot(fig)
                    plt.close()
                    
                    if hasattr(manager.model, 'intercept_'):
                        st.write(f"**Intercept (biais):** {manager.model.intercept_}")

        # Visualisation Réseau de Neurones - Architecture
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        if isinstance(manager.model, (MLPClassifier, MLPRegressor)):
            st.subheader("Architecture du Réseau de Neurones")
            with st.expander("Voir l'architecture"):
                layers = [manager.X_test.shape[1]] + list(manager.model.hidden_layer_sizes)
                if manager.model_type == "classification":
                    layers.append(len(np.unique(manager.y_test)))
                else:
                    layers.append(1)
                
                st.write("**Structure du réseau:**")
                layer_names = ["Couche d'entrée"] + [f"Couche cachée {i+1}" for i in range(len(manager.model.hidden_layer_sizes))] + ["Couche de sortie"]
                for i, (name, size) in enumerate(zip(layer_names, layers)):
                    st.write(f"- {name}: {size} neurones")
                
                # Graphique simple de l'architecture
                fig, ax = plt.subplots(figsize=(12, 6))
                max_neurons = max(layers)
                
                for i, layer_size in enumerate(layers):
                    x = i * 2
                    y_positions = np.linspace(0, max_neurons, layer_size + 2)[1:-1]
                    ax.scatter([x] * layer_size, y_positions, s=200, c='steelblue', zorder=3)
                    
                    # Connexions vers la couche suivante
                    if i < len(layers) - 1:
                        next_y_positions = np.linspace(0, max_neurons, layers[i+1] + 2)[1:-1]
                        for y1 in y_positions:
                            for y2 in next_y_positions:
                                ax.plot([x, x+2], [y1, y2], 'gray', alpha=0.2, linewidth=0.5)
                
                ax.set_xlim(-1, len(layers) * 2)
                ax.set_ylim(-1, max_neurons + 1)
                ax.set_xticks(range(0, len(layers) * 2, 2))
                ax.set_xticklabels(layer_names, rotation=45, ha='right')
                ax.set_yticks([])
                ax.set_title("Architecture du Réseau de Neurones")
                ax.grid(False)
                st.pyplot(fig)
                plt.close()
                
                if hasattr(manager.model, 'loss_curve_'):
                    st.write("**Courbe d'apprentissage:**")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(manager.model.loss_curve_, linewidth=2)
                    ax.set_xlabel("Itérations")
                    ax.set_ylabel("Loss")
                    ax.set_title("Évolution de la fonction de perte")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()

# ==================== EXPORT ====================
elif page == "Export":

    st.header("Export des résultats du modèle")

    if not st.session_state.trained:
        st.warning("Veuillez entraîner un modèle avant d’exporter les résultats.")
        st.stop()

    manager = st.session_state.model_manager

    st.write(
        "Cette section permet d’exporter les résultats produits par le modèle "
        "entraîné : métriques de performance, prédictions, rapports complets "
        "et sauvegarde du modèle."
    )

    st.divider()

    # ==================================================
    # 1. EXPORT DES MÉTRIQUES
    # ==================================================
    st.subheader("1. Métriques de performance")

    st.write(
        "Les métriques dépendent du type de problème traité "
        "(classification, régression ou clustering)."
    )

    col1, col2 = st.columns(2)

    # ---------- CSV ----------
    with col1:
        if st.button("Télécharger les métriques (CSV)"):
            try:
                if manager.model_type == "regression":
                    metrics = manager.evaluate_regression()
                elif manager.model_type == "classification":
                    metrics = manager.evaluate_classification()
                else:
                    metrics = manager.evaluate_clustering(
                        manager.X_test, manager.predictions
                    )

                metrics_clean = {
                    k: v for k, v in metrics.items()
                    if k not in ["Confusion Matrix", "Classification Report"]
                }

                metrics_df = pd.DataFrame([metrics_clean])
                csv = metrics_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Sauvegarder le fichier CSV",
                    csv,
                    "metriques_modele.csv",
                    "text/csv"
                )

                st.success("Métriques exportées avec succès.")
            except Exception as e:
                st.error(f"Erreur lors de l’export : {str(e)}")

    # ---------- JSON ----------
    with col2:
        if st.button("Télécharger les métriques (JSON)"):
            try:
                import json

                if manager.model_type == "regression":
                    metrics = manager.evaluate_regression()
                elif manager.model_type == "classification":
                    metrics = manager.evaluate_classification()
                else:
                    metrics = manager.evaluate_clustering(
                        manager.X_test, manager.predictions
                    )

                metrics_clean = {}
                for k, v in metrics.items():
                    if k == "Classification Report":
                        metrics_clean[k] = str(v)
                    elif k != "Confusion Matrix" and isinstance(v, (int, float, str)):
                        metrics_clean[k] = v

                json_str = json.dumps(metrics_clean, indent=2).encode("utf-8")

                st.download_button(
                    "Sauvegarder le fichier JSON",
                    json_str,
                    "metriques_modele.json",
                    "application/json"
                )

                st.success("Métriques exportées avec succès.")
            except Exception as e:
                st.error(f"Erreur lors de l’export : {str(e)}")

    st.divider()

    # ==================================================
    # 2. EXPORT DES PRÉDICTIONS
    # ==================================================
    st.subheader("2. Résultats de prédiction")

    st.write(
        "Ce fichier contient les prédictions générées par le modèle. "
        "Pour les problèmes de régression, les erreurs sont également fournies."
    )

    if st.button("Télécharger les prédictions (CSV)"):
        try:
            if manager.model_type != "clustering":
                results_df = pd.DataFrame({
                    "Valeur reelle": manager.y_test,
                    "Prediction": manager.predictions
                })

                if manager.model_type == "regression":
                    results_df["Erreur"] = (
                        results_df["Valeur reelle"] - results_df["Prediction"]
                    )
                    results_df["Erreur absolue"] = abs(results_df["Erreur"])
            else:
                results_df = pd.DataFrame({
                    "Cluster": manager.predictions
                })

            csv = results_df.to_csv(
                index=False,   # enlève la colonne index
                sep=";",       # séparateur Excel FR
                encoding="utf-8"
            ).encode("utf-8")


            st.download_button(
                "Sauvegarder les prédictions",
                csv,
                "predictions.csv",
                "text/csv"
            )

            st.success("Prédictions exportées avec succès.")
        except Exception as e:
            st.error(f"Erreur lors de l’export : {str(e)}")

    st.divider()

   

    # ==================================================
    # 3. SAUVEGARDE DU MODÈLE
    # ==================================================
    st.subheader("3. Sauvegarde du modèle entraîné")

    st.write(
        "Le modèle entraîné peut être sauvegardé afin d’être réutilisé ultérieurement "
        "sans relancer l’entraînement."
    )

    model_name = st.text_input("Nom du fichier du modèle", "modele_ml")

    if st.button("Sauvegarder le modèle"):
        try:
            import pickle

            filename = f"{model_name}.pkl"

            with open(filename, "wb") as f:
                pickle.dump(manager, f)

            with open(filename, "rb") as f:
                st.download_button(
                    "Télécharger le modèle",
                    f,
                    filename,
                    "application/octet-stream"
                )

            st.success(f"Modèle sauvegardé sous le nom : {filename}")
        except Exception as e:
            st.error(f"Erreur lors de la sauvegarde : {str(e)}")


    # ==================== DOCUMENTATION ET TUTORIELS ====================
elif page == "Documentation":
    st.header("📚 Documentation et Tutoriels")
    
    st.markdown("""
    Cette section contient toute la documentation technique et les tutoriels pour vous aider 
    à utiliser l'application et comprendre les algorithmes de Machine Learning.
    """)
    
    # Tabs pour séparer Documentation et Tutoriels
    tab1, tab2 = st.tabs(["📖 Tutoriels", "🔧 Documentation Technique"])
    
    with tab1:
        st.markdown("### Tutoriels des Algorithmes")
        st.markdown("""
        Apprenez à utiliser chaque algorithme avec des guides pratiques étape par étape.
        """)
        
        try:
            with open('TUTORIELS.md', 'r', encoding='utf-8') as f:
                tutoriels_content = f.read()
            st.markdown(tutoriels_content)
        except FileNotFoundError:
            st.error("❌ Fichier TUTORIELS.md introuvable")
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des tutoriels: {str(e)}")
    
    with tab2:
        st.markdown("### Documentation Technique")
        st.markdown("""
        Documentation complète pour les développeurs et contributeurs.
        """)
        
        try:
            with open('DOCUMENTATION.md', 'r', encoding='utf-8') as f:
                doc_content = f.read()
            st.markdown(doc_content)
        except FileNotFoundError:
            st.error("❌ Fichier DOCUMENTATION.md introuvable")
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement de la documentation: {str(e)}")
    
    st.markdown("---")
    
    # Section d'aide rapide
    with st.expander("💡 Aide Rapide"):
        st.markdown("""
        ### Guide de démarrage rapide
        
        1. **Chargement des données** : Importez votre dataset (CSV ou Excel)
        2. **Exploration** : Visualisez les statistiques et distributions
        3. **Prétraitement** : Nettoyez et préparez vos données
        4. **Modélisation** : Choisissez et entraînez un algorithme
        5. **Évaluation** : Analysez les performances du modèle
        6. **Export** : Téléchargez vos résultats
        
        ### Choix d'algorithme rapide
        
        **Pour la Régression** (prédire un nombre) :
        - Débutant : Régression Linéaire
        - Avancé : Random Forest Regressor
        
        **Pour la Classification** (prédire une catégorie) :
        - Débutant : Régression Logistique
        - Avancé : Random Forest Classifier
        
        **Pour le Clustering** (grouper des données similaires) :
        - K-Means (commencez avec 3 clusters)
        
        ### Normalisation : Quand l'utiliser ?
        
        ✅ **OBLIGATOIRE pour** : SVM, KNN, Réseaux de Neurones
        
        ⚠️ **Recommandé pour** : Régression Linéaire, Régression Logistique
        
        ❌ **Optionnel pour** : Arbres de Décision, Random Forest, Naïve Bayes
        """)
    
    # Liens utiles
    with st.expander("🔗 Ressources Externes"):
        st.markdown("""
        ### Ressources d'apprentissage
        
        - [Scikit-learn Documentation](https://scikit-learn.org/stable/)
        - [Machine Learning Glossary](https://ml-cheatsheet.readthedocs.io/)
        - [Streamlit Documentation](https://docs.streamlit.io/)
        - [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
        
        ### Cours en ligne
        
        - [Machine Learning par Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning)
        - [Fast.ai - Practical Deep Learning](https://www.fast.ai/)
        - [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
        """)