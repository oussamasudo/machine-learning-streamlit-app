import pandas as pd
import streamlit as st

def load_data(uploaded_file):
    """
    Charge un fichier CSV ou Excel et retourne un DataFrame pandas
    """
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Format de fichier non supporté")
                return None
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")
            return None
    return None
