import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression

def calculate_trend_residuals(df, degree, is_3d, method, target_cols, x_col="X", y_col="Y", z_col="Z"):
    """
    Calcule la tendance et les résidus pour une ou plusieurs variables via régression polynomiale.
    
    Cette fonction généralise la logique de reconstruction de surface pour extraire
    la tendance (Drift) et les résidus aux points de données existants.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        degree (int): Le degré du polynôme (ex: 1, 2, 3).
        is_3d (bool): Si True, utilise [X, Y, Z] comme variables explicatives. 
                      Si False, utilise [X, Y].
        method (str): 'simple' (LinearRegression) ou 'ridge' (Ridge Regression).
        target_cols (str ou list): Le nom de la colonne (ou liste de colonnes) variable(s) d'intérêt.
        x_col (str): Nom de la colonne X (par défaut "X").
        y_col (str): Nom de la colonne Y (par défaut "Y").
        z_col (str): Nom de la colonne Z (par défaut "Z"), utilisé seulement si is_3d=True.

    Returns:
        dict: Un dictionnaire contenant :
            - 'trend': (np.array) Les valeurs de la tendance calculée.
            - 'residuals': (np.array) Les résidus (Valeur Réelle - Tendance).
            - 'model': (Pipeline) Le modèle scikit-learn entraîné.
            - 'feature_names': (list) Les noms des features polynomiales générées.
            - 'coefficients': (np.array) Les coefficients du modèle.
            - 'intercept': (float ou array) L'ordonnée à l'origine.
            - 'valid_index': (pd.Index) Les index des lignes utilisées (sans NaN).
    """
    
    # 1. Gestion des entrées (str ou list pour target_cols)
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    
    # 2. Définition des variables explicatives (Features)
    if is_3d:
        features_cols = [x_col, y_col, z_col]
    else:
        features_cols = [x_col, y_col]
        
    # 3. Vérification des colonnes
    required_cols = features_cols + target_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {missing}")

    # 4. Nettoyage des données (Suppression des NaNs)
    # On travaille sur une copie pour ne pas modifier l'original et on filtre les NaNs
    work_df = df[required_cols].dropna()
    
    if work_df.empty:
        raise ValueError("Aucune donnée valide après suppression des valeurs manquantes (NaNs).")

    X = work_df[features_cols].values
    y = work_df[target_cols].values
    
    # Si une seule cible, on aplatit y pour avoir un array 1D (standard sklearn)
    if len(target_cols) == 1:
        y = y.ravel()

    # 5. Configuration du Pipeline
    # Standardisation -> Polynôme -> Régression
    scaler = StandardScaler()
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    method_key = method.lower().strip()
    if method_key == 'ridge':
        # Alpha = 1.0 par défaut comme dans le script original
        regressor = Ridge(alpha=1.0)
    elif method_key in ['simple', 'linear', 'ols']:
        regressor = LinearRegression()
    else:
        raise ValueError(f"Méthode '{method}' non supportée. Utilisez 'simple' ou 'ridge'.")

    model = Pipeline([
        ("scaler", scaler),
        ("poly", poly),
        ("regressor", regressor)
    ])

    # 6. Entraînement du modèle
    model.fit(X, y)

    # 7. Calcul de la tendance et des résidus
    trend = model.predict(X)
    residuals = y - trend

    # 8. Extraction des métadonnées du modèle
    poly_step = model.named_steps['poly']
    reg_step = model.named_steps['regressor']
    
    feature_names = poly_step.get_feature_names_out(features_cols)
    
    return {
        "trend": trend,
        "residuals": residuals,
        "model": model,
        "feature_names": feature_names,
        "coefficients": reg_step.coef_,
        "intercept": reg_step.intercept_,
        "valid_index": work_df.index
    }
