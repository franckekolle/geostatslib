# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 08:27:17 2026

@author: ekolleessoh
"""
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import gstlearn as gl
#import numpy as np
from calculate_trend_residuals import calculate_trend_residuals



class geostat_library():
    
    def cova(model_type, h):

       # =====================================================================================
       # =====================================================================================
       # Calcule la corrélation spatiale (0 à 1) pour différents modèles.
       # h : array-like, distances transformées (normalisées)
       # model_type : int, 1=Sphérique, 2=Exponentiel, 3=Gaussien, 4=Effet de Pépite, 5=Cubique
       # % Auteur: Franck Ekolle
       # % Dernière mise à jour: 17/02/2026
       # ===================================================================================== 
        h = np.asanyarray(h)
        c = np.zeros_like(h, dtype=float)
        
        if model_type == 1: # Modèle Sphérique
            idx = h <= 1
            c[idx] = 1 - (1.5 * h[idx] - 0.5 * h[idx]**3)
            # c[~idx] est déjà à 0
            
        elif model_type == 2: # Modèle Exponentiel
            c = np.exp(-3 * h)
            
        elif model_type == 3: # Modèle Gaussien
            c = np.exp(-3 * h**2)
            
        elif model_type == 4: # Effet de pépite (Nugget effect)
            # On utilise une tolérance machine pour h == 0
            c = (h < np.finfo(float).eps).astype(float)
            
        elif model_type == 5: # Modèle Cubique
            idx = h <= 1
            hh = h[idx]
            c[idx] = 1 - (7*hh**2 - 8.75*hh**3 + 3.5*hh**5 - 0.75*hh**7)
            # c[~idx] est déjà à 0
            
        else:
            raise ValueError("Modèle inconnu. Types supportés : 1(Sph), 2(Exp), 3(Gau), 4(Nug), 5(Cub)")
            
        return c
    
    
    def setrot(model, it):
        """
        ------------------------------------------------------------
        Set up the matrix to transform the Cartesian coordinates to
        coordinates that account for angles and anisotropy
        ----------------------------
        INPUT:
            model : nested variogram model (nst x 7)
            it    : index of the nested structure (0-based in Python)

        OUTPUT:
            rotred_matrix : rotation + reduction matrix (3x3)

        Note:
            Rotation follows GSLIB conventions
        """

        deg2rad = np.pi / 180.0

        # MATLAB: ranges = model(it,2:4);
        ranges = model[it, 1:4]

        # MATLAB: angles = model(it,5:7);
        angles = model[it, 4:7]

        # ----------------------------------------------------------
        # Reduction matrix (anisotropy)
        # ----------------------------------------------------------
        redmat = np.diag(1.0 / (np.finfo(float).eps + ranges))

        # ----------------------------------------------------------
        # Rotation angles (GSLIB convention)
        # ----------------------------------------------------------
        a = (90.0 - angles[0]) * deg2rad
        b = -angles[1] * deg2rad
        c = angles[2] * deg2rad

        cosa = np.cos(a)
        sina = np.sin(a)
        cosb = np.cos(b)
        sinb = np.sin(b)
        cosc = np.cos(c)
        sinc = np.sin(c)

        # ----------------------------------------------------------
        # Rotation matrix
        # ----------------------------------------------------------
        rotmat = np.zeros((3, 3))

        rotmat[0, 0] = cosb * cosa
        rotmat[0, 1] = cosb * sina
        rotmat[0, 2] = -sinb

        rotmat[1, 0] = -cosc * sina + sinc * sinb * cosa
        rotmat[1, 1] = cosc * cosa + sinc * sinb * sina
        rotmat[1, 2] = sinc * cosb

        rotmat[2, 0] = sinc * sina + cosc * sinb * cosa
        rotmat[2, 1] = -sinc * cosa + cosc * sinb * sina
        rotmat[2, 2] = cosc * cosb

        # ----------------------------------------------------------
        # Final matrix (MATLAB: (redmat*rotmat)')
        # ----------------------------------------------------------
        rotred_matrix = (redmat @ rotmat).T

        return rotred_matrix
    
    def polynomial_trend_residuals(
        datacoord,
        datavalues,
        order=1,
        standardize=True,
        make_plots=False,
        figsize=(6, 5),
        return_model=False,
        var_names=None,
        output_csv=None
    ):
        """
        Tendance polynomiale spatiale (ordre 1 ou 2) + résidus
        Compatible mono- et multi-variables.

        datacoord : (n, 3)
        datavalues : (n,) ou (n, p)
        """
        # -----------------------------
        # Vérifications
        # -----------------------------
        datacoord = np.asarray(datacoord, float)
        datavalues = np.asarray(datavalues, float)
        # Forcer datavalues en 2D : (n, p) pour une gestion cohérente
        if datavalues.ndim == 1:
            datavalues = datavalues[:, None]

        # Gestion des NaN : doit se faire sur un tableau 2D en vérifiant chaque ligne
        mask = np.all(np.isfinite(datavalues), axis=1)
        
        if not np.all(mask):
            n_removed = np.sum(~mask)
            print(f"[INFO] {n_removed} échantillons retirés (NaN dans datavalues)")

        datacoord = datacoord[mask, :]
        datavalues = datavalues[mask, :]

        # Vérification flexible pour 2D ou 3D
        if datacoord.ndim != 2 or datacoord.shape[1] not in [2, 3]:
            raise ValueError("datacoord doit être de forme (n, 2) ou (n, 3)")

        if datavalues.shape[0] != datacoord.shape[0]:
            raise ValueError("datavalues doit avoir le même nombre de lignes que datacoord")

        n, p = datavalues.shape

        if var_names is None:
            var_names = [f"var_{i+1}" for i in range(p)]
        if len(var_names) != p:
            raise ValueError("var_names doit avoir la même longueur que le nombre de variables")

        if order not in (1, 2, 3, 4):
            raise ValueError("order doit être entre 1 et 4")
        
        ndim = datacoord.shape[1]
        if ndim == 2:
            coord_names = ['X', 'Y']
        else:
            coord_names = ['X', 'Y', 'Z']
        X = datacoord.copy()
        Y = datavalues

        # -----------------------------
        # Standardisation (coordonnées uniquement)
        # -----------------------------
        scaler = None
        if standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # -----------------------------
        # Base polynomiale
        # -----------------------------
        poly = PolynomialFeatures(degree=order, include_bias=True)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(coord_names)

        # -----------------------------
        # Régression multi-output
        # -----------------------------
        model = LinearRegression(fit_intercept=False)
        model.fit(X_poly, Y)

        trend = model.predict(X_poly)        # (n, p)
        residuals = Y - trend                # (n, p)

        # -----------------------------
        # Coefficients explicites
        # -----------------------------
        # model.coef_ : (p, n_features)
        coefficients = {}
        for j, var in enumerate(var_names):
            coefficients[var] = {
                name: float(coef)
                for name, coef in zip(feature_names, model.coef_[j])
            }

        # -----------------------------
        # Diagnostics
        # -----------------------------
        diagnostics = {}
        for j, var in enumerate(var_names):
            diagnostics[var] = {
                "mean_residuals": float(np.mean(residuals[:, j])),
                "var_residuals": float(np.var(residuals[:, j])),
                "r2_trend": float(r2_score(Y[:, j], trend[:, j]))
            }

        # -----------------------------
        # Graphiques (1 variable à la fois)
        # -----------------------------
        if make_plots:
            for j, var in enumerate(var_names):
                plt.figure(figsize=figsize)
                plt.scatter(trend[:, j], Y[:, j], alpha=0.6)
                plt.plot([trend[:, j].min(), trend[:, j].max()],
                         [trend[:, j].min(), trend[:, j].max()], '--k')
                plt.xlabel("Tendance")
                plt.ylabel("Observé")
                plt.title(f"Observé vs Tendance – {var}")
                plt.grid(True); plt.show()

                plt.figure(figsize=figsize)
                plt.hist(residuals[:, j], bins=30, edgecolor='black')
                plt.xlabel("Résidus")
                plt.ylabel("Fréquence")
                plt.title(f"Histogramme des résidus – {var}")
                plt.grid(True); plt.show()

                # Le graphique vs Z n'est pertinent qu'en 3D
                if ndim == 3:
                    plt.figure(figsize=figsize)
                    plt.scatter(datacoord[:, 2], residuals[:, j], alpha=0.6)
                    plt.axhline(0, color='black', linestyle='--')
                    plt.xlabel("Z")
                    plt.ylabel("Résidus")
                    plt.title(f"Résidus vs Z – {var}")
                    plt.grid(True); plt.show()


        # -----------------------------
        # DataFrame & Export
        # -----------------------------
        df_results = pd.DataFrame(datacoord, columns=coord_names)
        for j, var in enumerate(var_names):
            df_results[var] = Y[:, j]
            df_results[f"Trend_{var}"] = trend[:, j]
            df_results[f"Residual_{var}"] = residuals[:, j]

        if output_csv:
            df_results.to_csv(output_csv, index=False, sep = ";")
            print(f"[INFO] Résultats exportés vers : {output_csv}")
            
        # -----------------------------
        # Sorties
        # -----------------------------
        results = {
            "trend": trend,                 # (n, p)
            "residuals": residuals,         # (n, p)
            "coefficients": coefficients,   # dict[var][term]
            "feature_names": feature_names,
            "diagnostics": diagnostics
        }

    #   out_csv = r"D:\Documents\ekolleessoh\Travail\Projet_ANDRA\Lissage\05_LS0_Subsample_2000_residu.csv"

        if return_model:
            results.update({
                "model": model,
                "poly": poly,
                "scaler": scaler
            })

        return results, df_results
    
    
    def calculate_variogram(
        self,
        data_df: pd.DataFrame,
        is_3d: bool,
        coord_cols: list,
        variable_cols: list,
        directions: list,
        calcul: str,
        resid: bool,
        degree: int, 
        method: str,
        x_col: float,
        y_col: float,
        z_col: float
    ) -> tuple[gl.Vario, gl.Db]:
        """
        Calcule un variogramme expérimental à partir d'un DataFrame et de paramètres.

        Args:
            data_df (`pd.DataFrame`): Le DataFrame contenant les données.
            is_3d (bool): Vrai si l'espace est en 3D, Faux pour 2D.
            coord_cols (list): Liste des noms de colonnes pour les coordonnées (['X', 'Y'] ou ['X', 'Y', 'Z']).
            variable_cols (list): Liste des noms de colonnes pour la ou les variables Z.
            directions (list): Une liste de dictionnaires, où chaque dictionnaire
                               définit une direction de calcul pour le variogramme.
                               Ex: [{'nlag': 10, 'dlag': 100}, {'nlag': 15, 'dlag': 5, 'codir': [0,0,1]}]
            method (str): 'simple' (LinearRegression) ou 'ridge' (Ridge Regression).
            resid (bool): Calculer le résidu ou pas
            degree (int): Le dégré du polynôme pour le calcul du residu
            calcul (str): La fonction calculée (variogram, covariance, etc...)

        Returns:
            tuple[gl.Vario, gl.Db]: Un tuple contenant l'objet Vario calculé et l'objet Db utilisé.
        """
        # 1. Définir l'espace (2D ou 3D) ---
        ndim = 3 if is_3d else 2
        gl.defineDefaultSpace(gl.ESpaceType.RN, ndim)
        if len(coord_cols) != ndim:
            raise ValueError(f"Pour un espace {ndim}D, {ndim} colonnes de coordonnées sont attendues, mais {len(coord_cols)} ont été fournies.")

        results = None
        
        if resid:
            results = calculate_trend_residuals(
                df=data_df,
                degree=degree,
                is_3d=is_3d,
                method=method,
                target_cols=variable_cols,
                x_col=x_col,
                y_col=y_col,
                z_col=z_col)
            #coef = results['coefficients']
            #intercept =  results['intercept']
            resi =  results['residuals']
            trend =  results['trend']
            nbre_cols = resi.shape[1]
            # 2. Générer la liste de noms synthétiques
            # On nomme la première 'var_princ' et les autres 'var_aux1', 'var_aux2', etc.
            noms_colonnes = [
                "Res_princ" if i == 0 else f"Res_var_aux{i}" 
                for i in range(nbre_cols)]
            noms_cls = [
                "Trend_princ" if i == 0 else f"trend_var_aux{i}" 
                for i in range(nbre_cols)]
            
            if not noms_colonnes:
                raise ValueError("Au moins une colonne de variable doit être fournie.")
            
            # 3. Créer le DataFrame
            coords = [x_col, y_col, z_col] if (z_col is not None and is_3d) else [x_col, y_col]
            residu = pd.DataFrame(resi, columns=noms_colonnes)
            tendance = pd.DataFrame(trend, columns=noms_cls)
            df_final = pd.concat([coords, residu, tendance], axis=1)
            
            db = gl.Db_fromPandas(df_final)
            if len(noms_colonnes) == 1:
                db.setLocator(noms_colonnes[0], gl.ELoc.Z)
            else:
                db.setLocators(noms_colonnes, gl.ELoc.Z)

        else:
            db = gl.Db_fromPandas(data_df)
            db.setLocators(coord_cols, gl.ELoc.X)
            if not variable_cols:
                raise ValueError("Au moins une colonne de variable doit être fournie.")
            if len(variable_cols) == 1:
                db.setLocator(variable_cols[0], gl.ELoc.Z)
            else:
                db.setLocators(variable_cols, gl.ELoc.Z)
                
        # 4. Paramétrer et calculer le variogramme ---
        varioparam = gl.VarioParam()
        for dir_params in directions:
            # On passe directement le dictionnaire de paramètres à la création
            # gstlearn est assez flexible pour gérer les clés (nlag, dlag, angle2D, codir, etc.)
            dir_param = gl.DirParam.create(**dir_params)
            varioparam.addDir(dir_param)

        # 5. Mapping de la méthode de calcul choisie vers l'énumération gstlearn
        calc_methods = {
            'E_VARIOGRAM': gl.ECalcVario.VARIOGRAM,
            'E_MADOGRAM': gl.ECalcVario.MADOGRAM,
            'E_RODOGRAM': gl.ECalcVario.RODOGRAM,
            'E_POISSON': gl.ECalcVario.POISSON,
            'E_GENERAL1': gl.ECalcVario.GENERAL1,
            'E_GENERAL2': gl.ECalcVario.GENERAL2,
            'E_GENERAL3': gl.ECalcVario.GENERAL3,
            'E_ORDER4': gl.ECalcVario.ORDER4,
            'E_TRANS1': gl.ECalcVario.TRANS1,
            'E_TRANS2': gl.ECalcVario.TRANS2,
            'E_BINORMAL': gl.ECalcVario.BINORMAL,
            'E_COVARIANCE': gl.ECalcVario.COVARIANCE,
            'E_COVARIANCE_NC': gl.ECalcVario.COVARIANCE_NC,
            'E_COVARIOGRAM': gl.ECalcVario.COVARIOGRAM
        }
        
        # Récupération de l'énumération correspondante (valeur par défaut : VARIOGRAM)
        calc_enum = calc_methods.get(calcul, gl.ECalcVario.VARIOGRAM)

        # Calculer le variogramme à partir du Db et des paramètres
        vario = gl.Vario.computeFromDb(varioparam, db, calc_enum)

        return vario, db


    def perform_cross_validation(
        self,
        db: gl.Db,
        model: gl.Model,
        neigh_type: str = 'moving',
        radius: float = None,
        nmaxi: int = None,
        prefix: str = "Xvalid",
        verbose: bool = False,
        plot_histograms: bool = True,
        plot_scatter: bool = True
    ) -> tuple[dict, gl.Db]:
        """
        Effectue une validation croisée (Unique ou Glissante) sur un jeu de données.

        Args:
            db (gl.Db): La base de données contenant les variables.
            model (gl.Model): Le modèle variographique ajusté.
            neigh_type (str): Type de voisinage ('unique' ou 'moving').
            radius (float): Rayon de recherche pour le voisinage glissant (requis si neigh_type='moving').
            nmaxi (int): Nombre maximum d'échantillons pour le voisinage glissant (requis si neigh_type='moving').
            prefix (str): Le préfixe utilisé pour nommer les colonnes de résultat.
            verbose (bool): Activer ou non les impressions de débogage de gstlearn.
            plot_histograms (bool): Afficher ou non les histogrammes des erreurs.
            plot_scatter (bool): Afficher ou non les nuages de points vraies vs estimées.

        Returns:
            tuple[dict, gl.Db]: Un dictionnaire avec les MSE calculées et la Db mise à jour.
        """
        
        # 1. Configuration du voisinage
        if neigh_type.lower() == 'unique':
            neigh = gl.NeighUnique.create()
        elif neigh_type.lower() == 'moving':
            if radius is None or nmaxi is None:
                raise ValueError("Pour un voisinage glissant ('moving'), 'radius' et 'nmaxi' doivent être spécifiés.")
            neigh = gl.NeighMoving.create(radius=radius, nmaxi=nmaxi)
        else:
            raise ValueError("Le paramètre 'neigh_type' doit être 'unique' ou 'moving'.")

        # 2. Configuration du mode verbeux (Debug)
        if verbose:
            gl.OptDbg.setReference(1)

        # 3. Exécution de la validation croisée
        namconv = gl.NamingConvention(prefix, True, True, False)
        
        err = gl.xvalid(
            db=db, 
            model=model, 
            neigh=neigh, 
            flag_xvalid_est=-1, 
            flag_xvalid_std=-1,  
            namconv=namconv
        )
        
        if err != 0:
            print(f"Attention: La validation croisée a retourné un code d'erreur ({err}).")

        # 4. Extraction, calcul des erreurs et graphiques
        all_names = db.getAllNames()
        z_names = db.getNamesByLocator(gl.ELoc.Z)
        
        # On cherche les colonnes générées par les flags -1 (estim et stdev)
        estim_cols = [name for name in all_names if "estim" in name and name.startswith(prefix)]
        stdev_cols = [name for name in all_names if "stdev" in name and name.startswith(prefix)]
        
        results = {}
        print("\n--- Résultats de la Validation Croisée ---")
        
        for z_name in z_names:
            # Trouver la colonne d'estimation correspondante
            matching_est = [col for col in estim_cols if z_name in col]
            if not matching_est:
                continue
            est_name = matching_est[-1] # Prendre la plus récente
            
            true_vals = np.array(db.getColumn(z_name))
            est_vals = np.array(db.getColumn(est_name))
            
            # Calcul de l'erreur brute (Estimé - Vrai)
            err_vals = est_vals - true_vals
            mse = np.nanmean(np.square(err_vals))
            results[f"MSE_{z_name}"] = mse
            print(f"\nMean squared cross-validation error ({z_name}): {round(mse, 3)}")
            
            if plot_histograms:
                vals_clean = err_vals[~np.isnan(err_vals)]
                if len(vals_clean) > 0:
                    plt.figure(figsize=(7, 4))
                    plt.hist(vals_clean, bins=30, color="blue", alpha=0.7, edgecolor='black')
                    plt.xlabel("Estimation Errors")
                    plt.ylabel("Fréquence")
                    plt.title(f"Cross-Validation: Erreur d'estimation ({z_name})")
                    plt.show()

            # Calcul de l'erreur standardisée
            matching_std = [col for col in stdev_cols if z_name in col]
            if matching_std:
                std_name = matching_std[-1]
                stdev_vals = np.array(db.getColumn(std_name))
                
                # Ignorer les divisions par zéro
                with np.errstate(divide='ignore', invalid='ignore'):
                    stderr_vals = err_vals / stdev_vals
                    
                mse_std = np.nanmean(np.square(stderr_vals))
                results[f"MSE_std_{z_name}"] = mse_std
                print(f"Mean squared standardized error ({z_name}): {round(mse_std, 3)}")
                
                if plot_histograms:
                    vals_clean_std = stderr_vals[~np.isnan(stderr_vals) & ~np.isinf(stderr_vals)]
                    if len(vals_clean_std) > 0:
                        plt.figure(figsize=(7, 4))
                        plt.hist(vals_clean_std, bins=30, color="green", alpha=0.7, edgecolor='black')
                        plt.xlabel("Standardized Errors")
                        plt.ylabel("Fréquence")
                        plt.title(f"Cross-Validation: Erreur standardisée ({z_name})")
                        plt.show()

            # 5. Nuages de points (Valeurs Vraies vs Valeurs Estimées)
            if plot_scatter:
                mask = ~np.isnan(true_vals) & ~np.isnan(est_vals)
                true_clean = true_vals[mask]
                est_clean = est_vals[mask]
                
                if len(true_clean) > 0:
                    corr_coef = np.corrcoef(true_clean, est_clean)[0, 1]
                    ss_res = np.sum((true_clean - est_clean) ** 2)
                    ss_tot = np.sum((true_clean - np.mean(true_clean)) ** 2)
                    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                    
                    plt.figure(figsize=(7, 6))
                    plt.scatter(
                        true_clean, est_clean, 
                        color="darkorange", alpha=0.7, edgecolor='k',
                        label=f'Estimations (points orange)\n$r$ (Corrélation) = {corr_coef:.3f}\n$R^2$ (Détermination) = {r2_score:.3f}'
                    )
                    
                    min_val = min(np.min(true_clean), np.min(est_clean))
                    max_val = max(np.max(true_clean), np.max(est_clean))
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Droite $y=x$ (pointillés rouges)')
                    
                    plt.xlabel(f"Valeurs vraies ({z_name})")
                    plt.ylabel(f"Valeurs estimées ({est_name.split('.')[-1]})")
                    plt.title(f"Validation Croisée : {z_name} vs Estimé")
                    plt.legend(loc='best')
                    plt.grid(True, alpha=0.3)
                    plt.show()

        return results, db
    
    import gstlearn as gl

    def fit_variogram_model(
        vario: gl.Vario,
        mode: str = 'auto',
        auto_types: list = None,
        manual_params: list = None,
        use_drift: bool = False,
        drift_order: int = 1
    ) -> gl.Model:
        """
        Ajuste un modèle sur un variogramme expérimental (automatique ou manuel).

        Args:
            vario (gl.Vario): Le variogramme expérimental calculé avec gstlearn.
            mode (str): 'auto' pour un ajustement automatique, 'manual' pour manuel.
            auto_types (list): Liste de chaînes définissant les structures autorisées
                               pour l'ajustement automatique.
                               Ex: ['E_NUGGET', 'E_SPHERICAL']
            manual_params (list): Liste de dictionnaires contenant les paramètres pour
                                  l'ajustement manuel.
                                  Ex (1 var) : [{'type': 'E_NUGGET', 'sill': 0.2}, 
                                                {'type': 'E_SPHERICAL', 'range': 150, 'sill': 0.8}]
                                  Ex (N vars): [{'type': 'E_NUGGET', 'sills': [[1.0, 0.5], [0.5, 2.0]]},
                                                {'type': 'E_SPHERICAL', 'range': 150, 
                                                 'sills': np.array([[0.8, 0.1], [0.1, 1.2]])}]
            use_drift (bool): Indique s'il faut ajouter une dérive (IRF) au modèle.
            drift_order (int): L'ordre de la dérive à appliquer si use_drift est True.

        Returns:
            gl.Model: Le modèle variographique ajusté.
        """
        
        # Mapping des chaînes de caractères vers l'énumération gl.ECov
        cov_map = {
            'E_NUGGET': gl.ECov.NUGGET,
            'E_SPHERICAL': gl.ECov.SPHERICAL,
            'E_EXPONENTIAL': gl.ECov.EXPONENTIAL,
            'E_GAUSSIAN': gl.ECov.GAUSSIAN,
            'E_CUBIC': gl.ECov.CUBIC,
            'E_MATERN': gl.ECov.MATERN,
            'E_LINEAR': gl.ECov.LINEAR,
            'E_POWER': gl.ECov.POWER
        }
        
        if mode.lower() == 'auto':
            # Initialisation d'un modèle vide (adapté automatiquement par model.fit)
            model = gl.Model()

            # Structures par défaut si non spécifiées
            if auto_types is None:
                auto_types = ['E_NUGGET', 'E_SPHERICAL', 'E_CUBIC']
                
            # Conversion des chaînes en types énumérés
            types_enum = [cov_map.get(t, gl.ECov.SPHERICAL) for t in auto_types]
            
            # Ajustement automatique de gstlearn
            err = model.fit(vario, types_enum)
            if err != 0:
                print(f"Attention: L'ajustement automatique a rencontré un code de retour non nul ({err}).")
                
        elif mode.lower() == 'manual':
            model = None
            if not manual_params:
                raise ValueError("En mode manuel, 'manual_params' doit être fourni et non vide.")
                
            for struct in manual_params:
                # Récupération du type de covariance
                c_type_str = struct.get('type', 'E_SPHERICAL')
                c_type = cov_map.get(c_type_str, gl.ECov.SPHERICAL)
                
                c_range = struct.get('range', 0.0)
                c_param = struct.get('param', 1.0) # Paramètre tiers (ex: paramètre nu pour le Matérn)
                
                # Récupération des paramètres de palier : 'sill' (univarié) ou 'sills' (multivarié)
                kwargs = {'range': c_range, 'param': c_param}
                if 'sills' in struct:
                    kwargs['sills'] = struct['sills']
                elif 'sill' in struct:
                    kwargs['sill'] = struct['sill']
                else:
                    kwargs['sill'] = 1.0  # Valeur par défaut

                if model is None:
                    # Création du modèle avec la première structure (configure automatiquement nvar)
                    try:
                        model = gl.Model.createFromParam(c_type, **kwargs)
                    except TypeError:
                        val_sill = kwargs.get('sills', kwargs.get('sill'))
                        model = gl.Model.createFromParam(c_type, c_range, c_param, val_sill)
                else:
                    # Ajout des structures suivantes
                    try:
                        # Appel dynamique avec sill ou sills selon ce qui a été fourni
                        model.addCovFromParam(c_type, **kwargs)
                    except TypeError:
                        # Solution de repli au cas où l'API soit positionnelle sans les kwargs explicites
                        val_sill = kwargs.get('sills', kwargs.get('sill'))
                        model.addCovFromParam(c_type, c_range, c_param, val_sill)
                    
        else:
            raise ValueError("Le mode spécifié doit être 'auto' ou 'manual'.")

        # Ajout de la dérive si l'utilisateur le demande
        if use_drift:
            err_drift = model.setDriftIRF(order=drift_order)
            if err_drift != 0:
                print(f"Attention: La configuration de la dérive a retourné un code d'erreur ({err_drift}).")
                
        return model
    
    def perform_kriging(
        self,
        db_in: gl.Db,
        model: gl.Model,
        neigh: int,
        radius: int,
        nmaxi: int,
        target_cols: list = None,
        grid_nx: list = None,
        grid_dx: list = None,
        grid_x0: list = None,
        is_3d: bool = False,
        prefix: str = "Kriging"
    ) -> gl.DbGrid:
        """
        Effectue une estimation par krigeage sur une grille.

        Args:
            db_in (gl.Db): La base de données d'entrée contenant les échantillons.
            model (gl.Model): Le modèle variographique ajusté.
            neigh (gl.Neigh): Le paramétrage du voisinage (Unique ou Glissant).
            target_cols (list ou str): Colonne(s) cible(s) pour l'estimation (utile si modélisé sur résidus).
            grid_nx (list): Nombre de mailles [nx, ny, nz]. Calculé automatiquement si None.
            grid_dx (list): Taille des mailles [dx, dy, dz]. Calculé automatiquement si None.
            grid_x0 (list): Origine de la grille [x0, y0, z0]. Calculée automatiquement si None.
            is_3d (bool): Vrai si l'espace est en 3D.
            prefix (str): Le préfixe utilisé pour nommer les colonnes de résultat.

        Returns:
            gl.DbGrid: La grille contenant les résultats d'estimation.
        """
        
        # 0. Sélection des variables cibles
        # Utile pour estimer les variables brutes si le modèle a été ajusté sur les résidus
        if target_cols is not None:
            db_in.clearLocators(gl.ELoc.Z)
            if isinstance(target_cols, str):
                target_cols = [target_cols]
            db_in.setLocators(target_cols, gl.ELoc.Z)

        # 1. Création de la grille couvrant les données (createCoveringDb)
        kwargs = {}
        if grid_nx is not None:
            kwargs['nx'] = grid_nx
        else:
            kwargs['nx'] = [100] * db_in.getNDim() # Par défaut 100 mailles
            
        if grid_dx is not None: kwargs['dx'] = grid_dx
        if grid_x0 is not None: kwargs['x0'] = grid_x0
            
        grid = gl.DbGrid.createCoveringDb(db_in, **kwargs)
        
        # 2. Exécution du Krigeage
        namconv = gl.NamingConvention(prefix, True, True, False)
        if neigh == 1:
            neighb = gl.NeighUnique.create()
        else:
            neighb = gl.NeighMoving.create(radius = radius, nmaxi = nmaxi)
            
        print("\n--- Lancement du Krigeage ---")
        err = gl.kriging(dbin=db_in, dbout=grid, model=model, neigh=neighb, 
                         flag_est=True, flag_std=True, flag_varz=False, 
                         namconv=namconv)
        
        if err != 0:
            print(f"Attention: Le krigeage a retourné un code d'erreur ({err}).")
        else:
            print("Krigeage terminé avec succès !")

        return grid
    
    def perform_simulation(
        self,
        db_in: gl.Db,
        model: gl.Model,
        neigh: int,
        radius: float,
        nmaxi: int,
        nbsimu: int = 1,
        nbtuba: int = 1000,
        is_block: bool = False,
        ndis: list = None,
        target_cols: list = None,
        grid_nx: list = None,
        grid_dx: list = None,
        grid_x0: list = None,
        is_3d: bool = False,
        seed: int = 12345,
        prefix: str = "Simu"
    ) -> gl.DbGrid:
        """
        Effectue des simulations conditionnelles par Bandes Tournantes sur une grille.

        Args:
            db_in (gl.Db): La base de données d'entrée contenant les échantillons.
            model (gl.Model): Le modèle variographique ajusté.
            neigh (int): Choix du type de voisinage (1 = Unique, autre = Glissant).
            radius (float): Rayon de recherche (utilisé si voisinage glissant).
            nmaxi (int): Nombre maximum de voisins (utilisé si voisinage glissant).
            nbsimu (int): Le nombre de réalisations à générer.
            nbtuba (int): Le nombre de bandes tournantes (ex: 1000).
            is_block (bool): True pour une simulation de bloc, False pour une simulation ponctuelle.
            ndis (list): Discrétisation des blocs (ex: [10, 10, 5]). Utilisé si is_block=True.
            target_cols (list ou str): Colonne(s) cible(s) pour la simulation.
            grid_nx (list): Nombre de mailles [nx, ny, nz]. Calculé automatiquement si None.
            grid_dx (list): Taille des mailles [dx, dy, dz]. Calculé automatiquement si None.
            grid_x0 (list): Origine de la grille [x0, y0, z0]. Calculée automatiquement si None.
            is_3d (bool): Vrai si l'espace est en 3D.
            seed (int): Graine de génération aléatoire pour garantir la reproductibilité.
            prefix (str): Le préfixe utilisé pour nommer les colonnes de résultat.

        Returns:
            gl.DbGrid: La grille contenant les résultats des simulations.
        """
        
        # 0. Sélection des variables cibles
        if target_cols is not None:
            db_in.clearLocators(gl.ELoc.Z)
            if isinstance(target_cols, str):
                target_cols = [target_cols]
            db_in.setLocators(target_cols, gl.ELoc.Z)

        # 1. Création de la grille couvrant les données (createCoveringDb)
        kwargs = {}
        if grid_nx is not None:
            kwargs['nx'] = grid_nx
        else:
            kwargs['nx'] = [100] * db_in.getNDim() # Par défaut 100 mailles
            
        if grid_dx is not None: kwargs['dx'] = grid_dx
        if grid_x0 is not None: kwargs['x0'] = grid_x0
            
        grid = gl.DbGrid.createCoveringDb(db_in, **kwargs)
        
        # 2. Configuration de l'aléatoire et exécution
        gl.law_set_random_seed(seed)
        namconv = gl.NamingConvention(prefix, True, True, False)
        
        if neigh == 1:
            neighb = gl.NeighUnique.create()
        else:
            neighb = gl.NeighMoving.create(radius=radius, nmaxi=nmaxi)
            
        print(f"\n--- Lancement des Simulations ({nbsimu} réalisations) ---")
        
        if is_block:
            raise ValueError(
                "gstlearn ne supporte pas directement la simulation de blocs via 'ndis' avec simtub().\n"
                "Pour simuler des blocs, vous devez créer une grille plus fine (ponctuelle), "
                "lancer la simulation, puis faire une moyenne par blocs (upscaling) post-simulation."
            )
        else:
            print("Mode : Simulation Ponctuelle")
            err = gl.simtub(
                dbin=db_in,
                dbout=grid,
                model=model,
                neigh=neighb,
                nbsimu=nbsimu,
                nbtuba=nbtuba,
                seed=seed,
                namconv=namconv
            )
        
        if err != 0:
            print(f"Attention: La simulation a retourné un code d'erreur ({err}).")
        else:
            print("Simulations terminées avec succès !")

        return grid