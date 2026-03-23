from geostat_library import geostat_library
vario_calc = geostat_library()


import pandas as pd

df = pd.read_csv(
    r"D:\Documents\ekolleessoh\Travail\Projet_ANDRA\Interpolation_cokrigeage\Original_data\Data_RGT_For_Sis.csv",
    sep=';',encoding="latin1")

# Paramètres de tendance/résidus (activé ici pour l'exemple)
resid = False     # Mettre True si vous souhaitez calculer un résidu avant le variogramme
degree = 1        # Dégré du polynôme (si resid=True)
method = 'simple' # Méthode de régression (si resid=True)
x_col = df['X']
y_col = df['Y']
z_col = df['Z']
directions = [
    {'nlag': 10, 'dlag': 500},  # Direction horizontale (X)
]

vario, db = vario_calc.calculate_variogram(df,
                                           True,
                                           ['X', 'Y', 'Z'], 
                                           ['RGT_For','RGT_Sis'],
                                           directions,
                                           'E_VARIOGRAM',
                                           resid,
                                           degree, 
                                           method, 
                                           x_col, 
                                           y_col, 
                                           z_col)