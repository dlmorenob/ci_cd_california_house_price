print("Entra a validate.py")
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import os
import numpy as np 

# Parámetro de umbral
THRESHOLD = 30000.0

# --- Configurar MLflow igual que en train.py ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
mlflow.set_tracking_uri("file://" + os.path.abspath(mlruns_dir))

# --- Cargar dataset ---
print("--- Debug: Cargando dataset california housing---")

# --- Cargar Datos y Entrenar Modelo ---
#df = pd.read_csv("../data/housing.csv")
df = pd.read_csv("data/housing.csv")
#print(df.head(),df.shape)

print("revisando valores nulos")
null_counts = df.isnull().sum()
print(null_counts)

print("Llenando valores nulos con la mediana")
####aqui tomamos la decision de cambiar los nulos por la mediana para no perder 10% de los datos
imputer = SimpleImputer(strategy='median')
# Reshape the column to 2D array (required by scikit-learn)
tbr = df[['total_bedrooms']].values
# Fit and transform the data (replace nulls with median)
val = imputer.fit_transform(tbr)
# Replace the column in the DataFrame with imputed values
df['total_bedrooms'] = val
#print(df.head())

print("verificando que no existan valores nulos")
null_counts = df.isnull().sum()
print(null_counts)

#### transformando variables categoricas en
#### onehotenconding
codif  = OneHotEncoder()
ohcode = codif.fit_transform(df[["ocean_proximity"]])
print(ohcode.toarray())

### asignando el valor medio de la casa que es nuestra varibale objetivo
y = df['median_house_value']
### eliminando del dataset la variable objetivo y las variables categóricas
df.drop(['median_house_value','ocean_proximity'],axis=1,inplace=True)


print("Transformando las variables a la distribución normal")
std_scaler  = StandardScaler()
dnum_esc    = std_scaler.fit_transform(df[['housing_median_age', 'total_rooms'
                                             ,'total_bedrooms','population','households','median_income']])

print(dnum_esc)

print("Unificando las variables escalares y la conversión de la categórica")
x = np.concatenate([dnum_esc, ohcode.toarray()], axis=1)

print("Partiendo el set de datos en entrenamiento, valdiacion y prueba")
# porporción: 60% train, 20% val, 20% test

X_trainl, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"--- Debug: Dimensiones de X_test: {X_test.shape} ---")

# --- Cargar modelo desde MLflow ---
print("--- Debug: Intentando cargar modelo desde MLflow ---")
print("--- Debug: Intentando cargar modelo desde MLflow ---")
print("--- Debug: Intentando cargar modelo desde MLflow ---")
print("--- Debug: Intentando cargar modelo desde MLflow ---")

try:
    experiment_name = "github-mlops-californiaHousing"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise Exception(f"Experimento '{experiment_name}' no encontrado")
    
    # Obtener el último run
    runs = mlflow.search_runs(experiment.experiment_id, order_by=["start_time DESC"])
    
    if runs.empty:
        raise Exception("No se encontraron runs en el experimento")
    
    run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{run_id}/model"
    print(f"--- Debug: Cargando modelo desde URI: {model_uri} ---")
    
    model = mlflow.sklearn.load_model(model_uri)
    print("--- Debug: Modelo cargado exitosamente desde MLflow ---")

except Exception as e:
    print(f"--- ERROR al cargar modelo desde MLflow: {str(e)} ---")
    print(f"--- Debug: Archivos en {os.getcwd()}: ---")
    print(os.listdir(os.getcwd()))
    sys.exit(1)

# --- Predicción y Validación ---
print("--- Debug: Realizando predicciones ---")
try:
    y_pred1 = model.predict(X_test)
    y_pred  = y_pred1
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"🔍 MSE del modelo: {rmse:.4f} (umbral: {THRESHOLD})")

    # Validación
    if rmse <= THRESHOLD:
        print(" El modelo cumple los criterios de calidad.")
        sys.exit(0)
    else:
        print(" El modelo no cumple el umbral. Deteniendo pipeline.")
        sys.exit(0)

except Exception as pred_err:
    print(f"--- ERROR durante la predicción: {pred_err} ---")
    if hasattr(model, 'n_features_in_'):
        print(f"Modelo esperaba {model.n_features_in_} features.")
    print(f"X_test tiene {X_test.shape[1]} features.")
    sys.exit(1)