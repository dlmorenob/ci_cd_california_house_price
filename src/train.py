import os
import mlflow
import sys
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import traceback
from mlflow.models import infer_signature

print(f"--- Entra a Entrenamiento del dataset")
print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Define Paths ---
# Usar rutas absolutas dentro del workspace del runner
workspace_dir = os.getcwd() 
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
#tracking_uri = "http://127.0.0.1:8089"
# Definir explícitamente la ubicación base deseada para los artefactos
### DAVID REVISAR y COMPARAR CON EL EJERCICIO QUE SE HIZO ANTES. EL DEL ARTEFACTO PARA SUBIR EL CSV
artifact_location = "file://" + os.path.abspath(mlruns_dir)

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: MLRuns Dir: {mlruns_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Desired Artifact Location Base: {artifact_location} ---")


# --- Asegurar que el directorio MLRuns exista ---
os.makedirs(mlruns_dir, exist_ok=True)

# --- Configurar MLflow ---
mlflow.set_tracking_uri(tracking_uri)


# --- Crear o Establecer Experimento Explícitamente con Artifact Location ---
experiment_name = "github-mlops-californiaHousing"
experiment_id = None # Inicializar variable

try:
    # Intentar crear el experimento, proporcionando la ubicación del artefacto
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location # ¡Forzar la ubicación aquí!
    )
    print(f"--- Debug: Creado Experimento '{experiment_name}' con ID: {experiment_id} ---")
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"--- Debug: Experimento '{experiment_name}' ya existe. Obteniendo ID. ---")
        # Obtener el experimento existente para conseguir su ID
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"--- Debug: ID del Experimento Existente: {experiment_id} ---")
            print(f"--- Debug: Ubicación de Artefacto del Experimento Existente: {experiment.artifact_location} ---")
            # Opcional: Verificar si la ubicación del artefacto es la correcta
            if experiment.artifact_location != artifact_location:
                print(f"--- WARNING: La ubicación del artefacto del experimento existente ('{experiment.artifact_location}') NO coincide con la deseada ('{artifact_location}')! ---")
        else:
            # Esto no debería ocurrir si RESOURCE_ALREADY_EXISTS fue el error
            print(f"--- ERROR: No se pudo obtener el experimento existente '{experiment_name}' por nombre. ---")
            sys.exit(1)
    else:
        print(f"--- ERROR creando/obteniendo experimento: {e} ---")
        raise e # Relanzar otros errores

# Asegurarse de que tenemos un experiment_id válido
if experiment_id is None:
    print(f"--- ERROR FATAL: No se pudo obtener un ID de experimento válido para '{experiment_name}'. ---")
    sys.exit(1)


#################################################################################
#################################################################################
# --- Cargar Datos y Entrenar Modelo ---
df = pd.read_csv("../data/housing.csv")
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

X_train_val, X_test, y_train_val, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)


print("Probando una regresión lineal")
model = LinearRegression()
model.fit(X_train,y_train)

r_squared = model.score(X_train, y_train)
print(f"R-squared: {r_squared:.4f}")

####### validate part
y_predict = model.predict(X_train)
lin_rmse = root_mean_squared_error(y_predict, y_train)
print(f"Linear model rmse: {lin_rmse}")

print("Ahora probando una regresión lineal")
#### using a forest regression algorithm
#### there is overfitting

forest_reg =  RandomForestRegressor(random_state=42)
forest_reg.fit(X_train,y_train)

r_squared = forest_reg.score(X_train, y_train)
print(f"R-squared: {r_squared:.4f}")

####### validation part 
y_predict = forest_reg.predict(X_train)
rf_rmse  = root_mean_squared_error(y_predict, y_train)
print(f"Random Forest: {rf_rmse:.4f}")
#################################################################################
#################################################################################

# Infer signature & log with input example
signature = infer_signature(X_train, model.predict(X_train))
input_example = X_train[0:1]  # Example: first row

# --- Iniciar Run de MLflow ---
print(f"-- Debug: Iniciando run de MLflow en Experimento ID: {experiment_id} ---")
# Añadir ID aquí
run = None
try:
    # Iniciar el run PASANDO EXPLÍCITAMENTE el experiment_id
    with mlflow.start_run(experiment_id=experiment_id) as run: # <--- CAMBIO CLAVE
        run_id = run.info.run_id
        actual_artifact_uri = run.info.artifact_uri
        print(f"--- Debug: Run ID: {run_id} ---")
        print(f"--- Debug: URI Real del Artefacto del Run: {actual_artifact_uri} ---")
        # Comprobar si coincide con el patrón esperado basado en artifact_location del experimento
        # (La artifact_uri del run incluirá el run_id)
        expected_artifact_uri_base = os.path.join(artifact_location, run_id, "artifacts")
        if actual_artifact_uri != expected_artifact_uri_base:
            print(f"--- WARNING: La URI del Artefacto del Run '{actual_artifact_uri}' no coincide exactamente con la esperada '{expected_artifact_uri_base}' (esto puede ser normal si la estructura difiere ligeramente). Lo importante es que NO sea la ruta local incorrecta. ---")
            #if "/home/david/Documents/maestria_ean_ciencias_datos/mlops/ci_cd_mlops/ci-cd-mlops" in actual_artifact_uri:
            #    print(f"--- ¡¡¡ERROR CRÍTICO!!!: La URI del Artefacto del Run '{actual_artifact_uri}' TODAVÍA contiene la ruta local incorrecta! ---")
        
        mlflow.log_metric("lm-mse", lin_rmse)
        mlflow.log_metric("rf-mse", rf_rmse)
        print(f"--- Debug: Intentando log_model con artifact_path='model' ---")
        mlflow.sklearn.log_model(
             sk_model=forest_reg
            ,artifact_path="model"
            ,signature = infer_signature(X_train, forest_reg.predict(X_train))
            ,input_example =input_example 
        )
        print(f" Modelo registrado correctamente. Modelo Random Forest MSE: {rf_rmse:.4f}")
except Exception as e:
    print(f"\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    print(f"--- Fin de la Traza de Error ---")
    print(f"CWD actual en el error: {os.getcwd()}")
    print(f"Tracking URI usada: {mlflow.get_tracking_uri()}")
    print(f"Experiment ID intentado: {experiment_id}") # Añadir ID aquí
    if run:
        print(f"URI del Artefacto del Run en el error: {run.info.artifact_uri}")
    else:
        print("El objeto Run no se creó con éxito.")
    sys.exit(1)