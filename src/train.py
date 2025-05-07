import os


print(f"--- Entra a Entrenamiento")

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Define Paths ---
# Usar rutas absolutas dentro del workspace del runner
workspace_dir = os.getcwd() 
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
# Definir explícitamente la ubicación base deseada para los artefactos
### DAVID REVISAR y COMPARAR CON EL EJERCICIO QUE SE HIZO ANTES. EL DEL ARTEFACTO PARA SUBIR EL CSV
artifact_location = "file://" + os.path.abspath(mlruns_dir)

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: MLRuns Dir: {mlruns_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Desired Artifact Location Base: {artifact_location} ---")



