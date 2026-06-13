import pandas as pd
import re
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import numpy as np

# 1. PARSEO CON FECHA
log_path = 'dataset_entrenamiento.log'
data = []

# Captura "Failed" y "Accepted"
regex_linea = r'(?P<fecha>\w+\s+\d+\s+\d+:\d+:\d+).*sshd\[.*\]: (?:Failed|Accepted) password.*from (?P<ip>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'

try:
    with open(log_path, 'r') as file:
        for line in file:
            if 'Failed password' in line or 'Accepted password' in line: 
                match = re.search(regex_linea, line)
                if match:
                    fecha_str = match.group('fecha')
                    fecha_dt = datetime.strptime(f"{datetime.now().year} {fecha_str}", "%Y %b %d %H:%M:%S")
                    data.append({'IP': match.group('ip'), 'Tiempo': fecha_dt, 'Evento': 'Fallo' if 'Failed' in line else 'Exito'})
except Exception as e:
    print(f"Error: {e}")
    exit()

if not data:
    print("No se encontraron registros de SSH.")
    exit()

# 2. PROCESAMIENTO
df = pd.DataFrame(data)
df = df.sort_values(['IP', 'Tiempo'])

# Calculamos la diferencia temporal entre CUALQUIER evento de la misma IP
df['Diff_Tiempo'] = df.groupby('IP')['Tiempo'].diff().dt.total_seconds().fillna(0)

# --- LÍMITE DE TIEMPO (TIME CAPPING) ---
# Si pasan más de 3600 segundos (1 hora) entre intentos, lo limitamos a 3600.
# Esto evita que una pausa de "varios días" rompa el Ritmo_Promedio y el Ritmo_Estabilidad.
df['Diff_Tiempo'] = df['Diff_Tiempo'].clip(upper=3600)

# Agrupamos por IP
caracteristicas = df.groupby('IP').agg(
    Total_Intentos=('IP', 'count'),
    # Queremos saber cuántos fallos tuvo, esto es clave para la detección
    Total_Fallos=('Evento', lambda x: (x == 'Fallo').sum()),
    Ritmo_Promedio=('Diff_Tiempo', 'mean'),
    Ritmo_Estabilidad=('Diff_Tiempo', 'std') 
).fillna(0).reset_index()

caracteristicas['Ratio_Fallos'] = caracteristicas['Total_Fallos'] / caracteristicas['Total_Intentos']

# 3. DETECCIÓN
# Incluimos 'Total_Fallos' en el análisis para darle más contexto al modelo
caracteristicas_filtradas = caracteristicas[caracteristicas['Total_Intentos'] >= 4].copy()

# --- AJUSTE DE SENSIBILIDAD POR VOLUMEN ---
# Si el usuario tiene muy pocos intentos, vamos a 'suavizar' su ratio de fallos.
# Esto reduce el impacto de los usuarios que se equivocan pero mantiene a los atacantes de fuerza bruta.

def suavizar_ratio(row):
    if row['Total_Intentos'] < 10:
        # Si tiene pocos intentos, reducimos manual su ratio de fallos 
        # para que el modelo no lo vea tan "agresivo".
        return row['Ratio_Fallos'] * 0.5 
    return row['Ratio_Fallos']

caracteristicas_filtradas['Ratio_Fallos'] = caracteristicas_filtradas.apply(suavizar_ratio, axis=1)
X = caracteristicas_filtradas[['Total_Intentos', 'Ratio_Fallos', 'Total_Fallos', 'Ritmo_Promedio', 'Ritmo_Estabilidad']].copy()
# APLICAMOS PESO: Multiplicamos el Ratio_Fallos por 5 y Total_Fallos por 2.
# Esto hace que la diferencia entre 0.1 (normal) y 0.87 (atacante) sea grande para el modelo.
X['Ratio_Fallos'] = X['Ratio_Fallos'] * 5.0
X['Total_Fallos'] = X['Total_Fallos'] * 2.0

scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

modelo = IsolationForest(random_state=42)
modelo.fit(X_escalado)
puntajes = modelo.decision_function(X_escalado)

umbral_final = 0.0

caracteristicas_filtradas['Anomalia'] = np.where(puntajes < umbral_final, -1, 1)
caracteristicas_filtradas['Puntaje_Exacto'] = puntajes

# REGLA DE SENTIDO COMÚN
# Aca ignoramos al modelo si estamos seguros de que el usuario es normal
def regla_de_sentido_comun(row):
    # Si tiene bajo ratio de fallos (< 20%), es normal.
    if row['Ratio_Fallos'] < 0.20:
        return 1
    # Si el modelo dijo anomalía, pero el ratio es bajo, perdonamos.
    return row['Anomalia']

# Aplicamos la regla usando el 'Ratio_Fallos' original
caracteristicas_filtradas['Anomalia'] = caracteristicas_filtradas.apply(regla_de_sentido_comun, axis=1)

# 4. Armamos el resultado final incluyendo el puntaje
resultado_final = caracteristicas.merge(caracteristicas_filtradas[['IP', 'Anomalia', 'Puntaje_Exacto']], on='IP', how='left')

# Los usuarios con < 4 intentos que no pasaron por el modelo los dejamos como normales (1)
resultado_final['Anomalia'] = resultado_final['Anomalia'].fillna(1)
# Les ponemos un puntaje positivo "falso" para que se entienda que son normales
resultado_final['Puntaje_Exacto'] = resultado_final['Puntaje_Exacto'].fillna(0.10) 

print(resultado_final)
