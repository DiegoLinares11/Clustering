import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Metodo del codo Parte 4. 
# Cargar datos
df = pd.read_csv("../files/iris.csv")
df_con_respuestas = pd.read_csv("../files/iris-con-respuestas.csv")
# Calcular inercia para diferentes k

X = df[["sepal_length", "sepal_width"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker="o")
plt.xticks(range(1, 11))
plt.title("Método del codo")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.savefig("../images/MetodoDelCodo.png")  
plt.show()