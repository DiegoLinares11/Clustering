import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Estandarizacion Parte 3.
df = pd.read_csv("../files/iris.csv")
df_con_respuestas = pd.read_csv("../files/iris-con-respuestas.csv")
# Estandarizar características

X = df[["sepal_length", "sepal_width"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Nuevo modelo con datos estandarizados
kmeans_scaled = KMeans(n_clusters=2, random_state=42)
clusters_scaled = kmeans_scaled.fit_predict(X_scaled)

# Graficar clusters estandarizados
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="sepal_length",
    y="sepal_width",
    hue=clusters_scaled,
    palette="tab10"
)
plt.title("Clusters (k=2) con datos estandarizados")
plt.xlabel("Longitud del sépalo (cm)")
plt.ylabel("Ancho del sépalo (cm)")
plt.savefig("../images/Estandarizacion.png")  
plt.show()