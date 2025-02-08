import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Visualizar datos (sépalo) Parte 2. 
# Clustering
# Seleccionar características del sépalo
# Cargar datos
df = pd.read_csv("../files/iris.csv")
df_con_respuestas = pd.read_csv("../files/iris-con-respuestas.csv")

X = df[["sepal_length", "sepal_width"]]

# Modelo K-Means con 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

# Graficar clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="sepal_length",
    y="sepal_width",
    hue=clusters,
    palette="tab10"
)
plt.title("Clusters (k=2) basados en el sépalo")
plt.xlabel("Longitud del sépalo (cm)")
plt.ylabel("Ancho del sépalo (cm)")
plt.savefig("../images/Clusters.png")  
plt.show()