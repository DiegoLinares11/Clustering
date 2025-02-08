import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Parte 5 Consecuencia del Metodo del codo. 
# Cargar datos
df = pd.read_csv("../files/iris.csv")
df_con_respuestas = pd.read_csv("../files/iris-con-respuestas.csv")
# Calcular inercia para diferentes k

X = df[["sepal_length", "sepal_width"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
for k in [3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="sepal_length",
        y="sepal_width",
        hue=clusters,
        palette="tab10"
    )
    plt.title(f"Clusters (k={k}) con datos estandarizados")
    plt.xlabel("Longitud del sépalo (cm)")
    plt.ylabel("Ancho del sépalo (cm)")
    plt.savefig(f"../graphics/clustersUsados{k}.png")
    plt.show()