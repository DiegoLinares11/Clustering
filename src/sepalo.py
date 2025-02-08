import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar datos
df = pd.read_csv("../files/iris.csv")
df_con_respuestas = pd.read_csv("../files/iris-con-respuestas.csv")

# Visualizar datos (sépalo) Parte 1.
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="sepal_length",
    y="sepal_width",
    palette="viridis"
)
plt.title("Relación entre longitud y ancho del sépalo")
plt.xlabel("Longitud del sépalo (cm)")
plt.ylabel("Ancho del sépalo (cm)")
plt.savefig("../graphics/sepalo_raw.png")  
plt.show()
