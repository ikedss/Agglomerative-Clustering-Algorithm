from sklearn.cluster import AgglomerativeClustering
import plotly.figure_factory as ff
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time

df = pd.read_csv("Your file here")

# printa informações sobre o arquivo // print file information
print(df.describe(), df.head(), df.shape, df.info())

# apresenta um dendograma do arquivo // displays a dendrogram of the file
fig = ff.create_dendrogram(df)
fig.update_layout(width=800, height=500)
fig.show()

# verificar o tempo de clusterização e clusteriza as amostras do documento // check the clustering time and cluster the document samples
start = time.time()
clusters = AgglomerativeClustering(n_clusters=2, linkage = "ward").fit(df)
end = time.time()

# calcula a silhueta dos clusters // calculates the silhouette of the clusters
lable = clusters.labels_
print("silhouette_score:", metrics.silhouette_score(df, lable))

plot = sns.scatterplot(data = df, x = "X", y = "Y", hue = lable, legend = "full", palette = "deep")
sns.move_legend(plot, "upper right", bbox_to_anchor = (1.16, 1), title = "Clusters")

print("execution time:", end - start)
plt.show()