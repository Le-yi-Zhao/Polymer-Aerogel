import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
#核心思路：将particle_size和pore_size进行0，1向量化，成为3维向量，然后进行聚类
particle_df=pd.read_excel('polymer analysis\particle sizes.xlsx')
pore_df=pd.read_excel('polymer analysis\pore size.xlsx')
particle_long=particle_df.melt(var_name='Concentration',value_name='Value')
particle_long['Type']='Particle Size'
pore_long=pore_df.melt(var_name='Concentration',value_name='Value')
pore_long['Type']='Pore Size'
combined_df=pd.concat([particle_long,pore_long],ignore_index=True)
combined_df['Type_Code']=combined_df['Type'].map({'Particle Size':0,'Pore Size':1})
combined_df['Concentration']=combined_df['Concentration'].astype(float)
features=combined_df[['Concentration','Type_Code','Value']]
scaler=StandardScaler()
scaled_features=scaler.fit_transform(features)
#样本量较小，使用轮廓系数法选择聚类数量
range_n_clusters=range(2,10)
silhouette_scores=[]
for k in range_n_clusters:
    kmeans=KMeans(n_clusters=k,random_state=42)
    cluster_labels=kmeans.fit_predict(scaled_features)
    score=silhouette_score(scaled_features,cluster_labels)
    silhouette_scores.append(score)
optimal_k=range_n_clusters[np.argmax(silhouette_scores)]
kmeans_final=KMeans(n_clusters=optimal_k,random_state=42)
combined_df['Cluster']=kmeans_final.fit_predict(scaled_features)
#三维可视化操作
fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection='3d')
scatter=ax.scatter(
    combined_df['Concentration'],
    combined_df['Type_Code'],
    combined_df['Value'],
    c=combined_df['Cluster'],cmap='Set1',s=50
)
ax.set_title("Clustering")
ax.set_xlabel('Concentration')
ax.set_ylabel('Type Code(0=Particle size,1=Pore Size)')
ax.set_zlabel('Value')
plt.colorbar(scatter,label='Cluster')
plt.tight_layout()
plt.show()
plt.savefig('Clustering.png',dpi=300)
#基本特征统计
cluster_stats=combined_df.groupby(['Cluster','Type'])['Value'].agg(['count','mean','std','min','max']).reset_index()
print(cluster_stats)
cluster_stats.to_csv('cluster_stats.csv',index=False)
#查看每个簇的具体数据点
for i in range(0,optimal_k):
    combined_data=combined_df['Cluster']==i
    combined_data.to_csv(f"cluster_{i}_data.csv",index=False)
#提取特定簇的数据(take cluster1 as an example,后续可以打印出来)
cluster_2_particles=combined_df[(combined_df['Cluster']==2)&(combined_df['Type_Code']==0)]
print(cluster_2_particles)
#标记异常值（这里只是通过四分位方法进行分析，可以考虑结合专家知识进行参数确定）
particle_df_only=combined_df[combined_df['Type']=='Particle Size']
Q1=particle_df_only['Value'].quantile(0.25)
Q3=particle_df_only['Value'].quantile(0.75)
IQR=Q3-Q1#这部分的参数需要通过专家参数进行分析