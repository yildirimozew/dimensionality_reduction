In this three part work:\
  For the first part:\
  1- I implemented three distance/similarity metrics (Cosine, Minkowski, Mahalanobis).\
  2- I implemented a KNN class.\
  3- I performed 10-fold cross-validation (with stratification) for hyperparameter tuning (via grid search).\
  \
  For the second part:\
  1- Visualized the elbow method for kmeans and kmedoids, separately for loss and silhouette scores.\
  2- Determined the best k value using the visualizations.\
  3- Implemented PCA and Autoencoder classes from scratch.\
  4- I employed PCA, autoencoder, t-SNE and UMAP to visualize a high dimensional dataset in 2-D.\
  5- Calculated the worst-case run time analysis of kmeans.
  \
  For the third part: \
  1- Visualized clustering using HAC and DBSCAN algorithms.\
  2- Performed grid hyperparameter search for HAC and DBSCAN.\
  3- Performed silhouette analysis for HAC and DBSCAN hyperparameter configurations.\
  4- I employed PCA, autoencoder, t-SNE and UMAP to visualize another high dimensional dataset in 2-D.\
  5- Calculated the worst-case run time analysis of HAC.\
\
Results for the Part 1:\
![image](https://github.com/user-attachments/assets/6ebd3cc7-0eb9-47eb-b1dd-843ad872b257)
\
Results for the Part 2:\
\
Resulting plots for kmeans with confidence intervals:

![dataset1-kmeans-loss](https://github.com/user-attachments/assets/81a7596f-90dd-4b79-8716-8b698671e6ce)
![dataset1-kmeans-silhouette](https://github.com/user-attachments/assets/3c52d29d-7961-40f7-97cd-6a9cf0fec98e)\
As can be seen k=6 seem to be the best k value for this dataset according to both loss and silhouette scores.\
\
Resulting plots for kmedoids with confidence intervals:
![dataset1-kmedoids-loss](https://github.com/user-attachments/assets/276dd071-3285-4242-a7ea-75b23b900344)
![dataset1-kmedoids-silhouette](https://github.com/user-attachments/assets/b565c923-d20e-4811-9afa-a8aec2f3dab2)\
It can be seen that the elbow point is at k=6 and silhouette score tops at that point as well, thus it is our
optimal k.\
\
Here are the plots of the reduction methods I’ve used for Dataset1:\
![dataset1-autoencoder](https://github.com/user-attachments/assets/f7e8d780-7ef5-4025-b9d6-fbf7ae50cef8)
![dataset1-pca](https://github.com/user-attachments/assets/0db316c3-8fd2-4c75-bc4e-920a3ae151f0)
![dataset1-tsne](https://github.com/user-attachments/assets/63be336f-31a1-4ad1-a812-484ac99376a1)
![dataset1-umap](https://github.com/user-attachments/assets/97c20c87-f452-4f69-b142-82bb558eec8e)\
We can see that every method gave nearly the same results with 6 distinct clusters. Thus our result from
elbow method was accurate.\
\
Worst-case run time analysis:\
For kmeans, in every iteration, every point(N) is compared with each cluster centroid(K) to find the nearest one.
Computing the distances between them depends on vector dimension(d). This is repeated I times so complexity
is O(K * N * I * d) Updating the centroids have the same complexity. For kmedoids, assigning centers to points
is the same as kmeans, however when you want to update the medoids, you have to compute the cost for every n
data points instead of calculating only once, thus the complexity becomes O(n2). The final complexity becomes
O(K* N 2 * I * d).\

Results for Part 3:\
Since there are a lot of plots in this part, I'm only adding a sample of them. You can see all the results and plots in report.pdf.\
![HAC-1-2](https://github.com/user-attachments/assets/20d264d7-e68e-4352-a93b-75606a2dc996)
![HAC-1](https://github.com/user-attachments/assets/f29d2a5a-0943-46b7-83a2-b52d33835663)
![DBSCAN1](https://github.com/user-attachments/assets/a63ab213-a9ee-45d7-b861-01cfbe5e5d25)\

Worst-case run time analysis:\
HAC starts by calculating distances between all pairs of data points, which takes O(n2 ∗D). Then merges the
clusters which takes O(n2). This is done n times so the complexity is O(n3). If the algorithm utilizes a heap it
can be optimized to O(n2logn) Thus the final equation becomes O(n2 ∗d + n2logn). 

