# Cluster Analysis

The focus of this project is building a **k-means** cluster algorithm from scratch without use of any publicly avaliable libraries. The algorithm is then applied for Fashion MNIST dataset.

## Clustering

Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group called cluster are more similar to each other than to those in other groups or clusters. We can use clustering analysis to gain some valuable insights from our data by seeing what groups the data points fall into when we apply a clustering algorithm

| ![eda of mnist data](images/exploration1.PNG) | ![eda of cluster](images/exploration2.PNG) |
| :-------------------------------------------: | :----------------------------------------: |


There are two main types of classification:

    - k-means clustering
    - Hierarchical clustering

## k-means Clustering

k-means is one of the simplest unsupervised learning algorithms that solve the well known clustering problem. The procedure follows a simple and easy way to classify a given data set through a certain number of clusters (assume k clusters) fixed apriori. The main idea is to define k centers, one for each cluster. These centers should be placed in a cunning way because of different location causes different result. So, the better choice is to place them as much as possible far away from each other. The next step is to take each point belonging to a given data set and associate it to the nearest center. When no point is pending, the first step is completed and an early group age is done. At this point we need to re-calculate k new centroids as barycenter of the clusters resulting from the previous step. After we have these k new centroids, a new binding has to be done between the same data set points and the nearest new center. A loop has been generated. As a result of this loop we may notice that the k centers change their location step by step until no more changes are done or in other words centers do not move any more. Finally, this algorithm aims at minimizing an objective function know as squared error

| ![eda of mnist data](images/cluster1.PNG) | ![eda of cluster](images/cluster2.PNG) | ![eda of cluster](images/cluster3.PNG) |
| :---------------------------------------: | :------------------------------------: | -------------------------------------- |


### pseudocode:

- Input dataset D, with k no of clusters
- Initialize cluster representatives C
- Randomly choose k data points from D
- Use k points as initial set of cluster representatives C
- repeat
  - Reassign points in D to closest cluster mean
  - Update m such that m_i is cluster ID of ith point in D
- Update C such that c_i is mean of points in jth cluster until convergence

```
kmeans(trainingSet,maxIterations,metric,metric_type)
where,
trainingSet   --> A comma seperated csv file that need not be pre processed
maxIterations --> Maximum number of iterations if not algorithm converges
metric        --> True/False
metric_type   --> wcSSE,silhouetteCoeff,nmi

```

#### Cluster Validation

    1. Silhoutte
        Silhouette refers to a method of interpretation and validation of consistency within clusters of data. The technique provides a succinct graphical representation of how well each object has been classified.The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.The silhouette can be calculated with any distance metric, such as the Euclidean distance or the Manhattan distance.

    2. Within Cluster Sum of squares
        The sum of the squared deviations from each observation and the cluster centroid.
        Interpretation. The within-cluster sum of squares is a measure of the variability of the observations within each cluster. In general, a cluster that has a small sum of squares is more compact than a cluster that has a large sum of squares. Clusters that have higher values exhibit greater variability of the observations within the cluster.

    3. Normalized mutuall information gain
        Normalized Mutual Information (NMI) is a normalization of the Mutual Information (MI) score to scale the results between 0 (no mutual information) and 1 (perfect correlation). In this function, mutual information is normalized by some generalized mean of H(labels_true) and H(labels_pred)), defined by the average_method.

| ![metric evaluation](images/group.JPG) |
| :------------------------------------: |

