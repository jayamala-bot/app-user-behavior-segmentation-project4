# App User Behavior Segmentation (K-Means Clustering)

## Problem
An app has data on 50,000 users but no labels saying who is "good" or "about to leave".
I used unsupervised machine learning to group users by similar behavior.

## Dataset
- 50,000 users, 25 columns (usage, devices, subscription, scores)
- Only `rating_given` had missing values (about 10%)

## Approach
1. Loaded and understood the data
2. Cleaned it: filled missing ratings with the median, clipped extreme outliers
3. EDA: histograms and a correlation heatmap
4. Selected 10 behavior features (sessions, session length, pages viewed, days since last login, scores, etc.)
5. Scaled them with StandardScaler
6. Chose k with the Elbow Method and silhouette score
7. Trained K-Means with k = 4
8. Visualized with PCA
9. Profiled each cluster and listed the user IDs in each
10. Mapped each cluster to a business action

## Results
| Segment | Users | What sets it apart | Action |
|---|---|---|---|
| C0: Recently-Active | 15,842 | Last login about 8 days ago | Nurture and upsell |
| C1: Long-Session | 7,722 | About 28.6-minute sessions | Loyalty rewards |
| C2: Deep-Browsing + Lapsing | 13,131 | About 19.5 pages per session, about 31 days since login | Win-back campaign |
| C3: Light-Browsing + Lapsing | 13,305 | About 7.5 pages per session, about 31 days since login | Re-engagement digest |

![PCA plot](outputs/plots/04_pca_clusters.png)
![Elbow plot](outputs/plots/03_elbow_silhouette.png)

## Cluster quality (honest evaluation)
- Silhouette: 0.069 (weak separation)
- Davies-Bouldin: 2.80, Calinski-Harabasz: 3,300
- Shuffled, pattern-free data scores about the same (0.068), so this dataset has little natural structure.
- Segments are descriptive. They should be validated with real churn outcomes.

## Future work
Gaussian Mixture or hierarchical clustering, RFM features, real churn data, A/B tests.

## How to run
```
pip install -r requirements.txt
```
Open the notebook and click **Run All**.

## Folder structure
- `outputs/` plots, cluster profiles and user lists
- the notebook with the full analysis

