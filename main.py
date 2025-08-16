import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
# --- 1. Synthetic Data Generation ---
# Generate synthetic patient data (discharge summaries and readmission status)
np.random.seed(42)  # for reproducibility
num_patients = 100
discharge_summaries = [
    "Patient had a successful surgery and is recovering well.",
    "Patient experienced complications and required additional treatment.",
    "Patient's condition improved significantly after medication.",
    "Patient's condition worsened and was readmitted.",
    "Routine post-op care, patient stable."
] * 20  # Simulate different summary types
readmission = np.random.randint(0, 2, num_patients)  # 0: no readmission, 1: readmission
data = {'summary': discharge_summaries, 'readmission': readmission}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preprocessing ---
# Create a TF-IDF matrix from the discharge summaries
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['summary'])
# --- 3. Unsupervised Clustering ---
# Apply KMeans clustering to group patients based on their discharge summaries
kmeans = KMeans(n_clusters=3, random_state=42) #Experiment with different n_clusters
kmeans.fit(tfidf_matrix)
df['cluster'] = kmeans.labels_
# --- 4. Analysis ---
# Analyze the characteristics of each cluster (e.g., average readmission rate)
cluster_analysis = df.groupby('cluster')['readmission'].agg(['mean', 'count'])
print("Cluster Analysis (Readmission Rate & Count):\n", cluster_analysis)
# --- 5. Visualization ---
# Visualize the clusters (e.g., readmission rate per cluster)
plt.figure(figsize=(8, 6))
plt.bar(cluster_analysis.index, cluster_analysis['mean'])
plt.xlabel("Cluster")
plt.ylabel("Average Readmission Rate")
plt.title("Readmission Rate per Cluster")
plt.savefig('readmission_rate_per_cluster.png')
print("Plot saved to readmission_rate_per_cluster.png")
#Further analysis could involve exploring the top words in each cluster to understand the characteristics of high-risk patients.  This would require additional code using the vectorizer.get_feature_names_out() and would be a natural extension of this analysis.