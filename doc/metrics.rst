*** Begin Patch
*** Update File: doc/metrics.rst
@@
-XXX can also be useful in a purely unsupervised setting as a
-  building block for a Consensus Index that can be used for clustering
-  model selection (TODO).
+XXX can also be useful in a purely unsupervised setting as a
+  building block for a **Consensus Index** that can be used for clustering
+  model selection. The Consensus Index aggregates multiple clustering
+  validation metrics (both internal and external) into a single score that
+  can be used to compare different clustering solutions when ground‑truth
+  labels are not available. Below is a short example of how to construct a
+  Consensus Index using scikit‑learn's clustering metrics.
+
+  .. rubric:: Example workflow
+
+  .. code-block:: python
+
+      import numpy as np
+      from sklearn.metrics import adjusted_rand_score, silhouette_score
+      from sklearn.cluster import KMeans
+
+      # Generate synthetic data
+      X, _ = make_blobs(n_samples=500, centers=4, random_state=0)
+
+      # Fit several clustering models with different numbers of clusters
+      models = [KMeans(n_clusters=k, random_state=0).fit(X) for k in range(2, 7)]
+
+      # Compute a set of validation scores for each model
+      scores = []
+      for model in models:
+          labels = model.labels_
+          # Internal metric – silhouette (higher is better)
+          sil = silhouette_score(X, labels)
+          # External metric – adjusted Rand (requires ground truth, here we use
+          # the true labels only for illustration; in practice you would drop
+          # this term when no ground truth is available)
+          # ari = adjusted_rand_score(true_labels, labels)
+          # Combine the metrics; we simply take the mean after scaling to
+          # the [0, 1] interval.  More sophisticated aggregation (e.g., weighted
+          # geometric mean) can also be used.
+          # Here we only use silhouette as an example of an unsupervised metric.
+          scores.append(sil)
+
+      # The consensus index is the model with the highest aggregated score
+      best_k = np.arange(2, 7)[np.argmax(scores)]
+      print(f"Best number of clusters according to the consensus index: {best_k}")
+
+  The above snippet demonstrates a minimal consensus approach based on the
+  silhouette score. In practice, you can extend the ``scores`` list with any
+  number of clustering metrics (e.g., Calinski‑Harabasz, Davies‑Bouldin, etc.)
+  and combine them using a weighted average or a more robust statistic such
+  as the median. The key idea is that each metric provides a different view of
+  clustering quality, and their aggregation yields a more stable model
+  selection criterion.
*** End Patch
*** End Patch