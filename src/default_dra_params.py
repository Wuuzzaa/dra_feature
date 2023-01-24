DEFAULT_DRA_PARAMS = \
    {
        "pca": {
            "random_state": 42,
            "svd_solver": "full",  # "arpack",
        },
        "kpca": {
            # "n_components": None,
            "random_state": 42,
            "kernel": "rbf",
            "n_jobs": -1,
            "copy_X": False,

            # "auto" did run in a first test. "randomized" is faster and should be used when n_components is low according to
            # sklearn docu/guide.
            "eigen_solver": "randomized"
        },
        "lda":
            {},
        "umap": {
            # for clustering https://umap-learn.readthedocs.io/en/latest/clustering.html
            "n_neighbors": 15,  # default 15.
            "n_jobs": -1,

            # do not use a random state if you want to run umap on all cores according to faq
            # https://umap-learn.readthedocs.io/en/latest/faq.html
            # tested this and it makes no difference for me
            "random_state": 42,
            "verbose": False,
            "min_dist": 0,
        },
        "kmeans":
            {
                # "n_clusters": 8, # DO NOT SET THIS BECAUSE WE USE BRUTE FORCE TO DETERMINE THIS VALUE
                "batch_size": 256 * 16,  # 256 * cpu threads is suggested in sklearn docu
                "verbose": 0,
                "random_state": 42,
            },
    }