import pandas as pd
from sklearn.model_selection import cross_val_score
from kneed import KneeLocator


def _search_optimal_n_components(
        transformer,
        X_train,
        max_n_components_to_create,
        y_train,
        prefix,
        estimator,
        early_stopping=10,
) -> (int, float):

    # cv_scores_dict
    # key n_components: int
    # value cv_score: float
    cv_scores_dict = {}

    last_knee = None
    last_knee_found_n_components = None

    # LDA restrictions:
    # n_components cannot be larger than min(n_features, n_classes - 1)
    if transformer.__class__.__name__ == "LinearDiscriminantAnalysis":
        n_classes = len(y_train.value_counts())
        range_upper_bound = min(n_classes, len(X_train.columns))
    else:
        range_upper_bound = min(max_n_components_to_create + 1, len(X_train.columns))

    # test different parameters for n_components
    for n_components in range(1, range_upper_bound):
        print("\n---")
        print(f"test n_components = {n_components}")

        # K-Means uses n_clusters instead of n components
        if transformer.__class__.__name__ == 'MiniBatchKMeans':
            transformer.set_params(**{"n_clusters": n_components})

        else:
            transformer.set_params(**{"n_components": n_components})

        print("fit...")
        transformer.fit(X_train, y_train)

        print("transform train data...")
        # use predict for kmeans instead of transform
        if transformer.__class__.__name__ == 'MiniBatchKMeans':
            X_train_trans = pd.DataFrame(transformer.predict(X_train)).add_prefix(prefix)
        else:
            X_train_trans = pd.DataFrame(transformer.transform(X_train)).add_prefix(prefix)

        # calc cross validation score using the new features and baseline features
        X_train_trans_baseline = pd.concat([X_train, X_train_trans], axis="columns")

        cv_score = cross_val_score(estimator, X_train_trans_baseline, y_train, cv=5, n_jobs=-1).mean()

        cv_scores_dict[n_components] = cv_score

        print(f"Cross validation score: {cv_score}")

        # condition of a knee is at least two points
        if len(cv_scores_dict) > 1:
            # detect the knee
            kneedle = KneeLocator(
                x=list(cv_scores_dict.keys()),
                y=list(cv_scores_dict.values()),
                curve="concave",
                direction="increasing"
            )

            # knee
            if kneedle.knee:
                if last_knee is None:
                    last_knee = kneedle.knee
                    last_knee_found_n_components = n_components

                elif kneedle.knee > last_knee:
                    last_knee = kneedle.knee
                    last_knee_found_n_components = n_components

                print(f"knee found at x={kneedle.knee}")

                # check for early stopping
                rounds_since_last_knee = n_components - last_knee_found_n_components
                print(f"rounds since last knee-point found: {rounds_since_last_knee}")

                if rounds_since_last_knee >= early_stopping:
                    print(f"Early stopping of {early_stopping} reached.")
                    break

            else:
                print(f"no knee found so far")

    # optimal n_components should be the last found knee point
    optimal_n_components = last_knee
    print("\n---")
    print(f"optimal n_components = {optimal_n_components}")
    print("---\n")

    # optimal n_components can be None when there are just a few features to test and no knee can be detected.
    if optimal_n_components is None:
        print(f"optimal n_components is None. Fall back to the best cv_score n_components")
        optimal_n_components_by_cv_score = max(cv_scores_dict, key=cv_scores_dict.get)

        print(f"optimal n_components determined by cv_score: {optimal_n_components_by_cv_score}")

        return optimal_n_components_by_cv_score, max(cv_scores_dict.values())

    return optimal_n_components, cv_scores_dict[optimal_n_components]