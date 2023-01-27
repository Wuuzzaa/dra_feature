import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from search_optimal_n_components import _search_optimal_n_components
from umap import UMAP
from src.dra_feature.default_dra_params import DEFAULT_DRA_PARAMS


class DraFeature:
    def __init__(
            self,
            dra_params: dict = None,
            estimator: BaseEstimator = None,
            random_state: int = 42,
            sample_size: int = 1_000,
            train_size: float = 0.7,
            max_n_components_to_create=3,
            return_only_improving_featuretypes=True,
            early_stopping=10,
    ):
        self.dra_params = dra_params
        self.estimator = estimator
        self.random_state = random_state
        self.sample_size = sample_size
        self.train_size = train_size
        self.max_n_components_to_create = max_n_components_to_create
        self.return_only_improving_featuretypes = return_only_improving_featuretypes
        self.early_stopping = early_stopping

        self.cv_scores_sample = None
        self.basefeaturenames = None
        self.dra_transformers = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # load default parameters for dimension reduction algorithms
        if dra_params is None:
            self.dra_params = DEFAULT_DRA_PARAMS

        # set random forest classifier as default if no other estimator was provided
        if self.estimator is None:
            self.estimator = RandomForestClassifier(n_jobs=-1, random_state=self.random_state)

    def _train_test_sample_split(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            train_size=self.train_size,
            random_state=self.random_state
        )

        # use sampling
        if self.sample_size is not None:
            if len(self.X_train) > self.sample_size:
                self.X_train = self.X_train.head(self.sample_size)
                self.y_train = self.y_train.head(self.sample_size)

        # reset index
        self.X_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)
        self.y_test.reset_index(drop=True, inplace=True)

    def print_plan(self):
        print("Plan to generate Features based on Dimension Reduction Algorithms:")
        for featuretype, parameters in self.dra_params.items():
            print("---")
            print(f"Featuretype: {featuretype}:")
            print("---")
            print("Parameters:")

            for key, value in parameters.items():
                print(f"{key}: {value}")

            print()

        print("---")
        print(f"Estimator used:")
        print("---")
        print(self.estimator.__class__.__name__)
        print()
        print("Estimator parameter:")
        for key, value in vars(self.estimator).items():
            print(f"{key}: {value}")
        print()

        print("---")
        print("Samplesize used for training:")
        print("---")
        print(f"{self.sample_size}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # train test split with sample size constraint
        self._train_test_sample_split(X, y)

        # clear the transformers and cv_scores
        self.dra_transformers = {}
        self.cv_scores_sample = {}

        # extract the basefeaturenames
        self.basefeaturenames = [str(x) for x in X.columns]

        # get the cv score for the basefeatures
        self.cv_scores_sample["baseline"] = cross_val_score(self.estimator, self.X_train, self.y_train, cv=5).mean()

        for featuretype, params in self.dra_params.items():
            print()
            print("#"*80)
            print(f"fit {featuretype}")
            print(f"baseline cross validation score: {self.cv_scores_sample['baseline']}")
            print("#"*80)
            print()

            # matches the featuretype strings like "pca" or "lda" to the object init method.
            transformers_dict = {
                "pca": PCA(),
                "kpca": KernelPCA(),
                "lda": LinearDiscriminantAnalysis(),
                "umap": UMAP(),
                "kmeans": MiniBatchKMeans()
            }

            if featuretype not in transformers_dict:
                raise(ValueError(f"{featuretype} is not known."))

            transformer = transformers_dict[featuretype]
            transformer.set_params(**params)
            self.dra_transformers[featuretype] = self._fit_transformer(transformer, featuretype)

    def transform(self, X:pd.DataFrame, y:pd.Series):
        print()
        print("#"*80)
        print("Transform")
        print("#" * 80)
        print()
        print(f"Baseline cross validation score: {self.cv_scores_sample['baseline']}")

        for featuretype, params in self.dra_params.items():
            print()
            print("---")
            print(f"Featuretype: {featuretype}")
            print(f"cross validation score: {self.cv_scores_sample[featuretype]}")
            print("---")

            # skip featuretypes which do not improve the cross validation score
            if self.return_only_improving_featuretypes:
                if self.cv_scores_sample["baseline"] >= self.cv_scores_sample[featuretype]:
                    print(f"Featuretype: {featuretype} do not improve the cross validation score. skip it.")
                    continue

            print(f"generate {featuretype} features")
            transformer = self.dra_transformers[featuretype]

            # use predict for kmeans instead of transform
            if transformer.__class__.__name__ == 'MiniBatchKMeans':
                new_features_df = pd.DataFrame(transformer.predict(X[self.basefeaturenames])).add_prefix(f"{featuretype}_")
            else:
                new_features_df = pd.DataFrame(transformer.transform(X[self.basefeaturenames])).add_prefix(f"{featuretype}_")

            # concate features and new ones
            X = pd.concat([X, new_features_df], axis="columns")

        return X

    def _fit_transformer(self, transformer, featuretype):
        # search the optimal n components
        optimal_n_components, cv_score = _search_optimal_n_components(
            estimator=self.estimator,
            transformer=transformer,
            X_train=self.X_train,
            max_n_components_to_create=self.max_n_components_to_create,
            y_train=self.y_train,
            prefix=transformer.__class__.__name__,
            early_stopping=self.early_stopping,
        )

        # store the cv scores
        self.cv_scores_sample[featuretype] = cv_score

        # set the optimal n components
        # K-Means uses n_clusters instead of n components
        if transformer.__class__.__name__ == 'MiniBatchKMeans':
            transformer.set_params(**{"n_clusters": optimal_n_components})

        else:
            transformer.set_params(**{"n_components": optimal_n_components})

        # fit the transformer
        transformer.fit(self.X_train, self.y_train)

        return transformer


if __name__ == '__main__':
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.datasets import load_wine

    # custom toy data
    # X, y = make_classification(n_samples=50000, n_features=20, n_informative=3, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=42)
    #
    # # make pandas objects for X and y
    # X = pd.DataFrame(X)
    # X.columns = [str(i) for i in range(20)]
    # y = pd.Series(y)

    # breast cancer dataset
    # X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # wine dataset
    X, y = load_wine(return_X_y=True, as_frame=True)

    # classifiers
    lr = LogisticRegression(n_jobs=-1, random_state=42)
    nb = GaussianNB()
    svc = SVC(random_state=42)
    rf = RandomForestClassifier(n_jobs=-1, random_state=42)

    estimator = nb

    dra_feature = DraFeature(estimator=estimator)
    dra_feature.print_plan()
    dra_feature.fit(X, y)
    X_trans = dra_feature.transform(X, y)

    # print shape of X and X_trans
    print(f"X shape: {X.shape}")
    print(f"X_trans shape: {X_trans.shape}")

    # compare cv scores with and without the new features
    cv_score_baseline = round(cross_val_score(estimator, X, y, cv=5, n_jobs=-1).mean(), 5)
    cv_score_baseline_and_new_features = round(cross_val_score(estimator, X_trans, y, cv=5, n_jobs=-1).mean(), 5)

    print()
    print("#"*80)
    print(f"baseline cv: {cv_score_baseline}")
    print(f"baseline and new features cv: {cv_score_baseline_and_new_features}")
    print("done")


