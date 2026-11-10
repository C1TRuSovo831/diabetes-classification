import numpy as np
from scipy.special import logsumexp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from .config import BINARY, CATEGORICAL, CONTINUOUS
from .preprocessing import make_encoder


class MixedNaiveBayes:
    """连续特征使用高斯密度，离散特征使用加一平滑的条件频数。"""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.distributions_ = []
        discrete = BINARY + CATEGORICAL
        # 预留一个未见类别；平滑后，已见与未见类别的概率之和为 1。
        vocabulary_sizes = X[discrete].nunique() + 1
        for label in self.classes_:
            subset = X.loc[y == label]
            values = subset[CONTINUOUS].to_numpy(dtype=float)
            variances = np.maximum(values.var(axis=0), 1e-9)
            tables = {}
            for column in discrete:
                denominator = len(subset) + vocabulary_sizes[column]
                counts = subset[column].value_counts()
                tables[column] = (np.log((counts + 1) / denominator), -np.log(denominator))
            self.distributions_.append((
                np.log(len(subset) / len(X)), values.mean(axis=0), variances, tables
            ))
        return self

    def predict_proba(self, X):
        values = X[CONTINUOUS].to_numpy(dtype=float)
        scores = []
        for prior, means, variances, tables in self.distributions_:
            log_density = -0.5 * (np.log(2 * np.pi * variances) + (values - means)**2 / variances)
            score = prior + log_density.sum(axis=1)
            for column, (log_prob, unseen) in tables.items():
                score += X[column].map(log_prob).fillna(unseen).to_numpy(dtype=float)
            scores.append(score)
        scores = np.column_stack(scores)
        return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))


def make_knn(k):
    return make_pipeline(make_encoder(), KNeighborsClassifier(
        n_neighbors=k, weights="uniform", algorithm="kd_tree", metric="euclidean"
    ))


def make_lda():
    return make_pipeline(make_encoder(), LinearDiscriminantAnalysis(solver="svd"))
