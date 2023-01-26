import argparse
from multiprocessing import cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint, truncnorm, uniform
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

N_COMPONENTS_90 = 86


def get_tuned_params(
    X: np.ndarray,
    y: np.ndarray,
    estimator,
    random_state: int,
    distributions: dict,
    n_cv: int = 5,
):
    """RandomizedSearchCVでチューニングされたパラメータを取得する
    Args:
        X:
    """
    clf = RandomizedSearchCV(
        estimator=estimator,
        random_state=random_state,
        cv=n_cv,
        n_jobs=(cpu_count() - 1),
        param_distributions=distributions,
        verbose=3,
        scoring="accuracy",
    )
    clf.fit(X, y)
    return clf.best_params_


def get_final_pred(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return pred


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--test_size", default=0.2, type=float)
    parser.add_argument("--n_cv_for_tune", default=5, type=int)
    parser.add_argument("--dst_dir", default="./dst")
    parser.add_argument("--apply_pca", action="store_true")
    return parser.parse_args()


def prepare_dataset(seed: int = 0, test_size: float = 0.2, apply_pca: bool = True):
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
    )
    X = preprocess(X)
    if apply_pca:
        # 学習用データで事前に計算しておいたn_components
        print(f"PCA apply. feature shape change {X.shape[-1]} => {N_COMPONENTS_90}.")
        X = PCA(n_components=N_COMPONENTS_90).fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def preprocess(X: np.ndarray):
    return X / 255


def main():
    args = get_args()
    print("start prepare data...")
    X_train, X_test, y_train, y_test = prepare_dataset(
        args.seed, args.test_size, apply_pca=args.apply_pca
    )
    print("end prepare data...")
    # RandomForestClassifierのチューニング
    print("finding RandomForestClassifier params...")
    rfc_distributions = dict(
        max_depth=randint(2, 20),
        max_features=truncnorm(a=0, b=1, loc=0.25, scale=0.1),
        min_samples_split=randint(2, 50),
    )
    rfc = RandomForestClassifier(random_state=args.seed)
    rfc_params = get_tuned_params(
        X_train,
        y_train,
        rfc,
        args.seed,
        rfc_distributions,
        args.n_cv_for_tune,
    )
    print("finding RandomForestClassifier params end")
    # svcのチューニング
    print("finding LogisticRegression params...")
    lr_distributions = dict(C=uniform(0.1, 10))
    lr = LogisticRegression(max_iter=50000)
    lr_params = get_tuned_params(
        X_train,
        y_train,
        lr,
        args.seed,
        lr_distributions,
        args.n_cv_for_tune,
    )
    print("finding SVC params end")
    # KNeighborsClassifierのチューニング
    print("finding KNeighborsClassifier params...")
    knn_distributions = dict(n_neighbors=randint(2, 20))
    knn = KNeighborsClassifier()
    knn_params = get_tuned_params(
        X_train, y_train, knn, args.seed, knn_distributions, args.n_cv_for_tune
    )
    print("finding KNeighborsClassifier params end")
    # 個別の作成
    print("finalizing each models...")
    estimators = [
        ("random forest", RandomForestClassifier(**rfc_params, random_state=args.seed)),
        ("logistic regression", LogisticRegression(**lr_params, max_iter=50000)),
        ("knn", KNeighborsClassifier(**knn_params)),
    ]
    result = []
    for model_name, model in estimators:
        pred = get_final_pred(model, X_train, y_train, X_test)
        acc = accuracy_score(y_true=y_test, y_pred=pred)
        result.append({"model": model_name, "acc": acc})
    # voting classifierの学習
    voting_clf = VotingClassifier(estimators=estimators, voting="soft")
    pred = get_final_pred(voting_clf, X_train, y_train, X_test)
    result.append(
        {"model": "voting", "acc": accuracy_score(y_true=y_test, y_pred=pred)}
    )
    # 結果の保存
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(result)
    result.to_csv(dst_dir / "result.csv", index=False)
    sns.barplot(data=result, x="model", y="acc")
    plt.savefig(dst_dir / "model_and_acc.png")


if __name__ == "__main__":
    main()
