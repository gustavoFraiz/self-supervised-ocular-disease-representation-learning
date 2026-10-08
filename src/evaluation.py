"""Feature extraction and downstream classifiers recovered from the TCC notebook."""

import numpy as np
import torch
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def extract_features(encoder, data_loader, device, desc=""):
    """Run a frozen encoder over a labeled data loader and collect latent features."""
    encoder.eval()
    features, labels = [], []

    with torch.no_grad():
        for images, batch_labels in tqdm(data_loader, desc=desc):
            images = images.to(device)
            feature_vectors = encoder(images)
            features.append(feature_vectors.cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def build_svm_search():
    """Recovered RBF-SVM grid used in the notebook."""
    return GridSearchCV(
        SVC(kernel="rbf"),
        {
            "C": [1, 10, 100],
            "gamma": [0.0001, 0.001, "scale"],
        },
        cv=3,
        n_jobs=-1,
    )


def build_knn_search():
    """Recovered KNN neighbor search."""
    return GridSearchCV(
        KNeighborsClassifier(),
        {"n_neighbors": list(range(3, 16, 2))},
        cv=3,
        n_jobs=-1,
    )


def build_random_forest_search():
    """Recovered Random Forest estimator search."""
    return GridSearchCV(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100, 200]},
        cv=3,
        n_jobs=-1,
    )
