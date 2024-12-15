import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

if not os.path.exists("imagens"):
    os.makedirs("imagens")

data = pd.read_csv('breastcancer.csv')

# Separar atributos (X) e rótulos (y)
X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values   

# Funções para normalização
def normalize_train(X_train):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    return (X_train - mean) / std, mean, std

def normalize_test(X_test, mean, std):
    return (X_test - mean) / std

# cálculo da média
def calculate_mean(X):
    return np.sum(X, axis=0) / len(X)

# cálculo da variância
def calculate_variance(X):
    mean = calculate_mean(X)
    return np.sum((X - mean) ** 2, axis=0) / len(X)

# cálculo da matriz de covariância
def calculate_covariance(X):
    mean = calculate_mean(X)
    n_samples = X.shape[0]
    return (X - mean).T @ (X - mean) / n_samples

class GaussianDiscriminantAnalysis:
    def __init__(self):
        self.mu = {}
        self.sigma = {}
        self.priors = {}

    def fit(self, X, y):
        classes = np.unique(y)
        for c in classes:
            X_c = X[y == c]
            self.mu[c] = calculate_mean(X_c)
            self.sigma[c] = calculate_covariance(X_c)
            self.priors[c] = len(X_c) / len(y)

    def predict(self, X):
        predictions = []
        for x in X:
            scores = {}
            for c in self.mu:
                mu_k = self.mu[c]
                sigma_k = self.sigma[c]
                prior = self.priors[c]
                det_sigma_k = np.linalg.det(sigma_k)
                inv_sigma_k = np.linalg.inv(sigma_k)
                term1 = np.log(prior)
                term2 = -0.5 * np.log(det_sigma_k)
                diff = x - mu_k
                term3 = -0.5 * diff.T @ inv_sigma_k @ diff
                scores[c] = term1 + term2 + term3
            predictions.append(max(scores, key=scores.get))
        return np.array(predictions)

class GaussianNaiveBayes:
    def __init__(self):
        self.mu = {}
        self.sigma = {}
        self.priors = {}

    def fit(self, X, y):
        classes = np.unique(y)
        for c in classes:
            X_c = X[y == c]
            self.mu[c] = calculate_mean(X_c)
            self.sigma[c] = calculate_variance(X_c)
            self.priors[c] = len(X_c) / len(y)

    def predict(self, X):
        predictions = []
        for x in X:
            scores = {}
            for c in self.mu:
                prior = np.log(self.priors[c])
                likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.sigma[c]))
                likelihood -= 0.5 * np.sum(((x - self.mu[c]) ** 2) / self.sigma[c])
                scores[c] = prior + likelihood
            predictions.append(max(scores, key=scores.get))
        return np.array(predictions)

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.array([self.euclidean_distance(x, x_train) for x_train in self.X_train])
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            unique, counts = np.unique(k_labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])
        return np.array(predictions)

# Validação cruzada com 10 folds
def cross_validate(model, X, y):
    n_samples = len(X)
    fold_size = n_samples // 10
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    accuracies, precisions, recalls, f1_scores = [], [], [], []

    for i in range(10):
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, test_indices)

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        X_train, mean, std = normalize_train(X_train)
        X_test = normalize_test(X_test, mean, std)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        tn = np.sum((y_pred == 0) & (y_test == 0))

        accuracy = (tp + tn) / len(y_test)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": (np.mean(accuracies), np.std(accuracies)),
        "precision": (np.mean(precisions), np.std(precisions)),
        "recall": (np.mean(recalls), np.std(recalls)),
        "f1_score": (np.mean(f1_scores), np.std(f1_scores))
    }

models = {
    "Gaussian Discriminant Analysis": GaussianDiscriminantAnalysis(),
    "Gaussian Naive Bayes": GaussianNaiveBayes(),
    "KNN (k=3)": KNN(k=3)
}

results = {}
metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]

for name, model in models.items():
    results[name] = cross_validate(model, X, y)

for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    for name in models.keys():
        mean, std = results[name][metric]
        plt.bar(name, mean, yerr=std, capsize=5, label=name)
    plt.title(f"Comparação de {metric.capitalize()} entre Modelos")
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"imagens/{metric}.png")
    plt.close()

for name, metrics in results.items():
    print(f"Resultados para {name}:")
    for metric, values in metrics.items():
        print(f"  {metric.capitalize()}: Média = {values[0]:.4f}, Desvio Padrão = {values[1]:.4f}")
    print()
