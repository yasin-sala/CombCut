import numpy as np
import graphlearning as gl

# Number of independent trials
n_runs = 20
rates = [1, 2, 3, 4, 5]

# Load cifar10 labels once
labels = gl.datasets.load('cifar10', labels_only=True)


# Pre‐build the graph and class priors (they don’t change across runs)
W = gl.weightmatrix.knn('cifar10', 10, metric='simclr', kernel='gaussian')
class_priors = gl.utils.class_priors(labels)

#Define all your models
models = [
   gl.ssl.laplace(W),
   gl.ssl.poisson(W),
   gl.ssl.plaplace(W, p=3),
   gl.ssl.amle(W),
   gl.ssl.volume_mbo(W, class_priors),
   gl.ssl.stiefel_ssl(W),
   gl.ssl.CombCutSSL(W, class_priors=class_priors),
]

# Prepare storage for accuracies
acc_records = {model.name: [] for model in models}

# store results per rate
results = {}  

for rate in rates:
    print(f"\n===== Rate = {rate} labels per class =====")

    # fresh accuracy storage per rate
    acc_records = {model.name: [] for model in models}

    # Repeat experiment n_runs times
    for run in range(n_runs):
        train_ind = gl.trainsets.generate(labels, rate=rate)
        train_labels = labels[train_ind]

        for model in models:
            pred = model.fit_predict(train_ind, train_labels)
            acc = gl.ssl.ssl_accuracy(pred, labels, train_ind)
            acc_records[model.name].append(acc)

    # Compute mean ± std
    results[rate] = {}
    for name, accs in acc_records.items():
        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)

        results[rate][name] = (mean_acc, std_acc)

        print(f"{name}: {mean_acc:.2f}% ± {std_acc:.2f}%")