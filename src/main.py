import numpy as np
import urllib.request
import os

def charger_mnist():
    fichier = "mnist.npz"
    if not os.path.exists(fichier):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        urllib.request.urlretrieve(url, fichier)
    d = np.load(fichier)
    X_train = d["x_train"].reshape(-1, 784) / 255.0    # On met les valeurs dans un vecteur et on les normalise
    X_test  = d["x_test"].reshape(-1, 784) / 255.0
    y_train, y_test = d["y_train"], d["y_test"]    # On recupère les données (aucun traitement nécessaire)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = charger_mnist()
print(X_train.shape, X_test.shape)   # shape donne le tuple (nblignes,nbcolonnes) du tableau