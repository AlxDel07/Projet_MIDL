import numpy as np
import urllib.request
import os

TAILLE_COUCHE_INT = 32

def initialiser(taille_cachee=TAILLE_COUCHE_INT, graine=0):
    rng = np.random.default_rng(graine)
    return {
        "W1": rng.normal(0, 0.1, (taille_cachee, 784)),
        "b1": np.zeros(taille_cachee),
        "W2": rng.normal(0, 0.1, (10, taille_cachee)),
        "b2": np.zeros(10),
    }

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

def sigmoide(x):
    return 1/(1+np.exp(-x))


def propagation_avant(x, params):
    z1 = params["W1"] @ x + params["b1"]
    a1 = sigmoide(z1)
    z2 = params["W2"] @ a1 + params["b2"]
    a2 = sigmoide(z2)
    return a2

X_train, y_train, X_test, y_test = charger_mnist()
print(X_train.shape, X_test.shape)   # shape donne le tuple (nblignes,nbcolonnes) du tableau
params = initialiser()
sortie = propagation_avant(X_train[0], params)
print("Argmax : ", np.argmax(sortie),"\n")  # Argmax renvoie l'indice de la valeur la plus grande du tableau
print(sortie)



