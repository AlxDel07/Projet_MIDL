# -*- coding: utf-8 -*-
"""
Hello ce code implémente :
    structure de données
    fonctions auxiliaires (cout, sigmoide etc)
    apprentissage
    prédiction
    initialisation
    test
    
Mais il ne marche pas...


Détails de ma compréhension du problème (je crois...) :
    
    La descente de gradient cherche un minimum local de la fonction de coût.
    J'ai l'impression que le réseau à tendance à se stabiliser vers 
    un état où toutes les valeurs de la couche de sortie sont quasi nulles
    ... ce qui permet effectivement de faire baisser la fonction de coût
    vers une moyenne environ égale à 1 (peut être inférieure ou supérieure)
    (Le réseau fait ça en mettant des biais très négatifs notamment)
    Puis l'apprentissage stagne.
    
    Conséquence inquiétante :
        Pour toutes les images, les résultats sont les mêmes !!!
        (aux arrondis de float près)
    
    Piste d'amélioration tombée à l'eau :
        Commencer avec des poids et biais qui donnent [0, 0, ..., 0]
        en sortie pour toute entrée (par ex : mettre tous les biais à -1000)
        
        Ainsi, la fonction de coût moyen est déjà à 1 au départ, 
        puis la descente de gradient permettrait de descendre en dessous
        
        L'expérience donne que cet état semble être un minimum local
        Donc le gradient vaut 0 et l'apprentissage ne mène nulle part
    
    
PS Remarque propreté :
    J'ai mis certaines fonctions dans la classe Reseau
    d'autres en dehors, sans raison particulière... 
    C'est peut-être à uniformiser
    

"""

import math
import numpy as np
import urllib.request
import os
import time

class Reseau :
    pass

class Neurone:
    poids : list[float]
    biais : float
    combi : float
    sortie : float
    reseau : Reseau
    idCouche : int
    idNeurone : int
    
    """
    on garde le résultat intermédiaire combi car on en a besoin
    dans le calcul du gradient !!
    """
    
    def __init__(self, reseau, couche, idNeurone) :
        self.reseau = reseau
        self.idCouche = couche.idCouche
        if self.idCouche>0 : 
            self.poids = np.random.uniform(-1.0, 1.0, self.reseau.couches[self.idCouche-1].nbNeurones).tolist()
        else :
            self.poids = [0.]
        self.biais = 0.0
        self.combi = 0.0
        self.sortie = 0.0
        self.idNeurone = idNeurone
    
    def changer_poids(self, params) :
        self.biais = params[-1]
        self.poids = params[:-1]
    
    def donner_poids(self) :
        params = self.poids + [self.biais]
        return params
        

class Couche:
    nbNeurones : int
    neurones : list[Neurone]
    idCouche : int
    reseau : Reseau
    erreurs : list[float]
    
    def __init__(self, nbNeurones, idCouche, reseau) :
        self.reseau = reseau
        self.nbNeurones = nbNeurones
        self.idCouche = idCouche
        self.neurones = []
        for i in range(nbNeurones) :
            self.neurones.append(Neurone(self.reseau, self, i))
        
    def changer_poids(self, params) :
        """
        remplace les poids et biais de la couche par params
        """
        #n = taille de la couche précédente
        n = self.reseau.couches[self.idCouche - 1].nbNeurones
        for i in range(self.nbNeurones - 1) :
            self.neurones[i].changer_poids(params[i*(n+1):(i+1)*(n+1)])
    
    def donner_poids(self) :
        """
        retourne le vecteur des poids et biais des neurones de la couche
        """
        params = []
        for n in self.neurones :
            params += n.donner_poids()
        return params
    
    def valeurs_actuelles(self) : #TODO: changer de nom
        """
        retourne le veteur des valeurs de sortie des neurones de la couche
        """
        return np.array([self.neurones[n].sortie for n in range(self.nbNeurones)])
        

    
class Reseau:
    nbCouches : int
    couches : list[Couche]
    
    def predire(self, image) :
        """
        prend en paramètre une image
        retourne la couche de sortie (vecteur prédiction)
        """
        #couche d'entrée
        for i in range(len(image)) :
            self.couches[0].neurones[i].combi = image[i]
            self.couches[0].neurones[i].sortie = image[i]
        
        for c in range(1, self.nbCouches) :
            sorties = self.couches[c-1].valeurs_actuelles()
            for n in range(self.couches[c].nbNeurones) :
                neurone = self.neurone(c, n)                
                neurone.combi = neurone.poids @ sorties + neurone.biais
                neurone.sortie = sigmoide(neurone.combi)

        return self.couches[-1].valeurs_actuelles()
    

    def __init__(self, nbNeuronesCouche) :
        self.nbCouches = len(nbNeuronesCouche)
        self.couches = []
        for i in range(len(nbNeuronesCouche)) :
            self.couches.append(Couche(nbNeuronesCouche[i], i, self))

    
    def neurone(self, n, i) :
        """
        retourne le neurone i de la n-ième couche
        """
        return self.couches[n].neurones[i]
    
    def donner_poids(self) :
        """
        retourne un tableau des poids et biais du reseau
        dans l'ordre naturel (càd : poids et biais du neurone 0 de la couche 1 puis 
        du neurone 1 de la couche 1 [...] puis du neurone 9 de la couche de sortie)
        """
        params = []
        for c in self.couches[1:] : #pas de poids pour la couche d'entrée !
            params += c.donner_poids()
        return params
    
    def changer_poids(self, params) :
        """
        remplace les poids et biais actuels du réseau par params
        nb : utilise le même ordre que donner_poids
        """
        for i in range(1, self.nbCouches) :
            n = self.couches[i-1].nbNeurones
            m = self.couches[i].nbNeurones
            self.couches[i].changer_poids(params[(i-1)*(n+1)*m:i*(n+1)*m])
    
    
    """REMARQUE :
        changer_poids() et donner_poids() vont de paire
        elles permettent les conversions entre Reseau et tableau (de poids et biais) -> donner_poids
        et entre tableau et Reseau -> changer_poids
    """
    

    def apprendre(self, ech_images, ech_labels, taux, nbEtapes) :
        """
        descente du gradient !!
        taux : taux d'apprentissage λ 
        (c'est une liste de longueur nbEtapes) -> permet de faire varier λ
        
        laisse une trace de l'évolution des coûts moyens et des temps de calcul dans cout.txt
        """
        f = open('cout.txt', 'w')
        f.write("Evolution de la fonction coût.\n")
        f.close()
        toc = time.time()
        for etape in range(nbEtapes) :
            print("Début de l'étape n°", etape)
            calculer_erreurs(self, ech_images, ech_labels)
            for n in range(1, self.nbCouches) :
                for i in range(self.couches[n].nbNeurones) :
                    neur = self.neurone(n, i)
                    err = self.couches[n].erreurs[i]
                    
                    neur.biais -= taux[etape] * err
                    for j in range(len(neur.poids)) :
                        neur.poids[j] -= taux[etape] * err * self.neurone(n-1, i).sortie
            tac = time.time()
            print("\nFin de l'étape n°", etape)
            f = open('cout.txt', 'a')
            f.write(f"\n\nFin de l'étape n°{etape} à date = {tac-toc}")
            if etape%10==0 or etape<5:
                #la fonction cout est lourde, on ne le calcule pas à chaque fois !
                f.write(f"\nCoût moyen : {cout_moyen(self, ech_images, ech_labels)}")
            f.close()
            
    def annulerErreurs(self) :
        '''
        remet les erreurs à zéro
        '''
        for c in self.couches :
            c.erreurs = np.zeros(c.nbNeurones)
    
    
    
""" ****************************** AUXILIAIRES *************************************"""

def sigmoide(x) :
    try :
        return 1/(1+math.exp(-x))
    except :
        return 0  #sinon il pète un plomb quand x est trop négatif


def cout_unique(reseau : Reseau, image, label) :
    """
    retourne l'erreur (somme des carrés des différences)
    du réseau pour UNE SEULE image
    """
    prediction = reseau.predire(image)  #Array
    ideal = np.array([0 for i in range(len(prediction))])
    ideal[label] = 1.0
    cout = np.sum((prediction - ideal)**2)
    return cout

def cout_moyen(reseau : Reseau, ech_images, ech_labels) :
    """
    retourne l'erreur moyenne sur tout l'échatillon
    """
    couts = np.array([cout_unique(reseau, ech_images[i], ech_labels[i]) for i in range(len(ech_images))])
    return np.mean(couts)


def charger_mnist():
    '''
    charge les données depuis mnist
    '''
    fichier = "mnist.npz"
    if not os.path.exists(fichier):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        urllib.request.urlretrieve(url, fichier)
    d = np.load(fichier)
    X_train = d["x_train"].reshape(-1, 784) / 255.0    # On met les valeurs dans un vecteur et on les normalise
    X_test  = d["x_test"].reshape(-1, 784) / 255.0
    y_train, y_test = d["y_train"], d["y_test"]    # On recupère les données (aucun traitement nécessaire)
    return X_train, y_train, X_test, y_test


def imprimer_image(image) : 
    '''
    Donne un aperçu du chiffre manuscrit dans la console
    '''
    for i in range(len(image)) :
        if i%28==0 :
            print()
        if image[i]<=0.2 :
            print('  ', end='')
        elif image[i]<=0.4 :
            print('░░', end='')
        elif image[i]<=0.6 :
            print('▒▒', end='')
        elif image[i]<=0.8 :
            print('▓▓', end='')
        else :
            print('██', end='')
    print()
    
    
""" ********************** INSTANT GRADIENT *****************************"""

def calculer_erreurs(reseau, ech_images, ech_labels) :
    '''
    calcule les erreurs (càd gradient !) à chaque couche
    voir https://fr.wikipedia.org/wiki/R%C3%A9tropropagation_du_gradient#R%C3%A9tropropagation_de_l'erreur
    pour les formules récursives
    '''
    n_sortie = reseau.nbCouches - 1
    reseau.annulerErreurs()
    for a in range(len(ech_images)) :
        prediction = reseau.predire(ech_images[a])
        attendu = [int(ech_labels[a]==i) for i in range(10)] 
        #1 à la bonne prediction, 0 sinon
        for m in range(1, reseau.nbCouches) :
            #on calcule de la sortie vers l'entrée !
            n = reseau.nbCouches - m #numéro de couche entre 1 et nbCouches
            erreurs = []
            for i in range(reseau.couches[n].nbNeurones) :
                s = sigmoide(reseau.neurone(n, i).combi)
                if n==n_sortie :
                    erreurs.append(2*(prediction[i] - attendu[i]) * s * (1 - s))
                else :
                    somme = sum([reseau.neurone(n+1, j).poids[i] * reseau.couches[n+1].erreurs[j] for j in range(reseau.couches[n+1].nbNeurones)])
                    erreurs.append(s*(1-s)*somme)
            reseau.couches[n].erreurs += np.array(erreurs)
    for n in range(reseau.nbCouches) :
        reseau.couches[n].erreurs /= len(ech_images)
        #pour avoir la moyenne





""" **************************** MAIN **************************"""

#MISE EN PLACE
reseau = Reseau([784, 16, 16, 10]) #modèle de la vidéo
ech_images, ech_labels, test_images, test_labels = charger_mnist()
TAILLE = min(1000, len(ech_labels))  #pour faire des tests rapides mettre TAILLE à 1000, pour des tests poussés mettre 60000
print("Taille de l'échantillon :", TAILLE)
nbEtapes = 5
taux = [10 for _ in range(nbEtapes)]  #liste pour pouvoir faire varier le taux d'apprentissage



#APPRENTISSAGE
tic = time.time()
print("Coût moyen avant apprentissage :", cout_moyen(reseau, ech_images[:TAILLE], ech_labels[:TAILLE]))
tac = time.time()
print("Temps d'éxecution de la fonction de coût :", tac-tic)
reseau.apprendre(ech_images[:TAILLE], ech_labels[:TAILLE], taux, nbEtapes)
print("Coût moyen après apprentissage :", cout_moyen(reseau, ech_images[:TAILLE], ech_labels[:TAILLE]))



#TEST
AFFICHAGE = True  #mettre à False pour plus de lisibilité
nb_ok = 0
nb_erreur = 0
for i in range(len(test_labels)) :
    prediction = np.array(reseau.predire(test_images[i]))
    probable = np.argmax(prediction)
    if probable == test_labels[i]:
        nb_ok +=1
    elif nb_erreur < 6 :
        if AFFICHAGE :
            print("ERREUR")
            imprimer_image(test_images[i])
            print("Prédiction = ", prediction, " ->", probable)
            print("Etiquette = ", test_labels[i])
        nb_erreur +=1
print(f"Taux de prédictions correctes : {nb_ok*100/len(test_labels)}%")

    
    







