# Apprendre le deep learning avec MNIST

Ce projet est un parcours d'apprentissage du deep learning avec **PyTorch** et le jeu de données **MNIST**. Le travail est présenté progressivement dans un notebook Jupyter et dans un script Python autonome.

## Objectifs

- Charger et explorer les images de chiffres manuscrits.
- Préparer les données avec des tenseurs PyTorch et des lots (*batches*).
- Construire un premier réseau de neurones entièrement connecté (MLP).
- Comprendre la fonction de perte, les gradients et la mise à jour des paramètres.
- Entraîner le modèle et mesurer ses performances sur des données de test.
- Visualiser certaines erreurs de classification et une matrice de confusion.

## Contenu actuel

Le notebook [`mnist_torch.ipynb`](./mnist_torch.ipynb) contient :

1. Le chargement des 60 000 images d'entraînement et des 10 000 images de test de MNIST.
2. La conversion des images en tenseurs et l'affichage d'exemples.
3. La préparation des données en lots de 64 images.
4. Un MLP avec l'architecture **784 → 128 → 10** :
   - 784 valeurs d'entrée, soit les pixels d'une image de 28 × 28 ;
   - une couche cachée de 128 neurones avec l'activation ReLU ;
   - 10 sorties, une pour chaque chiffre de 0 à 9.
5. L'entraînement avec `CrossEntropyLoss` et l'optimiseur SGD.
6. L'évaluation sur le jeu de test, l'affichage de prédictions incorrectes et la matrice de confusion.
7. Trois époques d'entraînement supplémentaires.

Le script [`mnist_torch.py`](./mnist_torch.py) rassemble ces étapes dans un programme exécutable de bout en bout. Il charge les données, entraîne le MLP, évalue ses résultats, affiche les erreurs et la matrice de confusion, puis effectue trois époques supplémentaires.

## Résultat obtenu

Après ces trois époques supplémentaires, le MLP a obtenu **96,3 % d'exactitude sur le jeu de test**. Ce résultat correspond à l'exécution actuelle du notebook et peut varier légèrement selon la configuration et l'aléatoire de l'entraînement.

## Prérequis

- Python 3
- PyTorch, Torchvision et Matplotlib
- Jupyter Notebook ou Visual Studio Code avec l'extension Jupyter (pour le notebook)

Installation des dépendances :

```bash
python -m pip install torch torchvision matplotlib jupyter
```

## Exécution

Pour exécuter le script :

```bash
python mnist_torch.py
```

Pour suivre les étapes dans le notebook :

1. Ouvrir `mnist_torch.ipynb` dans Jupyter ou Visual Studio Code.
2. Sélectionner un environnement Python où les dépendances sont installées.
3. Exécuter les cellules dans l'ordre, du début à la fin.

Au premier chargement, Torchvision télécharge MNIST dans le dossier `data/`.

## Suite du parcours

La prochaine étape prévue est de construire et d'entraîner un réseau convolutif (CNN), puis de comparer ses performances à celles du MLP. Le CNN n'est pas encore inclus dans le travail terminé décrit ci-dessus.
