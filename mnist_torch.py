from pathlib import Path

import matplotlib.pyplot as pyplot
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader


def main() -> None:
    # Affiche les versions des bibliotheques utilisees pour ce projet.
    print("PyTorch :", torch.__version__)
    print("torchvision :", torchvision.__version__)

    # Cree une transformation qui convertit chaque image en tenseur PyTorch.
    # Les pixels sont convertis en nombres decimaux compris entre 0 et 1.
    transform = torchvision.transforms.ToTensor()

    # Le dossier data est place a cote de ce script, quel que soit le dossier
    # depuis lequel la commande Python est lancee.
    data_directory = Path(__file__).resolve().parent / "data"

    # Telecharge et charge le jeu de donnees MNIST pour l'entrainement.
    # Le parametre train=True selectionne les 60 000 images d'entrainement.
    # Le parametre download telecharge les donnees si elles sont absentes.
    # Le parametre transform applique la conversion en tenseur a chaque image.
    train_data = torchvision.datasets.MNIST(
        root=str(data_directory),
        train=True,
        download=True,
        transform=transform,
    )

    # Telecharge et charge le jeu de donnees MNIST pour le test.
    # train=False selectionne les 10 000 images reservees a l'evaluation.
    test_data = torchvision.datasets.MNIST(
        root=str(data_directory),
        train=False,
        download=True,
        transform=transform,
    )

    # Affiche le nombre d'images dans les jeux d'entrainement et de test.
    print(f"Images d'entrainement : {len(train_data)}")
    print(f"Images de test : {len(test_data)}")

    # Recupere le premier exemple d'entrainement : image et chiffre attendu.
    image, label = train_data[0]
    print(f"Forme de l'image : {image.shape}")
    print(f"Chiffre attendu : {label}")

    # Affiche un exemple d'image en niveaux de gris.
    pyplot.imshow(image.squeeze(), cmap="gray")
    pyplot.title(f"Chiffre : {label}")
    pyplot.axis("off")
    pyplot.show()

    # Prepare les donnees en lots de 64 images.
    # shuffle=True melange les donnees d'entrainement a chaque epoque.
    batch_size = 64
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    # Les donnees de test ne sont pas melangees, ce qui conserve leur ordre.
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    # Recupere un premier lot pour verifier la forme des images et etiquettes.
    images, labels = next(iter(train_loader))
    print("Forme des images :", images.shape)
    print("Forme des etiquettes :", labels.shape)
    print("Quelques etiquettes :", labels[:10])

    # Cree un reseau entierement connecte qui traite les images aplaties.
    # Chaque image de 28 x 28 pixels donne 784 valeurs d'entree.
    # La couche cachee de 128 neurones apprend des motifs utiles.
    # Les 10 sorties correspondent aux chiffres possibles de 0 a 9.
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    # Fait passer un lot dans le modele et affiche la forme des scores.
    scores = model(images)
    print("Forme des scores :", scores.shape)
    print("Scores de la premiere image :", scores[0])

    # La fonction de perte compare les 10 scores aux bonnes etiquettes.
    # L'optimiseur SGD ajuste les parametres pour reduire cette perte.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Calcule une premiere perte sans encore modifier le modele.
    scores = model(images)
    loss = criterion(scores, labels)
    print("Perte initiale :", loss.item())

    # Effectue une mise a jour des parametres sur un lot d'images.
    # Les gradients precedents sont effaces avant le nouveau calcul.
    optimizer.zero_grad()
    scores = model(images)
    loss_before = criterion(scores, labels)

    # backward calcule les gradients et step ajuste les parametres du modele.
    loss_before.backward()
    optimizer.step()

    # Mesure la perte apres la mise a jour sans calculer de gradients.
    with torch.no_grad():
        scores_after = model(images)
        loss_after = criterion(scores_after, labels)

    print(f"Perte avant la mise a jour : {loss_before.item():.4f}")
    print(f"Perte apres la mise a jour : {loss_after.item():.4f}")

    # Entraine le modele sur tous les lots d'entrainement pendant une epoque.
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        scores = model(images)
        loss = criterion(scores, labels)

        loss.backward()
        optimizer.step()

        # La perte du lot est multipliee par sa taille pour calculer une
        # moyenne correcte, y compris si le dernier lot est plus petit.
        total_loss += loss.item() * labels.size(0)

        # argmax choisit le chiffre ayant le score le plus eleve.
        predictions = scores.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_images += labels.size(0)

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images
    print(f"Perte moyenne d'entrainement : {average_loss:.4f}")
    print(f"Exactitude pendant l'entrainement : {accuracy:.1%}")

    # Evalue le modele sur les images de test, sans mettre a jour ses poids.
    model.eval()
    test_loss = 0.0
    total_correct = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in test_loader:
            scores = model(images)
            loss = criterion(scores, labels)

            test_loss += loss.item() * labels.size(0)
            predictions = scores.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_images += labels.size(0)

    average_test_loss = test_loss / total_images
    test_accuracy = total_correct / total_images
    print(f"Perte moyenne sur les donnees de test : {average_test_loss:.4f}")
    print(f"Exactitude sur les donnees de test : {test_accuracy:.1%}")

    # Recherche jusqu'a huit images de test mal classees.
    wrong_images = []
    wrong_labels = []
    wrong_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            scores = model(images)
            predictions = scores.argmax(dim=1)
            wrong_indices = (predictions != labels).nonzero(as_tuple=True)[0]

            for index in wrong_indices.tolist()[: 8 - len(wrong_images)]:
                wrong_images.append(images[index].squeeze().cpu())
                wrong_labels.append(labels[index].item())
                wrong_predictions.append(predictions[index].item())

            if len(wrong_images) == 8:
                break

    # Affiche les erreurs pour comparer les chiffres attendus aux predictions.
    if wrong_images:
        figure, axes = pyplot.subplots(2, 4, figsize=(10, 5))

        for axis, image, expected, predicted in zip(
            axes.flat, wrong_images, wrong_labels, wrong_predictions
        ):
            axis.imshow(image, cmap="gray")
            axis.set_title(f"Reponse : {expected} | Predit : {predicted}")
            axis.axis("off")

        pyplot.tight_layout()
        pyplot.show()
    else:
        print("Aucune erreur trouvee.")

    # Cree une matrice de confusion : lignes = chiffres reels,
    # colonnes = chiffres predits.
    confusion = torch.zeros(10, 10, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in test_loader:
            scores = model(images)
            predictions = scores.argmax(dim=1)

            for expected, predicted in zip(labels, predictions):
                confusion[expected, predicted] += 1

    # Affiche les comptes de chaque paire chiffre reel / chiffre predit.
    pyplot.figure(figsize=(8, 6))
    pyplot.imshow(confusion, cmap="Blues")
    pyplot.colorbar()
    pyplot.xlabel("Chiffre predit")
    pyplot.ylabel("Vrai chiffre")
    pyplot.title("Matrice de confusion sur les donnees de test")

    for row in range(10):
        for column in range(10):
            pyplot.text(
                column,
                row,
                int(confusion[row, column]),
                ha="center",
                va="center",
            )

    pyplot.xticks(range(10))
    pyplot.yticks(range(10))
    pyplot.show()

    # Entraine le modele pendant trois epoques supplementaires.
    # Chaque epoque est suivie d'une evaluation sur le jeu de test.
    for epoch in range(1, 4):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            scores = model(images)
            loss = criterion(scores, labels)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (scores.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        average_train_loss = train_loss_sum / train_total
        train_accuracy = train_correct / train_total

        # Evalue chaque epoque sans calcul de gradients ni mise a jour des poids.
        model.eval()
        test_loss_sum = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                scores = model(images)
                loss = criterion(scores, labels)

                test_loss_sum += loss.item() * labels.size(0)
                test_correct += (scores.argmax(dim=1) == labels).sum().item()
                test_total += labels.size(0)

        average_test_loss = test_loss_sum / test_total
        test_accuracy = test_correct / test_total

        print(
            f"Epoque {epoch} : "
            f"perte entrainement = {average_train_loss:.4f}, "
            f"exactitude entrainement = {train_accuracy:.1%}, "
            f"perte test = {average_test_loss:.4f}, "
            f"exactitude test = {test_accuracy:.1%}"
        )


if __name__ == "__main__":
    main()
