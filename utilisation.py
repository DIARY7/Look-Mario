from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Chargement le modèle pré-entraîné
model = YOLO('runs/detect/train/weights/best.pt')

image = cv2.imread('test/frame_0304.jpg')
resultats = model(image)

# Générer des couleurs distinctes pour chaque classe
# Déterminer le nombre de classes dans le modèle
nb_classes = len(model.names)
# Générer des couleurs aléatoires distinctes (BGR pour OpenCV)
np.random.seed(42)  # Pour une cohérence des couleurs entre les exécutions
colors = np.random.randint(0, 255, size=(nb_classes, 3), dtype=np.uint8).tolist()

# Afficher les résultats
for r in resultats:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Classe et confiance
        classe = int(box.cls[0])
        nom_classe = model.names[classe]
        confiance = float(box.conf[0])

        print(f"Détecté: {nom_classe} avec confiance {confiance:.2f} à la position {x1, y1, x2, y2}")

        # Sélectionner la couleur pour cette classe
        color = colors[classe]

        # Dessiner la boîte sur l'image avec la couleur spécifique à la classe
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Ajouter un fond pour le texte pour améliorer la lisibilité
        text = f"{nom_classe} {confiance:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)

        # Ajouter le texte en blanc pour contraste
        cv2.putText(image, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Sauvegarder ou afficher l'image avec les détections
cv2.imwrite('resultat_detection.jpg', image)
cv2.imshow('Détections', image)
cv2.waitKey(0)