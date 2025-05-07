from ultralytics import YOLO
import cv2
import numpy as np
import time
import mss
import pygetwindow as gw  # Pour obtenir les fenêtres actives


def capture_game_screen(monitor=None):
    """Capture l'écran de jeu ou une fenêtre spécifique"""
    with mss.mss() as sct:
        if monitor is None:
            # Capture l'écran entier par défaut
            monitor = sct.monitors[1]  # Le moniteur principal

        screenshot = sct.grab(monitor)
        # Convertir en format numpy pour OpenCV
        img = np.array(screenshot)
        # Convertir de BGRA à BGR (supprimer le canal alpha)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def get_game_window(game_title):
    """Obtient les dimensions d'une fenêtre de jeu spécifique"""
    try:
        window = gw.getWindowsWithTitle(game_title)[0]
        window.activate()  # Mettre la fenêtre au premier plan

        # Obtenir les coordonnées de la fenêtre
        left, top = window.left, window.top
        width, height = window.width, window.height

        # Créer le moniteur pour mss
        monitor = {"top": top, "left": left, "width": width, "height": height}
        return monitor
    except IndexError:
        print(f"Fenêtre de jeu '{game_title}' non trouvée. Utilisation de l'écran complet.")
        return None


def main():
    # Charger le modèle YOLO entraîné
    model = YOLO('runs/detect/train/weights/best.pt')

    # Paramètres
    game_title = "yuzu"  # Remplacez par le nom exact de la fenêtre de votre jeu
    fps_target = 30  # Fréquence cible des détections

    # Générer des couleurs pour chaque classe
    nb_classes = len(model.names)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(nb_classes, 3), dtype=np.uint8).tolist()

    # Obtenir la fenêtre du jeu
    game_monitor = get_game_window(game_title)

    # Boucle principale pour la détection en temps réel
    overlay = None  # Initialisation de la fenêtre d'overlay

    print("Détection en cours. Appuyez sur 'q' pour quitter.")

    try:
        while True:
            start_time = time.time()

            # Capturer l'écran du jeu
            game_screen = capture_game_screen(game_monitor)
            height, width = game_screen.shape[:2]

            if overlay is None:
                # Créer la fenêtre d'overlay avec la taille appropriée
                overlay = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.namedWindow('Overlay Détections', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Overlay Détections', cv2.WND_PROP_TOPMOST, 1)  # Toujours au premier plan

            # Faire une copie pour dessiner les détections
            overlay.fill(0)  # Effacer l'overlay précédent

            # Exécuter la détection avec YOLO
            results = model(game_screen)

            # Traiter les résultats et dessiner les boîtes
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    classe = int(box.cls[0])
                    nom_classe = model.names[classe]
                    confiance = float(box.conf[0])

                    # Ne dessiner que si la confiance est suffisante
                    if confiance > 0.5:  # Seuil de confiance ajustable
                        color = colors[classe]

                        # Dessiner sur l'overlay au lieu de l'image originale
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                        # Ajouter le texte
                        text = f"{nom_classe} {confiance:.2f}"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                        cv2.rectangle(overlay, (x1, y1 - text_size[1] - 10),
                                      (x1 + text_size[0], y1), color, -1)
                        cv2.putText(overlay, text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Afficher l'overlay
            cv2.imshow('Overlay Détections', overlay)

            # Calculer le temps d'attente pour maintenir le FPS cible
            elapsed_time = time.time() - start_time
            wait_time = max(1, int((1.0 / fps_target - elapsed_time) * 1000))

            # Vérifier si l'utilisateur veut quitter
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

            # Afficher le FPS actuel
            fps = 1.0 / (time.time() - start_time)
            print(f"FPS: {fps:.2f}", end="\r")

    except KeyboardInterrupt:
        print("\nDétection interrompue par l'utilisateur")

    finally:
        cv2.destroyAllWindows()
        print("Programme terminé")


if __name__ == "__main__":
    main()