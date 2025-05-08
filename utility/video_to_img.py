import cv2
import os

# Chemin vers la vidéo
video_path = "./captures/video/mario_video.mkv"
# Dossier où stocker les images extraites
output_dir = "./captures/images"
os.makedirs(output_dir, exist_ok=True)

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)

# Récupérer le nombre de frames par seconde (fps) de la vidéo
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculer tous les combien de frames on doit capturer (0.25 sec)
frame_interval = int(fps * 0.25)

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # Enregistrer l'image
        filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Image enregistrée : {filename}")
        saved_count += 1

    frame_count += 1

cap.release()
print("✅ Terminé !")
