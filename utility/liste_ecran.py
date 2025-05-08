import pygetwindow as gw

# Liste toutes les fenêtres visibles
windows = gw.getAllTitles()

# Affiche seulement les titres non vides
for title in windows:
    if title.strip():  # Évite les titres vides
        print(title)
