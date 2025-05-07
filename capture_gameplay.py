import sys
import time
import numpy as np
import cv2
import mss
from ultralytics import YOLO
import pygetwindow as gw

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
from PyQt5.QtCore import Qt, QTimer


class OverlayWindow(QMainWindow):
    def __init__(self, model, game_title, fps=30):
        super().__init__()
        self.model = model
        self.fps = fps
        self.game_title = game_title
        self.monitor = self.get_game_window()
        self.boxes = []
        self.colors = [
            QColor(255, 165, 12, 180),
            QColor(0, 255, 0, 180)  
        ]
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // fps)

    def init_ui(self):
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.X11BypassWindowManagerHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        if self.monitor:
            self.setGeometry(
                self.monitor['left'], self.monitor['top'],
                self.monitor['width'], self.monitor['height']
            )
        self.show()

    def get_game_window(self):
        try:
            window = gw.getWindowsWithTitle(self.game_title)[0]
            return {
                'top': window.top,
                'left': window.left,
                'width': window.width,
                'height': window.height
            }
        except IndexError:
            print(f"Fenêtre '{self.game_title}' non trouvée.")
            return None

    def update_frame(self):
        if not self.monitor:
            return

        with mss.mss() as sct:
            img = np.array(sct.grab(self.monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = self.model(frame)
        self.boxes = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf > 0.5:
                    self.boxes.append({
                        'rect': (x1, y1, x2, y2),
                        'label': f"{self.model.names[cls]} {conf:.2f}",
                        'classe':cls
                    })

        self.repaint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont('Arial', 15))

        for box in self.boxes:
            x1, y1, x2, y2 = box['rect']
            pen = QPen(self.colors[box['classe']], 3)
            painter.setPen(pen)

            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawText(x1 + 5, y1 - 5, box['label'])


if __name__ == '__main__':
    game_title = "yuzu 1734 | Super Smash Bros. Ultimate (64-bit) | 13.0.2 | NVIDIA"
    model_path = "runs/detect/train/weights/best.pt"

    model = YOLO(model_path)

    app = QApplication(sys.argv)
    overlay = OverlayWindow(model, game_title)
    sys.exit(app.exec_())
