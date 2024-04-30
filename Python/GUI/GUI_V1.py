import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Your Software")
        self.setGeometry(100, 100, 640, 480)

        # Create a video widget
        self.video_widget = QVideoWidget(self)
        self.setCentralWidget(self.video_widget)

        # Create a media player object
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        # Create a label to display instructions
        self.label = QLabel("Loading...", self)
        self.label.setGeometry(10, 10, 300, 20)

        # Automatically play the video when the GUI opens
        self.play_video()

        # Create buttons
        self.apply_button = QPushButton("Apply Mathematical Framework", self)
        self.apply_button.setGeometry(10, 50, 200, 30)
        # Connect the apply_button to your method for applying the mathematical framework
        # self.apply_button.clicked.connect(self.apply_mathematical_framework)

    def play_video(self):
        media_content = QMediaContent(QUrl.fromLocalFile('/Users/oliversherwood/Documents/CODE/MEIDAS_MAIN/GUI/DySCO_opener.mp4'))
        self.media_player.setMedia(media_content)
        self.media_player.play()
        self.label.setText("Playing video...")

    def apply_mathematical_framework(self):
        # Your code to apply the mathematical framework
        pass

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
