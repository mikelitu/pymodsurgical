import sys
from PySide6.QtWidgets import QApplication, QLabel, QWidgetAction
from PySide6.QtWidgets import QVBoxLayout, QWidget, QMdiSubWindow, QMainWindow, QToolBar
from PySide6.QtGui import QIcon, QAction
from PySide6.QtMultimedia import QMediaPlayer
from video_analysis import VideoAnalyzer
from pymodal_surgical.apps.utils import ModeShapeCalculator

class VideoAnalysis(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Analysis")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        # self.video_analyzer = VideoAnalyzer(video_path, config=video_config)

        menu = QToolBar()
        self.addToolBar(menu)
        flow_action = QAction(QIcon("icons/flow.png"), "&Flow", self)
        flow_action.triggered.connect(self.show_flow)
        menu.addAction(flow_action)

        self.loading_window = QMdiSubWindow()
        self.loading_window.setWidget(QLabel("Loading..."))
        self.layout.addWidget(self.loading_window)
        self.loading_window.setVisible(True)


    def init(self, config):

        self.video_analyzer = ModeShapeCalculator(config)
        self.loading_window.setVisible(False)
        
    def show_flow(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_config = {
        "start": 0,
        "end": 0,
        "video_config": {
            "video_type": "mono"
        },
        "masking": {
            "enabled": False,
            "mask": "videos/mask/heart_beat.png"
        },
        "filtering": {
            "enabled": True,
            "size": 11,
            "sigma": 1.5
        }
    }
    window = VideoAnalysis("videos/heart_beating.mp4", video_config)
    window.show()
    sys.exit(app.exec())