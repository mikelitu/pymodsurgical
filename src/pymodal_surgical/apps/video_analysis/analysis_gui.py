import sys
from PySide6.QtWidgets import QApplication, QLabel, QWidgetAction, QDialog
from PySide6.QtWidgets import QVBoxLayout, QWidget, QMdiSubWindow, QMainWindow, QToolBar
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt, QSize
from PySide6.QtMultimedia import QMediaPlayer
from pymodal_surgical.apps.video_analysis.video_analysis import VideoAnalyzer
from pymodal_surgical.apps.utils import ModeShapeCalculator
from pathlib import Path

app_dir = Path(__file__).parent

class VideoAnalysis(QMainWindow):
    def __init__(self, video_config: dict):
        super().__init__()
        self.setWindowTitle("Video Analysis")
        self.setGeometry(100, 100, 800, 600)
        
        # self.video_analyzer = VideoAnalyzer(video_path, config=video_config)

        tool_bar = QToolBar()
        self.addToolBar(tool_bar)

        flow_menu = self.menuBar().addMenu("Flow")
        flow_icon = QIcon().addFile(str(app_dir/"icons/flow.png"))
        flow_action = QAction(flow_icon, "Flow...", self, triggered=self.show_flow)
        flow_menu.addAction(flow_action)
        tool_bar.addAction(flow_action)

        force_menu = self.menuBar().addMenu("Force")
        force_icon = QIcon().addFile(str(app_dir/"icons/force.png"), size=QSize(16, 16))
        force_action = QAction(force_icon, "Force...", self, triggered=self.show_force_config)
        force_config_action = tool_bar.addAction(force_icon, "Force config")
        force_config_action.triggered.connect(self.show_force_config)
        force_menu.addAction(force_action)
        tool_bar.addAction(force_action)

        self.layout = QVBoxLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        

        self.loading_window = QMdiSubWindow()
        self.loading_window.setWidget(QLabel("Loading..."))
        self.layout.addWidget(self.loading_window)
        self.loading_window.setVisible(True)


    def init(self, config):

        self.video_analyzer = ModeShapeCalculator(config)
        self.loading_window.setVisible(False)
        
    def show_flow(self):
        print("You pressed flow!")

    def show_force_config(self):
        print("You pressed force!")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_config = {
        "start": 0,
        "end": 0,
        "video_type": "mono",
        "fps": 20.0,
        "video_path": "videos/heart_beating.mp4",
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
    window = VideoAnalysis(video_config)
    window.show()
    sys.exit(app.exec())