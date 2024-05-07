# from .gui import MainWindow
# from .analysis_gui import VideoAnalysis
from pymodal_surgical.apps.video_analysis.gui import MainWindow
from pymodal_surgical.apps.video_analysis.analysis_gui import VideoAnalysis
# from .video_analysis import VideoAnalyzer
import sys
from PySide6.QtWidgets import QApplication, QStackedWidget
from pathlib import Path

def main():
    app = QApplication(sys.argv)
    widget = QStackedWidget()

    
    secondpage = VideoAnalysis()
    widget.addWidget(secondpage)
    firstpage = MainWindow(stacked_widget=widget, analyse_window=secondpage)
    widget.addWidget(firstpage)
    

    widget.setCurrentWidget(firstpage)
    widget.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()