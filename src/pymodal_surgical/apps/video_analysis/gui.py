import sys
from PySide6.QtCore import QStandardPaths, Qt, Slot, QUrl
from PySide6.QtGui import QAction, QIcon, QKeySequence, QImageReader
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog,
                               QMainWindow, QSlider, QStyle, QToolBar, QVBoxLayout, QWidget, QLabel, QCheckBox,
                               QInputDialog, QLineEdit, QPushButton, QMessageBox, QSpinBox, QDoubleSpinBox)
from PySide6.QtMultimedia import (QAudioOutput, QMediaFormat,
                                  QMediaPlayer)
from PySide6.QtMultimediaWidgets import QVideoWidget
import os
import cv2


AVI = "video/x-msvideo"  # AVI
MP4 = 'video/mp4'


def get_supported_mime_types():
    result = []
    for f in QMediaFormat().supportedFileFormats(QMediaFormat.ConversionMode.Decode):
        mime_type = QMediaFormat(f).mimeType()
        result.append(mime_type.name())
    return result

def get_supported_image_formats():
    result = []
    for f in QImageReader().supportedImageFormats():
        image_format = f.toStdString()
        result.append(image_format)
        # result.append(mime_type.name())
    return result

class ConfigWindow(QDialog):
    def __init__(self, parent: QMainWindow | None = None):
        super().__init__(parent)

        video_url: QUrl = parent._player.source()
        video_path = video_url.toLocalFile()
        video_data = parent._metadata
        self.setWindowTitle("Configuration")


        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        start_input = QSpinBox()
        start_input.setRange(0, video_data["duration"] // 1000 * video_data["fps"])
        start_input.setSingleStep(1)
        start_input.setValue(0)
        start_input.setFixedWidth(self.screen().availableGeometry().width() / 10)
        start_input.valueChanged.connect(lambda: self.on_input_returned(start_input, "start"))
        self.layout.addWidget(QLabel("Start frame: "))
        self.layout.addWidget(start_input)

        end_input = QSpinBox()
        end_input.setRange(0, video_data["duration"] // 1000 * video_data["fps"])
        end_input.setSingleStep(1)
        end_input.setValue(video_data["duration"] // 1000 * video_data["fps"])
        end_input.setFixedWidth(self.screen().availableGeometry().width() / 10)
        end_input.valueChanged.connect(lambda: self.on_input_returned(end_input, "end"))
        self.layout.addWidget(QLabel("End frame: "))
        self.layout.addWidget(end_input)

        self.filter_check_box = QCheckBox("Filter")
        self.layout.addWidget(self.filter_check_box)

        self.label_gaussian_size = QLabel("Gaussian filter size: ")
        self.gaussian_filter_size_input = QSpinBox()
        self.gaussian_filter_size_input.setRange(3, 21)
        self.gaussian_filter_size_input.setSingleStep(2)
        self.gaussian_filter_size_input.setValue(11)
        self.gaussian_filter_size_input.setFixedWidth(self.screen().availableGeometry().width() / 10)
        self.gaussian_filter_size_input.valueChanged.connect(lambda: self.on_input_returned(self.gaussian_filter_size_input, "size"))
       

        self.label_gaussian_sigma = QLabel("Gaussian filter sigma: ")
        self.gaussian_filter_sigma_input = QDoubleSpinBox()
        self.gaussian_filter_sigma_input.setRange(0.2, 10.0)
        self.gaussian_filter_sigma_input.setSingleStep(0.2)
        self.gaussian_filter_sigma_input.setValue(3.0)
        self.gaussian_filter_sigma_input.setFixedWidth(self.screen().availableGeometry().width() / 10)
        self.gaussian_filter_sigma_input.valueChanged.connect(lambda: self.on_input_returned(self.gaussian_filter_sigma_input, "sigma"))

        self.layout.addWidget(self.label_gaussian_size)
        self.layout.addWidget(self.gaussian_filter_size_input)
        self.layout.addWidget(self.label_gaussian_sigma)
        self.layout.addWidget(self.gaussian_filter_sigma_input)

        self.label_gaussian_size.setVisible(False)
        self.label_gaussian_sigma.setVisible(False)
        self.gaussian_filter_size_input.setVisible(False)
        self.gaussian_filter_sigma_input.setVisible(False)

        self.masking_check_box = QCheckBox("Masking")
        self.layout.addWidget(self.masking_check_box)
        self.filter_check_box.stateChanged.connect(self.filter_state_changed)
        self.masking_check_box.stateChanged.connect(self.masking_state_changed)

        self.file_button = QPushButton("Browse")
        self.file_button.clicked.connect(self.open)
        self.layout.addWidget(self.file_button)
        self.file_button.setVisible(False)

        self.config = {"start": 0, "end": video_data["duration"] // 1000 * video_data["fps"],
            "filtering": {"enabled": False, "size": (11, 11), "sigma": 3.0}, 
            "masking": {"enabled": False, "mask": ""}
        }
        
        self.config_label = QLabel(self.create_config_label())
        self.layout.addWidget(self.config_label)

        run_button = QPushButton("Run")
        run_button.clicked.connect(self.run_analysis)
        self.layout.addWidget(run_button)

    def create_config_label(self):
        range_text = "Start: {} \nEnd: {}".format(self.config["start"], self.config["end"])
        filter_text = "Filtering:\n\tSize: ({}, {}) \n\tSigma: {}".format(
            self.config["filtering"]["size"][0], self.config["filtering"]["size"][1], self.config["filtering"]["sigma"]
        )
        mask_text = "Masking:\n\tMask: {}".format(os.path.basename(self.config["masking"]["mask"]) if self.config["masking"]["mask"] else "None")

        config_text = "{}\n{}\n{}".format(
            range_text,
            filter_text if self.config["filtering"]["enabled"] else "", 
            mask_text if self.config["masking"]["enabled"] else ""
        )

        # config_text = """Filtering: {}\n\tGaussian filter size: ({}, {}) \n\tGaussian filter sigma: {}\nMasking: {}\n\tMask: {}""".format(
        #     self.config["filtering"]["enabled"], self.config["filtering"]["size"][0], self.config["filtering"]["size"][1], self.config["filtering"]["sigma"],
        #     self.config["masking"]["enabled"], os.path.basename(self.config["masking"]["mask"]) if self.config["masking"]["mask"] else "None"
        # )
        return config_text
    
    
    def run_analysis(self):
        ensure_dialog = QMessageBox(self)
        ensure_dialog.setText("Analysis will start with the following configuration: \n{}".format(self.create_config_label()))
        ensure_dialog.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        ensure_dialog.setDefaultButton(QMessageBox.StandardButton.Ok)
        result = ensure_dialog.exec()
        if result == QMessageBox.StandardButton.Ok:
            self.close()
            self.parent().close()
        else:
            ensure_dialog.close()
            return

    def filter_state_changed(self, state):

        if state == 2:
            self.label_gaussian_sigma.setVisible(True)
            self.label_gaussian_size.setVisible(True)
            self.gaussian_filter_sigma_input.setVisible(True)
            self.gaussian_filter_size_input.setVisible(True)
            self.config["filtering"]["enabled"] = True
            self.config_label.setText(self.create_config_label())
        else:
            self.label_gaussian_size.setVisible(False)
            self.label_gaussian_sigma.setVisible(False)
            self.gaussian_filter_size_input.setVisible(False)
            self.gaussian_filter_sigma_input.setVisible(False)
            self.config["filtering"]["enabled"] = False
            self.config_label.setText(self.create_config_label())
    
    def on_input_returned(self, input_field: QLineEdit, tag: str = None):
        user_input = input_field.text()
        if tag == "size":
            self.config["filtering"][tag] = (int(user_input), int(user_input))
        elif tag == "sigma":
            self.config["filtering"][tag] = float(user_input)
        else:
            self.config[tag] = int(user_input) 
            
        self.config_label.setText(self.create_config_label())
    

    def open(self):
        file_dialog = QFileDialog(self)
        file_dialog.setMimeTypeFilters(get_supported_image_formats())
        file_dialog.setDirectory(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation))
        if file_dialog.exec() == QDialog.DialogCode.Accepted:
            url = file_dialog.selectedUrls()[0]
            self.config["masking"]["mask"] = url.toLocalFile()
            filename = os.path.basename(url.toLocalFile())
            self.config_label.setText(self.create_config_label())

    def masking_state_changed(self, state):
        if state == 2:
            self.file_button.setVisible(True)
            self.config["masking"]["enabled"] = True
            self.config_label.setText(self.create_config_label())
        else:
            self.config["masking"]["enabled"] = False
            self.file_button.setVisible(False)
            self.config_label.setText(self.create_config_label())


    def show(self):
        super().show()
        self.exec()
        

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self._playlist = []  # FIXME 6.3: Replace by QMediaPlaylist?
        self._playlist_index = -1
        self._audio_output = QAudioOutput()
        self._player = QMediaPlayer()
        self.config_window = None
        self._player.setAudioOutput(self._audio_output)

        self._player.errorOccurred.connect(self._player_error)

        tool_bar = QToolBar()
        self.addToolBar(tool_bar)

        file_menu = self.menuBar().addMenu("&File")
        
        icon = QIcon.fromTheme("document-open")
        open_action = QAction(icon, "&Open...", self,
                              shortcut=QKeySequence.StandardKey.Open, triggered=self.open)
        file_menu.addAction(open_action)
        tool_bar.addAction(open_action)
        icon = QIcon.fromTheme("application-exit")
        exit_action = QAction(icon, "E&xit", self,
                              shortcut="Ctrl+Q", triggered=self.close)
        file_menu.addAction(exit_action)

        style = self.style()

        self.analyse_menu = self.menuBar().addMenu("&Analyse")
        icon = QIcon.fromTheme("compute",
                               style.standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        compute_action = QAction(icon, "&Start", self, 
                                 shortcut="Ctrl+A", triggered=self.open_config)
        
        flow_action = QAction(icon, "&Flow", self,
                              shortcut="Ctrl+F", triggered=self.open_config)
        
        self.analyse_menu.addAction(compute_action)
        tool_bar.addAction(compute_action)
        self.analyse_menu.addAction(flow_action)
        self.analyse_menu.setVisible(False)

        icon = QIcon.fromTheme("media-seek-backward")

        play_menu = self.menuBar().addMenu("&Play")
        
        
        icon = QIcon.fromTheme("media-playback-start",
                               style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self._play_action = tool_bar.addAction(icon, "Play")
        self._play_action.triggered.connect(self._player.play)
        play_menu.addAction(self._play_action)

        icon = QIcon.fromTheme("media-skip-backward",
                               style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward))
        self._previous_action = tool_bar.addAction(icon, "Previous")
        self._previous_action.triggered.connect(self.previous_clicked)
        play_menu.addAction(self._previous_action)

        icon = QIcon.fromTheme("media-playback-pause",
                               style.standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self._pause_action = tool_bar.addAction(icon, "Pause")
        self._pause_action.triggered.connect(self._player.pause)
        play_menu.addAction(self._pause_action)

        icon = QIcon.fromTheme("media-skip-forward",
                               style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))
        self._next_action = tool_bar.addAction(icon, "Next")
        self._next_action.triggered.connect(self.next_clicked)
        play_menu.addAction(self._next_action)

        icon = QIcon.fromTheme("media-playback-stop",
                               style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self._stop_action = tool_bar.addAction(icon, "Stop")
        self._stop_action.triggered.connect(self._ensure_stopped)
        play_menu.addAction(self._stop_action)

        # compute_action.setVisible(False)
        # flow_action.setVisible(False)

        self.main_layout = QVBoxLayout()
        self._video_widget = QVideoWidget()
        # self.setCentralWidget(self._video_widget)
        self._player.playbackStateChanged.connect(self.update_buttons)
        self._player.setVideoOutput(self._video_widget)
        self.main_layout.addWidget(self._video_widget)
        # video_layout.addWidget(self._video_widget)

        central_widget = QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        self.update_buttons(self._player.playbackState())
        self._mime_types = []


    def on_frame_slider_value_changed(self, value):
        self._player.setPosition(value * 1000)
    
    
    def closeEvent(self, event):
        self._ensure_stopped()
        event.accept()

    @Slot()
    def open(self):
        self._ensure_stopped()
        file_dialog = QFileDialog(self)

        is_windows = sys.platform == 'win32'
        if not self._mime_types:
            self._mime_types = get_supported_mime_types()
            if (is_windows and AVI not in self._mime_types):
                self._mime_types.append(AVI)
            elif MP4 not in self._mime_types:
                self._mime_types.append(MP4)

        file_dialog.setMimeTypeFilters(self._mime_types)

        default_mimetype = AVI if is_windows else MP4
        if default_mimetype in self._mime_types:
            file_dialog.selectMimeTypeFilter(default_mimetype)

        movies_location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.MoviesLocation)
        file_dialog.setDirectory(movies_location)
        if file_dialog.exec() == QDialog.DialogCode.Accepted:
            url = file_dialog.selectedUrls()[0]
            self._playlist.append(url)
            self._playlist_index = len(self._playlist) - 1
            self._player.setSource(url)
            self._metadata = self._get_video_stats(url.toLocalFile())
            self._player.pause()

            # Create a frame slider for the new media
            if self._metadata["duration"] > 0:
                 # layout.addLayout(video_layout)
                self._frame_slider = QSlider()
                self._frame_slider.setOrientation(Qt.Orientation.Horizontal)
                self._frame_slider.setMinimum(0)
                self._frame_slider.setMaximum(self._metadata["duration"] // 1000)
                self._frame_slider.setFixedWidth(self.screen().availableGeometry().width() / 3)
                self._frame_slider.setToolTip("Position")
                self._frame_slider.valueChanged.connect(self.on_frame_slider_value_changed)
                self.main_layout.addWidget(self._frame_slider)
                self.analyse_menu.setVisible(True)
                self._player.positionChanged.connect(self.on_media_state_changed)
            # self._frame_slider.setValue(0)
            # self._frame_slider.setMaximum(self._player.duration() // 1000)

    def on_media_state_changed(self, state):
            self._frame_slider.setSliderPosition(self._player.position() // 1000)

    def open_config(self):
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        
        if self._player.source().isEmpty():
            advise_window = QMessageBox(self)
            advise_window.setText("No video loaded. Please, load a video first.")
            advise_window.exec()
            return
        
        if self.config_window is None:
            self.config_window = ConfigWindow(self)
        
        self.config_window.show()

    def _get_video_stats(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
        except cv2.error as e:
            print(f"Error opening video file: {e}")
            return None
        
        metadata = {}
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(frame_count / fps) * 1000 # QMediaPlayer measures the duration in milliseconds
        cap.release()
        metadata = {"fps": fps, "duration": duration}
        return metadata
    
    @Slot()
    def _ensure_stopped(self):
        if self._player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
            self._player.stop()

    @Slot()
    def previous_clicked(self):
        # Go to previous track if we are within the first 5 seconds of playback
        # Otherwise, seek to the beginning.
        if self._player.position() <= 5000 and self._playlist_index > 0:
            self._playlist_index -= 1
            self._playlist.previous()
            self._player.setSource(self._playlist[self._playlist_index])
        else:
            self._player.setPosition(0)

    @Slot()
    def next_clicked(self):
        if self._playlist_index < len(self._playlist) - 1:
            self._playlist_index += 1
            self._player.setSource(self._playlist[self._playlist_index])


    @Slot("QMediaPlayer::PlaybackState")
    def update_buttons(self, state):
        media_count = len(self._playlist)
        self._play_action.setEnabled(media_count > 0 and state != QMediaPlayer.PlaybackState.PlayingState)
        self._pause_action.setEnabled(state == QMediaPlayer.PlaybackState.PlayingState)
        self._stop_action.setEnabled(state != QMediaPlayer.PlaybackState.StoppedState)
        self._previous_action.setEnabled(self._player.position() > 0)
        self._next_action.setEnabled(media_count > 1)

    def show_status_message(self, message):
        self.statusBar().showMessage(message, 5000)

    @Slot("QMediaPlayer::Error", str)
    def _player_error(self, error, error_string):
        print(error_string, file=sys.stderr)
        self.show_status_message(error_string)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    available_geometry = main_win.screen().availableGeometry()
    main_win.resize(available_geometry.width() / 3,
                    available_geometry.height() / 2)
    main_win.show()
    app.exec()