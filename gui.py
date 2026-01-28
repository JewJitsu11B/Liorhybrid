"""
Bayesian Cognitive Field - PyQt6 GUI Main Menu

Windows-style centered interface for training configuration.
"""
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QFileDialog, QMessageBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QTextEdit,
    QProgressBar, QStatusBar, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIcon, QPalette, QColor


class MainMenuWindow(QMainWindow):
    """Main menu window with Windows-style interface."""

    def __init__(self):
        super().__init__()
        self.selected_config = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Bayesian Cognitive Field - Training System")
        self.setMinimumSize(600, 500)
        self.resize(700, 550)

        # Center on screen
        self.center_on_screen()

        # Apply Windows style
        self.setStyle_windows()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        header = QLabel("BAYESIAN COGNITIVE FIELD")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_font = QFont("Segoe UI", 18, QFont.Weight.Bold)
        header.setFont(header_font)
        layout.addWidget(header)

        subtitle = QLabel("Advanced Physics-Based Multimodal AI Training System")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: #666;")
        layout.addWidget(subtitle)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # Main menu group
        menu_group = QGroupBox("Main Menu")
        menu_group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        menu_layout = QVBoxLayout(menu_group)
        menu_layout.setSpacing(8)

        # Menu buttons
        buttons = [
            ("0. ONE-TOUCH 250M", "Load all data, optimal config", self.on_one_touch),
            ("1. Quick Start (Geometric)", "Recommended - Train geometric weights", self.on_geometric),
            ("2. Full Training", "Train everything end-to-end", self.on_full_training),
            ("3. Resume from Checkpoint", "Continue previous training", self.on_resume),
            ("4. Generate Sample Dataset", "Create sample training data", self.on_generate_sample),
            ("5. Inference / Chat Mode", "Run inference with trained model", self.on_inference),
            ("6. Inspect Checkpoint", "View checkpoint details", self.on_inspect),
            ("7. Evaluate Checkpoint", "Run validation metrics", self.on_evaluate),
            ("8. Config Cost Calculator", "Estimate params/memory/compute", self.on_cost_calc),
        ]

        for text, tooltip, callback in buttons:
            btn = QPushButton(text)
            btn.setToolTip(tooltip)
            btn.setFont(QFont("Segoe UI", 10))
            btn.setMinimumHeight(36)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(callback)
            menu_layout.addWidget(btn)

        layout.addWidget(menu_group)

        # Spacer
        layout.addStretch()

        # Bottom buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()

        exit_btn = QPushButton("Exit")
        exit_btn.setFont(QFont("Segoe UI", 10))
        exit_btn.setMinimumWidth(100)
        exit_btn.setMinimumHeight(32)
        exit_btn.clicked.connect(self.close)
        bottom_layout.addWidget(exit_btn)

        layout.addLayout(bottom_layout)

        # Status bar
        self.statusBar().showMessage("Ready - Python 3.13t Free-Threaded")

    def center_on_screen(self):
        """Center window on the screen."""
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        x = (screen.width() - size.width()) // 2
        y = (screen.height() - size.height()) // 2
        self.move(x, y)

    def setStyle_windows(self):
        """Apply Windows-style appearance."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #333;
            }
            QPushButton {
                background-color: #e1e1e1;
                border: 1px solid #adadad;
                border-radius: 3px;
                padding: 6px 12px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #e5f1fb;
                border-color: #0078d7;
            }
            QPushButton:pressed {
                background-color: #cce4f7;
            }
            QStatusBar {
                background-color: #f0f0f0;
                border-top: 1px solid #c0c0c0;
            }
        """)

    def on_one_touch(self):
        """Handle ONE-TOUCH 250M option."""
        self.selected_config = {'action': 'one_touch_250m'}
        self.statusBar().showMessage("Starting ONE-TOUCH 250M configuration...")
        self.close()

    def on_geometric(self):
        """Handle Geometric Training option."""
        self.selected_config = {'action': 'geometric'}
        self.statusBar().showMessage("Starting Geometric Training...")
        self.close()

    def on_full_training(self):
        """Handle Full Training option."""
        self.selected_config = {'action': 'full'}
        self.statusBar().showMessage("Starting Full Training...")
        self.close()

    def on_resume(self):
        """Handle Resume from Checkpoint option."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", "./checkpoints",
            "Checkpoint Files (*.pt *.pth);;All Files (*)"
        )
        if file_path:
            self.selected_config = {'action': 'resume', 'checkpoint_path': file_path}
            self.close()

    def on_generate_sample(self):
        """Handle Generate Sample Dataset option."""
        self.selected_config = {'action': 'generate_sample'}
        QMessageBox.information(self, "Generate Sample", "Sample dataset generation started.")
        self.close()

    def on_inference(self):
        """Handle Inference/Chat Mode option."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint", "./checkpoints",
            "Checkpoint Files (*.pt *.pth);;All Files (*)"
        )
        if file_path:
            self.selected_config = {'action': 'inference', 'checkpoint_path': file_path}
            self.close()

    def on_inspect(self):
        """Handle Inspect Checkpoint option."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint to Inspect", "./checkpoints",
            "Checkpoint Files (*.pt *.pth);;All Files (*)"
        )
        if file_path:
            self.selected_config = {'action': 'inspect', 'checkpoint_path': file_path}
            self.close()

    def on_evaluate(self):
        """Handle Evaluate Checkpoint option."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint to Evaluate", "./checkpoints",
            "Checkpoint Files (*.pt *.pth);;All Files (*)"
        )
        if file_path:
            self.selected_config = {'action': 'evaluate', 'checkpoint_path': file_path}
            self.close()

    def on_cost_calc(self):
        """Handle Config Cost Calculator option."""
        self.selected_config = {'action': 'cost_calculator'}
        self.close()


class DataSelectionDialog(QMainWindow):
    """Dialog for selecting training data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_files = []
        self.data_type = 'text'
        self.init_ui()

    def init_ui(self):
        """Initialize the data selection UI."""
        self.setWindowTitle("Select Training Data")
        self.setMinimumSize(500, 400)
        self.resize(550, 450)

        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - 550) // 2
        y = (screen.height() - 450) // 2
        self.move(x, y)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)

        # Data type selection
        type_group = QGroupBox("Data Type")
        type_layout = QHBoxLayout(type_group)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["Text Files", "MNIST", "Image + Text", "Video + Text"])
        self.type_combo.setMinimumWidth(200)
        type_layout.addWidget(QLabel("Type:"))
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()

        layout.addWidget(type_group)

        # File selection
        file_group = QGroupBox("Selected Files")
        file_layout = QVBoxLayout(file_group)

        self.file_list = QTextEdit()
        self.file_list.setReadOnly(True)
        self.file_list.setPlaceholderText("No files selected...")
        file_layout.addWidget(self.file_list)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Files...")
        add_btn.clicked.connect(self.add_files)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_files)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        file_layout.addLayout(btn_layout)

        layout.addWidget(file_group)

        # Bottom buttons
        bottom = QHBoxLayout()
        bottom.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumWidth(80)
        cancel_btn.clicked.connect(self.reject)

        ok_btn = QPushButton("OK")
        ok_btn.setMinimumWidth(80)
        ok_btn.clicked.connect(self.accept)

        bottom.addWidget(cancel_btn)
        bottom.addWidget(ok_btn)
        layout.addLayout(bottom)

    def add_files(self):
        """Open file dialog to add files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Training Files", "./data",
            "All Supported (*.txt *.pdf *.docx *.json *.jsonl *.md *.py);;Text Files (*.txt);;All Files (*)"
        )
        if files:
            self.selected_files.extend(files)
            self.file_list.setPlainText("\n".join(self.selected_files))

    def clear_files(self):
        """Clear selected files."""
        self.selected_files = []
        self.file_list.clear()

    def accept(self):
        """Accept and close."""
        self.close()

    def reject(self):
        """Reject and close."""
        self.selected_files = []
        self.close()


def show_main_menu():
    """Show the main menu and return the selected configuration."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = MainMenuWindow()
    window.show()
    app.exec()

    return window.selected_config


def show_data_selection():
    """Show data selection dialog and return selected files."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = DataSelectionDialog()
    dialog.show()
    app.exec()

    return dialog.selected_files, dialog.type_combo.currentText()


if __name__ == "__main__":
    # Test the GUI
    config = show_main_menu()
    print(f"Selected config: {config}")
