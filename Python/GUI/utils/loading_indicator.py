from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen
import math

class LoadingSpinner(QWidget):
    """A circular loading spinner widget."""
    
    def __init__(self, parent=None, centerOnParent=True, disableParentWhenSpinning=False):
        super().__init__(parent)
        
        self._centerOnParent = centerOnParent
        self._disableParentWhenSpinning = disableParentWhenSpinning
        
        self._color = QColor(Qt.GlobalColor.white)
        self._roundness = 70.0
        self._minimumTrailOpacity = 15.0
        self._trailFadePercentage = 70.0
        self._revolutionsPerSecond = 1.5
        self._numberOfLines = 12
        self._lineLength = 10
        self._lineWidth = 2
        self._innerRadius = 10
        self._currentCounter = 0
        self._isSpinning = False
        
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.rotate)
        self.updateSize()
        self.updateTimer()
        self.hide()
    
    def paintEvent(self, event):
        self.updatePosition()
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        if self._currentCounter >= self._numberOfLines:
            self._currentCounter = 0
        
        painter.setPen(Qt.PenStyle.NoPen)
        
        for i in range(self._numberOfLines):
            painter.save()
            painter.translate(self.width() / 2, self.height() / 2)
            rotateAngle = float(360 * i) / float(self._numberOfLines)
            painter.rotate(rotateAngle)
            painter.translate(self._innerRadius, 0)
            distance = self._lineCountDistanceFromInner(i, self._innerRadius, self.width() / 2)
            painter.translate(distance, 0)
            painter.rotate(-rotateAngle)
            painter.translate(-self.width() / 2, -self.height() / 2)
            self._drawLine(painter, i)
            painter.restore()
    
    def _drawLine(self, painter, i):
        painter.save()
        painter.translate(self.width() / 2, self.height() / 2)
        rotateAngle = float(360 * i) / float(self._numberOfLines)
        painter.rotate(rotateAngle)
        painter.translate(self._innerRadius, 0)
        distance = self._lineCountDistanceFromInner(i, self._innerRadius, self.width() / 2)
        painter.translate(distance, 0)
        painter.rotate(-rotateAngle)
        painter.translate(-self.width() / 2, -self.height() / 2)
        
        lineWidth = self._lineWidth
        lineLength = self._lineLength
        
        painter.setPen(QPen(self._color, lineWidth, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPoint(0, -lineLength / 2), QPoint(0, lineLength / 2))
        painter.restore()
    
    def _lineCountDistanceFromInner(self, lineCount, innerRadius, totalRadius):
        return innerRadius + (totalRadius - innerRadius) * float(lineCount) / float(self._numberOfLines)
    
    def start(self):
        self.updatePosition()
        self._isSpinning = True
        self.show()
        
        if self.parentWidget() and self._disableParentWhenSpinning:
            self.parentWidget().setEnabled(False)
        
        if not self._timer.isActive():
            self._timer.start()
            self._currentCounter = 0
    
    def stop(self):
        self._isSpinning = False
        self.hide()
        
        if self.parentWidget() and self._disableParentWhenSpinning:
            self.parentWidget().setEnabled(True)
        
        if self._timer.isActive():
            self._timer.stop()
    
    def setNumberOfLines(self, lines):
        self._numberOfLines = lines
        self._currentCounter = 0
        self.updateTimer()
    
    def setLineLength(self, length):
        self._lineLength = length
        self.updateSize()
    
    def setLineWidth(self, width):
        self._lineWidth = width
        self.updateSize()
    
    def setInnerRadius(self, radius):
        self._innerRadius = radius
        self.updateSize()
    
    def setColor(self, color):
        self._color = color
    
    def setRevolutionsPerSecond(self, rps):
        self._revolutionsPerSecond = rps
        self.updateTimer()
    
    def setRoundness(self, roundness):
        self._roundness = max(0.0, min(100.0, roundness))
    
    def updateSize(self):
        size = (self._innerRadius + self._lineLength) * 2
        self.setFixedSize(size, size)
    
    def updateTimer(self):
        self._timer.setInterval(int(1000 / (self._numberOfLines * self._revolutionsPerSecond)))
    
    def updatePosition(self):
        if self.parentWidget() and self._centerOnParent:
            self.move(
                self.parentWidget().width() / 2 - self.width() / 2,
                self.parentWidget().height() / 2 - self.height() / 2
            )
    
    def rotate(self):
        self._currentCounter += 1
        if self._currentCounter >= self._numberOfLines:
            self._currentCounter = 0
        self.update()

class LoadingOverlay(QWidget):
    """A loading overlay with spinner and optional message."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Make the overlay cover the entire parent widget
        if parent:
            self.setGeometry(0, 0, parent.width(), parent.height())
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Create spinner
        self.spinner = LoadingSpinner(self)
        layout.addWidget(self.spinner, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Create message label
        self.message_label = QLabel(self)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: transparent;
                font-size: 14px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.message_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Set overlay style
        self.setStyleSheet("""
            LoadingOverlay {
                background-color: rgba(0, 0, 0, 0.7);
            }
        """)
        
        self.hide()
    
    def start(self, message=""):
        """Start the loading overlay with an optional message."""
        self.message_label.setText(message)
        self.spinner.start()
        self.show()
    
    def stop(self):
        """Stop the loading overlay."""
        self.spinner.stop()
        self.hide()
    
    def resizeEvent(self, event):
        """Handle resize events to ensure the overlay covers the parent."""
        if self.parentWidget():
            self.setGeometry(0, 0, self.parentWidget().width(), self.parentWidget().height())
        super().resizeEvent(event) 