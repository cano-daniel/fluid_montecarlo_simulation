"""
Simple, clean PyQtGraph plotter for Ising model visualization.
No errors, no warnings, just fast real-time plotting!

Installation:
    pip install pyqtgraph PyQt5 numpy
"""
from pathlib import Path
import subprocess
import shutil
from PIL import Image, Image as PILImage

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import time

class IsingPlotter:
    """
    Fast real-time plotter with configurable named sliders.
    
    Example usage:
        sliders_config = [
            {'name': 'temperature', 'min': 0.1, 'max': 5.0, 'initial': 2.0, 
             'position': 'bottom', 'orientation': 'horizontal'},
            {'name': 'field', 'min': -1.0, 'max': 1.0, 'initial': 0.0,
             'position': 'left', 'orientation': 'vertical'}
        ]
        
        plotter = IsingPlotter(matrix_shape=(100, 100), sliders=sliders_config)
        
        for step in range(1000):
            temp = plotter.get_slider('temperature')
            field = plotter.get_slider('field')
            
            matrix = your_simulation_step(temp, field)
            plotter.update(matrix)
        
        plotter.close()
    """
    
    def __init__(self, matrix_shape, colors=None, title="Ising Model",
                 sliders=None, window_size=(800, 900)):
        """
        Initialize plotter with configurable sliders.
        
        Args:
            matrix_shape: Tuple (height, width) of your matrix
            colors: List of 2 colors as hex strings or RGB tuples
            title: Window title
            sliders: List of slider configurations. Each dict contains:
                - name: Unique identifier for the slider
                - min: Minimum value
                - max: Maximum value
                - initial: Initial value
                - position: 'top', 'bottom', 'left', 'right' (default: 'bottom')
                - orientation: 'horizontal' or 'vertical' (auto-detected from position if not specified)
                - label: Display label (defaults to name if not specified)
            window_size: Window dimensions (width, height) in pixels
        """
        self.matrix_shape = matrix_shape
        self.title = title
        
        # Default slider config if none provided
        if sliders is None:
            sliders = [
                {'name': 'temperature', 'min': 0.1, 'max': 5.0, 'initial': 2.0,
                 'position': 'bottom', 'label': 'Temperature'}
            ]
        
        self.sliders_config = sliders
        self.sliders = {}  # Dictionary to store slider widgets by name
        self.slider_labels = {}  # Dictionary to store value labels by name
        
        # Parse colors
        if colors is None:
            self.colors = [(21, 16, 16), (123, 49, 41)]
        else:
            self.colors = [self._parse_color(c) for c in colors]
        
        # Create lookup table
        self.lut = self._create_lookup_table()
        
        # Initialize Qt application
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        
        # Create main window widget
        self.main_widget = QtWidgets.QWidget()
        self.main_widget.setWindowTitle(title)
        self.main_widget.resize(*window_size)
        
        # Create main layout
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        
        # Create horizontal layout for left sliders + center content + right sliders
        self.content_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.content_layout)
        
        # Containers for different slider positions
        self.left_sliders_layout = QtWidgets.QVBoxLayout()
        self.right_sliders_layout = QtWidgets.QVBoxLayout()
        self.top_sliders_layout = QtWidgets.QVBoxLayout()
        self.bottom_sliders_layout = QtWidgets.QVBoxLayout()
        
        # Create center layout for plot
        self.center_layout = QtWidgets.QVBoxLayout()
        
        # Add top sliders
        self.main_layout.insertLayout(0, self.top_sliders_layout)
        
        # Add left, center, right to content layout
        self.content_layout.addLayout(self.left_sliders_layout)
        self.content_layout.addLayout(self.center_layout)
        self.content_layout.addLayout(self.right_sliders_layout)
        
        # Add bottom sliders
        self.main_layout.addLayout(self.bottom_sliders_layout)
        
        # Create PyQtGraph graphics layout for the plot
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.center_layout.addWidget(self.graphics_widget)
        
        # Create plot area
        self.plot = self.graphics_widget.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.setTitle(title)
        
        # Create image item
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        self.image_item.setLookupTable(self.lut)
        
        # Initialize with zeros
        dummy_matrix = np.zeros(matrix_shape, dtype=np.uint8)
        self.image_item.setImage(dummy_matrix.T)
        
        # Create all sliders
        self._create_sliders()
        
        # Create info panel
        self._create_info_panel()
        
        # Show window
        self.main_widget.show()
        
        # Performance tracking
        self.update_count = 0
        self.last_update_time = time.time()
        self.fps = 0.0
        
        print(f"IsingPlotter initialized: {matrix_shape[0]}×{matrix_shape[1]}")
        print(f"Sliders created: {list(self.sliders.keys())}")
        print(f"Colors (RGB): {self.colors}")
    
    def _create_sliders(self):
        """Create all configured sliders"""
        for config in self.sliders_config:
            name = config['name']
            min_val = config.get('min', 0.0)
            max_val = config.get('max', 1.0)
            initial = config.get('initial', (min_val + max_val) / 2)
            position = config.get('position', 'bottom')
            label_text = config.get('label', name.capitalize())
            
            # Auto-detect orientation based on position
            if 'orientation' in config:
                orientation = config['orientation']
            else:
                orientation = 'vertical' if position in ['left', 'right'] else 'horizontal'
            
            # Store config info for later use
            config['_min'] = min_val
            config['_max'] = max_val
            config['_orientation'] = orientation
            
            # Create slider in appropriate position
            if position == 'left':
                self._add_slider_to_layout(self.left_sliders_layout, name, label_text,
                                          min_val, max_val, initial, orientation)
            elif position == 'right':
                self._add_slider_to_layout(self.right_sliders_layout, name, label_text,
                                          min_val, max_val, initial, orientation)
            elif position == 'top':
                self._add_slider_to_layout(self.top_sliders_layout, name, label_text,
                                          min_val, max_val, initial, orientation)
            else:  # bottom
                self._add_slider_to_layout(self.bottom_sliders_layout, name, label_text,
                                          min_val, max_val, initial, orientation)
    
    def _add_slider_to_layout(self, parent_layout, name, label_text, 
                             min_val, max_val, initial, orientation):
        """Add a slider to the specified layout"""
        is_horizontal = (orientation == 'horizontal')
        
        # Create container layout
        if is_horizontal:
            slider_layout = QtWidgets.QHBoxLayout()
        else:
            slider_layout = QtWidgets.QVBoxLayout()
        
        # Label
        label = QtWidgets.QLabel(label_text + ":")
        label.setStyleSheet("font-size: 14px; font-weight: bold;")
        slider_layout.addWidget(label)
        
        # Slider
        qt_orientation = QtCore.Qt.Orientation.Horizontal if is_horizontal else QtCore.Qt.Orientation.Vertical
        slider = QtWidgets.QSlider(qt_orientation)
        slider.setMinimum(0)
        slider.setMaximum(1000)
        
        # Set initial position
        initial_position = int((initial - min_val) / (max_val - min_val) * 1000)
        slider.setValue(initial_position)
        
        # Connect to update function
        slider.valueChanged.connect(lambda value, n=name: self._on_slider_change(n, value))
        
        slider_layout.addWidget(slider)
        
        # Value display
        value_label = QtWidgets.QLabel(f"{initial:.3f}")
        value_label.setStyleSheet("font-size: 14px; min-width: 60px;")
        slider_layout.addWidget(value_label)
        
        # Store references
        self.sliders[name] = slider
        self.slider_labels[name] = value_label
        
        # Add to parent layout
        parent_layout.addLayout(slider_layout)
    
    def _on_slider_change(self, name, value):
        """Called when any slider value changes"""
        # Find config for this slider
        config = next((c for c in self.sliders_config if c['name'] == name), None)
        if config is None:
            return
        
        min_val = config['_min']
        max_val = config['_max']
        
        # Convert slider position (0-1000) to actual value
        actual_value = min_val + (value / 1000.0) * (max_val - min_val)
        
        # Update label
        self.slider_labels[name].setText(f"{actual_value:.3f}")
    
    def get_slider(self, name):
        """
        Get current value of a named slider.
        
        Args:
            name: Name of the slider
            
        Returns:
            float: Current slider value
        """
        if name not in self.sliders:
            raise ValueError(f"Slider '{name}' not found. Available: {list(self.sliders.keys())}")
        
        config = next((c for c in self.sliders_config if c['name'] == name), None)
        if config is None:
            raise ValueError(f"Config for slider '{name}' not found")
        
        slider_value = self.sliders[name].value()
        min_val = config['_min']
        max_val = config['_max']
        actual_value = min_val + (slider_value / 1000.0) * (max_val - min_val)
        return actual_value
    
    def set_slider(self, name, value):
        """
        Programmatically set a slider value.
        
        Args:
            name: Name of the slider
            value: Value to set
        """
        if name not in self.sliders:
            raise ValueError(f"Slider '{name}' not found. Available: {list(self.sliders.keys())}")
        
        config = next((c for c in self.sliders_config if c['name'] == name), None)
        if config is None:
            raise ValueError(f"Config for slider '{name}' not found")
        
        min_val = config['_min']
        max_val = config['_max']
        
        # Clamp to valid range
        value = max(min_val, min(max_val, value))
        
        # Convert to slider position
        position = int((value - min_val) / (max_val - min_val) * 1000)
        
        self.sliders[name].setValue(position)
    
    def get_all_sliders(self):
        """
        Get all slider values as a dictionary.
        
        Returns:
            dict: Dictionary mapping slider names to their current values
        """
        return {name: self.get_slider(name) for name in self.sliders.keys()}
    
    def _create_info_panel(self):
        """Create info panel at the bottom"""
        info_layout = QtWidgets.QHBoxLayout()
        
        self.fps_label = QtWidgets.QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("font-size: 12px; color: gray;")
        info_layout.addWidget(self.fps_label)
        
        info_layout.addStretch()
        
        self.updates_label = QtWidgets.QLabel("Updates: 0")
        self.updates_label.setStyleSheet("font-size: 12px; color: gray;")
        info_layout.addWidget(self.updates_label)
        
        self.main_layout.addLayout(info_layout)
    
    def _parse_color(self, color):
        """Convert hex string or RGB tuple to RGB tuple"""
        if isinstance(color, str):
            color = color.lstrip('#')
            return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        return tuple(color[:3])
    
    def _create_lookup_table(self):
        """Create color lookup table"""
        lut = np.zeros((256, 3), dtype=np.uint8)
        
        for i in range(256):
            t = i / 255.0
            r = int(self.colors[0][0] * (1 - t) + self.colors[1][0] * t)
            g = int(self.colors[0][1] * (1 - t) + self.colors[1][1] * t)
            b = int(self.colors[0][2] * (1 - t) + self.colors[1][2] * t)
            lut[i] = [r, g, b]
        
        return lut
    
    def update(self, matrix):
        """
        Update the display with new matrix data.
        
        Args:
            matrix: 2D numpy array with values in [0, 1]
        """
        if matrix.shape != self.matrix_shape:
            raise ValueError(
                f"Matrix shape {matrix.shape} doesn't match "
                f"expected shape {self.matrix_shape}"
            )
        
        # Convert to uint8
        matrix_uint8 = (matrix * 255).astype(np.uint8)
        
        # Update image
        self.image_item.setImage(matrix_uint8.T, autoLevels=False)
        
        # Process Qt events
        if self.app is not None:
            self.app.processEvents()
        
        # Calculate FPS
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        if time_delta > 0:
            self.fps = 1.0 / time_delta
        self.last_update_time = current_time
        
        self.update_count += 1
        
        # Update info labels every 10 frames
        if self.update_count % 10 == 0:
            self.fps_label.setText(f"FPS: {self.fps:.1f}")
            self.updates_label.setText(f"Updates: {self.update_count}")
    
    def set_colors(self, colors):
        """Change the colormap"""
        self.colors = [self._parse_color(c) for c in colors]
        self.lut = self._create_lookup_table()
        self.image_item.setLookupTable(self.lut)
        print(f"Colors updated to: {self.colors}")
    
    def get_fps(self):
        """Get current FPS"""
        return self.fps
    
    def close(self):

        """Close the window"""
        self.main_widget.close()
        print(f"Plotter closed. Total updates: {self.update_count}")



class FastIsingRecorder:
    """
    Ultra-fast recorder that writes PNG frames directly during animation.
    No intermediate storage, no data conversion - just direct frame writing.
    """
    
    def __init__(self, output_folder="ising_frames", resolution=None):
        """
        Args:
            output_folder: Where to save frames
            resolution: (width, height) for output images. If None, uses matrix size.
        """
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.resolution = resolution
        self.frame_count = 0
        self.recording = False
        
        # Color mapping (0 -> color1, 1 -> color2)
        self.color_map = np.array([
            [21, 16, 16],      # Color for 0
            [123, 49, 41]      # Color for 1
        ], dtype=np.uint8)
        
        # Clear existing frames
        for file in self.output_folder.glob("frame_*.png"):
            file.unlink()
        
        print(f"FastIsingRecorder initialized: {self.output_folder}")
    
    def start_recording(self):
        """Start recording"""
        self.recording = True
        self.frame_count = 0
        print("Recording started...")
    
    def stop_recording(self):
        """Stop recording"""
        self.recording = False
        print(f"Recording stopped. {self.frame_count} frames saved.")
    
    def save_frame(self, matrix):
        """
        Save frame directly as PNG - ULTRA FAST!
        
        Args:
            matrix: 2D numpy array with values 0 or 1
        """
        if not self.recording:
            return
        
        # Convert binary matrix to RGB using color map
        # This is VERY fast - direct numpy indexing
        rgb_image = self.color_map[matrix.astype(np.uint8)]
        
        # Resize if needed
        if self.resolution is not None:
            img = Image.fromarray(rgb_image, mode='RGB')
            img = img.resize(self.resolution, Image.Resampling.NEAREST)
        else:
            img = Image.fromarray(rgb_image, mode='RGB')
        
        # Save directly to disk
        filename = self.output_folder / f"frame_{self.frame_count:06d}.png"
        img.save(filename, optimize=False)  # optimize=False for speed
        
        self.frame_count += 1
    
    def create_video(self, fps=30, output_name="ising_simulation", crf=18):
        """
        Create video from saved frames using FFmpeg.
        
        Args:
            fps: Frames per second
            output_name: Output filename (without extension)
            crf: Quality (0=lossless, 18=high quality, 23=default, 51=worst)
        """
        output_path = self.output_folder / f"{output_name}_{fps}fps.mp4"
        
        print(f"Creating video from {self.frame_count} frames at {fps} FPS...")
        
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(self.output_folder / 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', 'medium',
            '-pix_fmt', 'yuv420p',
            str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                file_size = output_path.stat().st_size / (1024*1024)
                print(f"✓ Video created: {output_path}")
                print(f"  Size: {file_size:.1f} MB")
                return output_path
            else:
                print(f"✗ Video creation failed: {result.stderr}")
                return None
        except Exception as e:
            print(f"Error creating video: {e}")
            return None
    
    def set_colors(self, color1, color2):
        """
        Change the color mapping.
        
        Args:
            color1: RGB tuple or hex string for value 0
            color2: RGB tuple or hex string for value 1
        """
        def parse_color(c):
            if isinstance(c, str):
                c = c.lstrip('#')
                return tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
            return tuple(c[:3])
        
        self.color_map[0] = parse_color(color1)
        self.color_map[1] = parse_color(color2)

