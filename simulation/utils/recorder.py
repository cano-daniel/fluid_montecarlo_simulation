import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
from pathlib import Path
import time
import subprocess
import shutil

class MatplotlibRecorder:
    """
    A two-phase recorder for Ising model simulations:
    1. Recording phase: Collects raw matrix data (fast)
    2. Generation phase: Creates images/videos from saved data (slow)
    """
    
    def __init__(self, output_folder="ising_frames", resolution=(3840, 2160), 
                 colors=None):
        """
        Initialize the recorder.
        
        Args:
            output_folder: Where to save frames and videos
            resolution: Target image size (width, height) in pixels
            colors: List of hex colors for states [state0_color, state1_color, ...]
                   Default: ["#151010FF", "#7B3129FF"] (dark brown, light brown)
        """
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.frame_count = 0
        self.recording = False
        self.resolution = (int(resolution[0]), int(resolution[1]))
        
        # Configurable colors!
        if colors is None:
            self.colors = ["#151010FF", "#7B3129FF"]  # Default colors
        else:
            self.colors = list(colors)
        
        # Create colormap from colors
        self.cmap = ListedColormap(self.colors)
        
        # Setup high-resolution figure parameters
        self.dpi = 100
        self.fig_width = self.resolution[0] / self.dpi
        self.fig_height = self.resolution[1] / self.dpi
        
        # Data storage buffer
        self.data_buffer = []
        
        # Clean output folder
        self._clean_output_folder()
        
        print(f"Matplotlib recorder initialized")
        print(f"Output folder: {self.output_folder}")
        print(f"Target resolution: {self.resolution[0]}×{self.resolution[1]} px")
        print(f"Colors: {self.colors}")
        print("Mode: Fast data collection (generate images later)")

    def _clean_output_folder(self):
        """Remove old output files"""
        for file in self.output_folder.glob("frame_*.png"):
            file.unlink()
        for file in self.output_folder.glob("*.mp4"):
            file.unlink()
        for file in self.output_folder.glob("*.gif"):
            file.unlink()
        for file in self.output_folder.glob("*.pkl"):
            file.unlink()

    def set_colors(self, colors):
        """
        Change the colormap colors.
        
        Args:
            colors: List of hex color strings, e.g., ["#000000", "#FFFFFF"]
        """
        self.colors = list(colors)
        self.cmap = ListedColormap(self.colors)
        print(f"Colors updated to: {self.colors}")

    def start_recording(self):
        """Start recording frames"""
        self.recording = True
        self.frame_count = 0
        self.data_buffer = []
        print("Recording started...")

    def stop_recording(self):
        """Stop recording and save all collected data"""
        self.recording = False
        
        if self.data_buffer:
            df = pd.DataFrame(self.data_buffer)
            data_file = self.output_folder / "matrix_data.pkl"
            df.to_pickle(data_file)
            print(f"Saved {len(self.data_buffer)} frames to {data_file}")
            
        print(f"Recording stopped. {self.frame_count} frames captured.")

    def save_frame(self, matrix_data, temperature):
        """
        Save current frame data (super fast - no image generation).
        
        Args:
            matrix_data: 2D numpy array representing the system state
            temperature: Current temperature value
        """
        if not self.recording:
            return
        
        # Validate input
        matrix_data = np.asarray(matrix_data)
        if matrix_data.ndim != 2:
            raise ValueError("matrix_data must be 2D")
        
        # Store frame data
        frame_record = {
            'frame_number': self.frame_count,
            'timestamp': time.time(),
            'temperature': float(temperature),
            'matrix_shape_height': int(matrix_data.shape[0]),
            'matrix_shape_width': int(matrix_data.shape[1]),
            'matrix_data': matrix_data.flatten().tolist(),
        }
        
        # Save matrix dimensions on first frame
        if self.frame_count == 0:
            self._save_dimensions(matrix_data.shape)
        
        self.data_buffer.append(frame_record)
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            print(f"Collected {self.frame_count} frames...")

    def _save_dimensions(self, shape):
        """Save matrix dimensions to file"""
        dimensions_file = self.output_folder / "matrix_dimensions.txt"
        with open(dimensions_file, 'w') as f:
            f.write(f"n_height={shape[0]}\n")
            f.write(f"n_width={shape[1]}\n")
            f.write(f"resolution={self.resolution[0]}x{self.resolution[1]}\n")
            f.write(f"dpi={self.dpi}\n")
            f.write(f"colors={','.join(self.colors)}\n")
        print(f"Saved dimensions: {shape[0]}×{shape[1]}")

    def _read_dimensions_file(self):
        """Read matrix_dimensions.txt and return (n_h, n_w, res_w, res_h, colors)"""
        dimensions_file = self.output_folder / "matrix_dimensions.txt"
        if not dimensions_file.exists():
            return None
        
        n_h = n_w = res_w = res_h = None
        colors = None
        
        with open(dimensions_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("n_height="):
                    n_h = int(line.split("=", 1)[1])
                elif line.startswith("n_width="):
                    n_w = int(line.split("=", 1)[1])
                elif line.startswith("resolution="):
                    res = line.split("=", 1)[1]
                    try:
                        a, b = res.lower().split("x")
                        res_w = int(a)
                        res_h = int(b)
                    except:
                        pass
                elif line.startswith("colors="):
                    colors = line.split("=", 1)[1].split(",")
        
        return (n_h, n_w, res_w, res_h, colors)

    def generate_images(self, max_frames=None, force_target_resolution=None, 
                       force_colors=None):
        """
        Generate PNG images from collected data using nearest-neighbor upscaling.
        
        Args:
            max_frames: Limit number of frames to generate
            force_target_resolution: Override output resolution (width, height)
            force_colors: Override colors from saved data
        """
        # Load saved data
        data_file = self.output_folder / "matrix_data.pkl"
        if not data_file.exists():
            print("No data file found! Run recording first.")
            return
        
        print("Loading saved data...")
        df = pd.read_pickle(data_file)
        if max_frames:
            df = df.head(max_frames)
        print(f"Generating {len(df)} images...")
        
        # Read dimensions file
        dims = self._read_dimensions_file()
        if dims:
            file_n_h, file_n_w, file_res_w, file_res_h, file_colors = dims
            print(f"Matrix: {file_n_h}×{file_n_w}, Resolution hint: {file_res_w}×{file_res_h}")
            if file_colors:
                print(f"Saved colors: {file_colors}")
        else:
            file_res_w = file_res_h = None
            file_colors = None
        
        # Determine target resolution
        if force_target_resolution:
            target_w, target_h = int(force_target_resolution[0]), int(force_target_resolution[1])
        elif file_res_w and file_res_h:
            target_w, target_h = int(file_res_w), int(file_res_h)
        else:
            target_w, target_h = int(self.resolution[0]), int(self.resolution[1])
        print(f"Output resolution: {target_w}×{target_h} px")
        
        # Determine colors to use
        if force_colors:
            use_colors = force_colors
        elif file_colors:
            use_colors = file_colors
        else:
            use_colors = self.colors
        
        cmap = ListedColormap(use_colors)
        print(f"Using colors: {use_colors}")
        
        # Prepare colormap converter
        norm = Normalize(vmin=0.0, vmax=1.0)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        
        # Generate images
        for idx, row in df.iterrows():
            try:
                # Reconstruct matrix
                matrix_data = np.array(row['matrix_data'], dtype=float).reshape(
                    int(row['matrix_shape_height']),
                    int(row['matrix_shape_width'])
                )
                
                # Upscale using nearest-neighbor
                resized = nearest_neighbor_upscale(
                    matrix_data, 
                    target_width=target_w, 
                    target_height=target_h
                )
                
                # Apply colormap
                rgba_img = mappable.to_rgba(resized)
                rgba_uint8 = (rgba_img * 255).astype(np.uint8)
                
                # Save image
                filename = self.output_folder / f"frame_{int(row['frame_number']):06d}.png"
                mpimg.imsave(str(filename), rgba_uint8)
                
                if (idx + 1) % 100 == 0:
                    print(f"Generated {idx + 1}/{len(df)} images...")
                    
            except Exception as e:
                print(f"Error generating frame {row.get('frame_number', idx)}: {e}")
        
        print(f"✓ Image generation complete! {len(df)} images saved")

    def create_video_direct(self, fps=30, max_frames=None, duration=None, 
                           output_name="ising_simulation"):
        """
        Create video from PNG files using ffmpeg.
        
        Args:
            fps: Frames per second
            max_frames: Limit number of frames
            duration: Alternative to max_frames (duration in seconds)
            output_name: Base name for output file
        """
        frame_files = sorted(self.output_folder.glob("frame_*.png"))
        if not frame_files:
            print("No frames found to create video")
            return None
        
        # Determine frame count
        if duration is not None:
            max_frames = int(fps * duration)
        if max_frames is not None:
            frame_files = frame_files[:max_frames]
        
        print(f"Creating video from {len(frame_files)} frames at {fps} FPS")
        
        # Try direct FFmpeg approach
        output_path = self.output_folder / f"{output_name}_{fps}fps.mp4"
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', str(self.output_folder / 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-crf', '18',  # High quality (lower = better, 18 is visually lossless)
                '-pix_fmt', 'yuv420p',
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                file_size = output_path.stat().st_size / (1024*1024)
                print(f"✓ Video created: {output_path}")
                print(f"  Size: {file_size:.1f} MB")
                return output_path
            else:
                print(f"✗ FFmpeg failed: {result.stderr}")
                return None
        except Exception as e:
            print(f"Error creating video: {e}")
            return None


# ============================================================================
# INDEPENDENT PLOTTING FUNCTIONS
# ============================================================================

def nearest_neighbor_upscale(matrix, target_width, target_height):
    """
    Upscale matrix to target size using nearest-neighbor interpolation.
    Each matrix cell becomes a rectangular block of pixels.
    
    Args:
        matrix: 2D numpy array (n_height, n_width)
        target_width: Output width in pixels
        target_height: Output height in pixels
    
    Returns:
        Upscaled 2D array of shape (target_height, target_width)
    """
    n_height, n_width = matrix.shape
    
    # Create index mappings for nearest-neighbor
    row_idx = np.floor(np.linspace(0, n_height, target_height, endpoint=False)).astype(int)
    col_idx = np.floor(np.linspace(0, n_width, target_width, endpoint=False)).astype(int)
    
    # Safety clamp
    row_idx = np.clip(row_idx, 0, n_height - 1)
    col_idx = np.clip(col_idx, 0, n_width - 1)
    
    # Map to nearest neighbors
    resized = matrix[np.ix_(row_idx, col_idx)]
    
    return resized


def plot_matrix_to_image(matrix, output_path, colors=None, resolution=(1920, 1080)):
    """
    Independent function to plot a matrix and save as image.
    
    Args:
        matrix: 2D numpy array to visualize
        output_path: Where to save the image (str or Path)
        colors: List of hex colors for states
        resolution: Output image size (width, height)
    
    Example:
        matrix = np.random.choice([0, 1], size=(100, 100))
        plot_matrix_to_image(matrix, "output.png", 
                           colors=["#000000", "#FFFFFF"],
                           resolution=(3840, 2160))
    """
    if colors is None:
        colors = ["#151010FF", "#7B3129FF"]
    
    # Upscale matrix to target resolution
    target_w, target_h = resolution
    resized = nearest_neighbor_upscale(matrix, target_w, target_h)
    
    # Apply colormap
    cmap = ListedColormap(colors)
    norm = Normalize(vmin=0.0, vmax=1.0)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    rgba_img = mappable.to_rgba(resized)
    rgba_uint8 = (rgba_img * 255).astype(np.uint8)
    
    # Save
    mpimg.imsave(str(output_path), rgba_uint8)
    print(f"Image saved to {output_path}")


def plot_matrix_matplotlib(matrix, colors=None, title="Ising Model", show=True):
    """
    Plot matrix using matplotlib (for interactive viewing/debugging).
    
    Args:
        matrix: 2D numpy array
        colors: List of hex colors
        title: Plot title
        show: Whether to display the plot
    
    Returns:
        fig, ax objects
    """
    if colors is None:
        colors = ["#151010FF", "#7B3129FF"]
    
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(matrix, cmap=cmap, interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    
    if show:
        plt.show()
    
    return fig, ax


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Basic recording with default colors
    print("=" * 60)
    print("Example 1: Recording with default colors")
    print("=" * 60)
    
    recorder = MatplotlibRecorder(
        output_folder="test_output",
        resolution=(1920, 1080)
    )
    
    recorder.start_recording()
    for i in range(10):
        fake_matrix = np.random.choice([0, 1], size=(100, 100))
        recorder.save_frame(fake_matrix, temperature=1.0 - i*0.1)
    recorder.stop_recording()
    
    recorder.generate_images(max_frames=10)
    
    # Example 2: Custom colors
    print("\n" + "=" * 60)
    print("Example 2: Custom colors")
    print("=" * 60)
    
    recorder2 = MatplotlibRecorder(
        output_folder="test_output_custom",
        resolution=(1920, 1080),
        colors=["#0000FF", "#FF0000"]  # Blue and Red
    )
    
    recorder2.start_recording()
    for i in range(5):
        fake_matrix = np.random.choice([0, 1], size=(50, 50))
        recorder2.save_frame(fake_matrix, temperature=2.0)
    recorder2.stop_recording()
    
    recorder2.generate_images()
    
    # Example 3: Independent plotting
    print("\n" + "=" * 60)
    print("Example 3: Independent plotting")
    print("=" * 60)
    
    matrix = np.random.choice([0, 1], size=(100, 100))
    
    # Save as high-res image
    plot_matrix_to_image(
        matrix, 
        "standalone_plot.png",
        colors=["#FFFF00", "#FF00FF"],  # Yellow and Magenta
        resolution=(3840, 2160)
    )
    
    # Or display interactively
    plot_matrix_matplotlib(
        matrix,
        colors=["#00FF00", "#0000FF"],  # Green and Blue
        title="My Ising Simulation",
        show=False  # Set to True to display
    )