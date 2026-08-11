"""Interactive tools for producing the CSV configuration files.

These open OpenCV/Tk windows and expect a human at the keyboard, so they are
kept out of the batch pipeline. Run them via `update_parameters.py`.
"""

from .points import get_screen_coordinates
from .sync import navigate_video

__all__ = ["get_screen_coordinates", "navigate_video"]
