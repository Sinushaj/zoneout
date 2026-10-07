"""Reading video frames for the calibration windows to show.

Shared by the court-point picker and the sync navigator, which both put one
frame of a match recording on a Tk canvas and let it be scrubbed. Nothing here
opens a window; it is only the decoding, kept in one place so the two windows
cannot end up disagreeing about what size a frame is.
"""

import cv2
from PIL import Image

# Frames are handed out at this size whatever the source is, and nothing
# downstream rescales them: `reconstruction`'s camera matrices are written for a
# 1920x1080 image, so a clicked pixel only means anything in that space.
FRAME_SIZE = (1920, 1080)


class VideoFrames:
    """One video, decoded a frame at a time for a calibration window to show.

    The capture is kept open for the life of the window rather than reopened
    per frame, and stepping to the next frame is a plain read: seeking into a
    long recording is what costs, so the common way of moving - one frame on -
    is the one that does not seek at all.
    """

    def __init__(self, video_file):
        self.capture = cv2.VideoCapture(video_file)
        if not self.capture.isOpened():
            raise IOError(f"Cannot open video file: {video_file}")

        self.count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.rate = self.capture.get(cv2.CAP_PROP_FPS) or 0.0
        # Where the capture will read next, so a step forward can skip the seek.
        self.position = 0

    def image(self, number):
        """Frame `number`, resized to the size the camera matrices assume."""
        if number != self.position:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, number)
            self.position = number

        read, frame = self.capture.read()
        if not read:
            # Leave the position alone: the capture is now somewhere unknown,
            # and the next request will seek rather than trust it.
            self.position = -1
            raise IOError(f"Cannot read frame {number}")

        self.position = number + 1
        frame = cv2.resize(frame, FRAME_SIZE)
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def close(self):
        self.capture.release()


def read_frame(video_file, frame_number=0):
    """One frame of a video, as a PIL image at the size the intrinsics assume."""
    frames = VideoFrames(video_file)
    try:
        return frames.image(frame_number)
    finally:
        frames.close()


def describe_time(frame_number, rate):
    """A frame number as a clock reading, for a window to put beside it.

    A frame number is what gets recorded, but it is not what anyone recognises
    in an hour-long recording; the time is how you tell whether you are in the
    right part of the match at all. A container that does not report its rate
    gets no reading rather than a wrong one.
    """
    if not rate:
        return ""

    seconds = frame_number / rate
    return (f"{int(seconds // 3600)}:{int(seconds // 60) % 60:02d}"
            f":{seconds % 60:05.2f}")
