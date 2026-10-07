"""Finding, by eye, the frame a camera shows a known action at.

`navigate_video` is what `update_parameters.py` uses to record the sync anchor:
the one pair of frame numbers every clip in a run is an offset from. It is a
window rather than a printed list of keys because the action being looked for
can be an hour into a recording, and stepping there a minute at a time is not a
reasonable thing to ask of anyone — so the picture fills the window, a scrubber
under it covers the whole recording in one drag, and the keys that were always
there still step frame by frame once the right rally is on screen.

The keys are the ones this project has always used (j/f, k/d, l/s, q); the
scrubber, the mouse wheel and the arrow keys are additions, not replacements.

Closing the window without accepting a frame returns None, so an abandoned sync
leaves `first_reception_frames.csv` alone rather than writing whichever frame
happened to be showing. That matters more here than anywhere else in the
calibration: a wrong anchor fails nothing, it cuts every clip of the run from
the wrong minute of the match.

`sync_two_videos` is an older standalone helper for working out the frame
mapping between two cameras by eye, and still draws its own OpenCV windows.

Nothing here writes files; the caller decides what to do with the frame.
"""

import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from .frames import FRAME_SIZE, VideoFrames, describe_time

# The side panel is fixed and the video takes everything else.
PANEL_WIDTH = 320

# How long the scrubber has to sit still before the frame under it is decoded.
# Dragging otherwise asks for one seek per pixel of travel, and a seek into a
# 33 GB recording is the expensive move (~0.3 s); waiting for the drag to pause
# means at most one seek per place the person actually stops to look.
SCRUB_SETTLE_MS = 120

# What each key does, in the order the panel lists them. The letters are the
# ones this project has used since the OpenCV version of this window, so a
# person who knows them does not have to learn anything to keep using them.
KEYS = (
    ("j / f", "one frame on / back"),
    ("k / d", "one second on / back"),
    ("l / s", "one minute on / back"),
    ("→ / ←", "one frame, shift for a second"),
    ("wheel", "scroll the video, shift for a second"),
    ("q, enter", "use the frame on screen"),
    ("esc", "cancel, leaving the anchor alone"),
)


def navigate_video(file_path, start_frame=0, title=None, note=None):
    """Scrub a video by hand and return the frame accepted, or None.

    Args:
        file_path: the recording to scrub.
        start_frame: where to open. Only where the scrubbing starts, never an
            answer: `update_parameters.py` seeds it from the previous anchor,
            which lands near the action when that sync was in this recording
            and somewhere useless when it was not.
        title: what to call this camera in the window title.
        note: what is being looked for, shown in the panel. The window is open
            for as long as it takes to find one rally in an hour of video, so
            it says which rally rather than leaving that in the terminal.

    Returns:
        The frame number stopped on, or None if the window was closed or
        cancelled without accepting one.
    """
    frames = VideoFrames(file_path)
    try:
        return _Navigator(frames, title or file_path, start_frame, note).run()
    finally:
        frames.close()


class _Navigator:
    """The window. One instance per camera; `run` returns the frame or None."""

    def __init__(self, frames, title, start_frame=0, note=None):
        self.frames = frames
        self.start_frame = min(max(int(start_frame), 0), frames.count - 1)
        self.frame_number = self.start_frame
        self.image = frames.image(self.frame_number)
        self.note = note
        self.result = None

        # A second's worth of frames, and a minute's. A container that does not
        # report its rate gets a sane stride rather than a step of zero.
        self.second = int(round(frames.rate)) or 60
        self.minute = self.second * 60

        # Set by _render_frame; the picture is fitted to whatever the canvas is.
        self.scale = 1.0
        self.offset = (0, 0)
        self.photo = None
        self.pending_scrub = None

        self._build(title)

    # -- window ----------------------------------------------------------

    def _build(self, title):
        self.root = tk.Tk()
        self.root.title(f"Sync — {title}")
        self.root.minsize(900, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._cancel)
        self._fill_screen()

        panel = ttk.Frame(self.root, padding=10)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        panel.configure(width=PANEL_WIDTH)

        left = ttk.Frame(self.root)
        left.pack(side="left", fill="both", expand=True)
        self._build_scrubber(left)

        self.video = tk.Canvas(left, bg="#111111", highlightthickness=0)
        self.video.pack(fill="both", expand=True)
        self.video.bind("<Configure>", lambda event: self._render_frame())

        self._build_panel(panel)
        self._bind_keys()
        self._show_readout(self.frame_number)

    def _fill_screen(self):
        """Open as large as the screen allows.

        The frame a serve is struck on is told apart from its neighbours by
        where the ball is, which is a few pixels of a 1920x1080 picture.
        """
        try:
            self.root.attributes("-zoomed", True)
        except tk.TclError:
            # Not every window manager supports that; size it by hand instead.
            self.root.geometry(f"{self.root.winfo_screenwidth() - 80}"
                               f"x{self.root.winfo_screenheight() - 120}+20+20")

    def _build_scrubber(self, parent):
        """The row under the picture that moves through the whole recording."""
        row = ttk.Frame(parent, padding=(6, 4))
        row.pack(side="bottom", fill="x")

        for text, step in (("<<", -self.minute), ("<", -self.second)):
            ttk.Button(row, text=text, width=3, takefocus=False,
                       command=lambda step=step: self._step(step)).pack(side="left")

        # The readout is built before the slider and packed from the right, so
        # that setting the slider's starting value - which calls straight back
        # into _on_slider - has a label to write to.
        self.frame_label = ttk.Label(row, font=("TkFixedFont", 9), width=30,
                                     anchor="e")
        self.frame_label.pack(side="right", padx=(8, 0))

        for text, step in ((">>", self.minute), (">", self.second)):
            ttk.Button(row, text=text, width=3, takefocus=False,
                       command=lambda step=step: self._step(step)).pack(side="right")

        self.slider = ttk.Scale(row, from_=0, to=max(self.frames.count - 1, 1),
                                orient="horizontal", takefocus=False,
                                command=self._on_slider)
        self.slider.set(self.frame_number)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)

    def _build_panel(self, panel):
        if self.note:
            ttk.Label(panel, text=self.note, wraplength=PANEL_WIDTH - 24,
                      justify="left", font=("TkDefaultFont", 10)).pack(
                fill="x", pady=(0, 10))

        self.heading = ttk.Label(panel, font=("TkFixedFont", 13, "bold"),
                                 anchor="w")
        self.heading.pack(fill="x")
        self.time_label = ttk.Label(panel, font=("TkFixedFont", 10), anchor="w",
                                    foreground="#4B5563")
        self.time_label.pack(fill="x", pady=(2, 0))
        self.offset_label = ttk.Label(panel, font=("TkFixedFont", 9), anchor="w",
                                      foreground="#4B5563")
        self.offset_label.pack(fill="x", pady=(2, 12))

        keys = ttk.Frame(panel)
        keys.pack(fill="x")
        for row, (key, what) in enumerate(KEYS):
            ttk.Label(keys, text=key, font=("TkFixedFont", 9, "bold"),
                      anchor="w").grid(row=row, column=0, sticky="w", padx=(0, 8))
            ttk.Label(keys, text=what, font=("TkDefaultFont", 9),
                      anchor="w").grid(row=row, column=1, sticky="w")

        # Packed bottom-up, so the way out keeps the foot of the panel.
        ttk.Button(panel, text="Cancel (esc)", takefocus=False,
                   command=self._cancel).pack(side="bottom", fill="x", pady=(4, 0))
        ttk.Button(panel, text="Use this frame (enter)", takefocus=False,
                   command=self._accept).pack(side="bottom", fill="x")

    def _bind_keys(self):
        for key, step in (("j", 1), ("f", -1),
                          ("k", self.second), ("d", -self.second),
                          ("l", self.minute), ("s", -self.minute)):
            self.root.bind(key, lambda event, step=step: self._step(step))

        for key, direction in (("Right", 1), ("Left", -1)):
            self.root.bind(f"<{key}>",
                           lambda event, d=direction: self._step(d))
            self.root.bind(f"<Shift-{key}>",
                           lambda event, d=direction: self._step(d * self.second))

        self.root.bind("q", lambda event: self._accept())
        self.root.bind("<Return>", lambda event: self._accept())
        self.root.bind("<Escape>", lambda event: self._cancel())

        # The wheel is two buttons on X11 and one event with a sign elsewhere,
        # so both are bound rather than assuming a platform.
        self.root.bind("<Button-4>", lambda event: self._on_wheel(event, 1))
        self.root.bind("<Button-5>", lambda event: self._on_wheel(event, -1))
        self.root.bind("<MouseWheel>",
                       lambda event: self._on_wheel(event, 1 if event.delta > 0 else -1))

    def run(self):
        self.root.mainloop()
        return self.result

    # -- moving ----------------------------------------------------------

    def _on_wheel(self, event, direction):
        """One notch of the wheel, a frame at a time unless shift is held.

        Shift is bit 0 of the event's state, whatever the keyboard layout, so
        it is read off the event rather than bound as a separate sequence -
        X11 reports a shifted wheel as the same two buttons.
        """
        step = self.second if event.state & 0x0001 else 1
        self._step(direction * step)

    def _step(self, delta):
        self._show_frame(self.frame_number + delta)

    def _on_slider(self, value):
        """Dragging reads out where it has got to; the frame follows on a pause.

        Decoding on every step of the drag would seek per pixel of travel, and
        each seek stalls the window, so the scrubber would fight back. Waiting
        for the drag to settle keeps it smooth and still shows every place the
        person stops.
        """
        number = min(max(round(float(value)), 0), self.frames.count - 1)
        if number == self.frame_number:
            return

        self._show_readout(number)
        if self.pending_scrub is not None:
            self.root.after_cancel(self.pending_scrub)
        self.pending_scrub = self.root.after(
            SCRUB_SETTLE_MS, lambda: self._show_frame(number, from_slider=True))

    def _show_frame(self, number, from_slider=False):
        """Put a different frame of the video up.

        Any scrub still waiting to be decoded is dropped: a key pressed while
        the scrubber is settling means the person has moved on, and letting the
        old callback fire would jump the video back under them.
        """
        if self.pending_scrub is not None:
            self.root.after_cancel(self.pending_scrub)
            self.pending_scrub = None

        number = min(max(int(number), 0), self.frames.count - 1)
        if number == self.frame_number:
            self._show_readout(number)
            return

        try:
            self.image = self.frames.image(number)
        except IOError:
            # A frame the container claims but cannot decode: stay where we are
            # rather than leaving the window showing one frame and saying another.
            self.slider.set(self.frame_number)
            self._show_readout(self.frame_number)
            return

        self.frame_number = number
        if not from_slider:
            self.slider.set(number)
        self._show_readout(number)
        self._render_frame()

    def _show_readout(self, number):
        self.frame_label.config(
            text=f"frame {number} / {self.frames.count - 1}"
                 f"   {describe_time(number, self.frames.rate)}")
        self.heading.config(text=f"frame {number}")
        self.time_label.config(text=describe_time(number, self.frames.rate))

        moved = number - self.start_frame
        if moved and self.frames.rate:
            self.offset_label.config(
                text=f"{moved:+d} frames ({moved / self.frames.rate:+.1f} s)"
                     " from where this opened")
        else:
            self.offset_label.config(text="")

    # -- leaving ---------------------------------------------------------

    def _accept(self):
        self.result = self.frame_number
        self.root.destroy()

    def _cancel(self):
        self.result = None
        self.root.destroy()

    # -- drawing ---------------------------------------------------------

    def _render_frame(self):
        """Fit the frame to the canvas, as large as it goes."""
        width = self.video.winfo_width()
        height = self.video.winfo_height()
        if width < 10 or height < 10:
            return

        self.scale = min(width / FRAME_SIZE[0], height / FRAME_SIZE[1])
        size = (max(int(FRAME_SIZE[0] * self.scale), 1),
                max(int(FRAME_SIZE[1] * self.scale), 1))
        self.offset = ((width - size[0]) // 2, (height - size[1]) // 2)

        # Kept on the instance: Tk drops an image the moment nothing references it.
        self.photo = ImageTk.PhotoImage(self.image.resize(size, Image.BILINEAR))
        self.video.delete("frame")
        self.video.create_image(self.offset, anchor="nw", image=self.photo,
                                tags="frame")


def sync_two_videos(video1_path, video2_path, fps1=60.0, fps2=60.0):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one of the video files.")
        return

    # Get total frame counts for boundary checks
    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    # Current frame indices
    frame1_idx = 0
    frame2_idx = 0

    selected_pair = None

    cv2.namedWindow('Video 1 (25 fps)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Video 2 (60 fps)', cv2.WINDOW_NORMAL)

    print("\nControls:")
    print("  Video 1: a (prev) / d (next)")
    print("  Video 2: z (prev) / c (next)")
    print("  s       – select current pair and finish")
    print("  q       – quit without selecting\n")

    while True:
        # Seek to current frame indices
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame1_idx)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame2_idx)

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Reached end of a video or cannot read frame.")
            break

        # Overlay frame numbers
        cv2.putText(frame1, f"Frame: {frame1_idx}/{total_frames1-1}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame2, f"Frame: {frame2_idx}/{total_frames2-1}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video 1 (25 fps)', frame1)
        cv2.imshow('Video 2 (60 fps)', frame2)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('u'):          # Video 1 previous frame
            frame1_idx = max(0, frame1_idx - 1)
        elif key == ord('i'):          # Video 1 next frame
            frame1_idx = min(total_frames1 - 1, frame1_idx + 1)
        elif key == ord('o'):          # Video 2 previous frame
            frame2_idx = max(0, frame2_idx - 1)
        elif key == ord('p'):          # Video 2 next frame
            frame2_idx = min(total_frames2 - 1, frame2_idx + 1)

        elif key == ord('j'):          # Video 1 previous second
            frame1_idx = max(0, frame1_idx - 25)
        elif key == ord('k'):          # Video 1 next second
            frame1_idx = min(total_frames1 - 1, frame1_idx + 25)
        elif key == ord('l'):          # Video 2 previous second
            frame2_idx = max(0, frame2_idx - 60)
        elif key == ord('ö'):          # Video 2 next second
            frame2_idx = min(total_frames2 - 1, frame2_idx + 60)

        elif key == ord('n'):          # Video 1 previous minute
            frame1_idx = max(0, frame1_idx - 25 * 60)
        elif key == ord('m'):          # Video 1 next minute
            frame1_idx = min(total_frames1 - 1, frame1_idx + 25 * 60)
        elif key == ord(','):          # Video 2 previous minute
            frame2_idx = max(0, frame2_idx - 60 * 60)
        elif key == ord('.'):          # Video 2 next minute
            frame2_idx = min(total_frames2 - 1, frame2_idx + 60 * 60)

        elif key == ord('s'):
            selected_pair = (frame1_idx, frame2_idx)
            print(f"\nSelected pair: video1 frame {frame1_idx}, video2 frame {frame2_idx}")
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    if selected_pair is not None:
        f1, f2 = selected_pair
        a = fps2 / fps1          # = 2.4
        b = f2 - a * f1
        print("\nMapping from video1 frame to video2 frame:")
        print(f"f2 = {a} * f1 + {b}")
        print(f"Python function: lambda f1: int(round({a} * f1 + {b}))")
    else:
        print("No pair selected.")
