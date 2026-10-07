"""Clicking the court reference points, with a diagram saying which one.

Produces the pixel coordinates stored in `gopro_points.csv` / `zve10_points.csv`.
The clicks are matched positionally with `zoneout.court.CALIBRATION_POINTS` when
they are handed to solvePnP, so the order is load-bearing and getting it wrong
does not fail — it silently solves for the wrong camera pose. The window
therefore asks for one point at a time and shows it on a court diagram
(`court_diagram.CourtDiagram`) rather than expecting the order to be
remembered.

The video frame fills the window and the diagram sits beside it in a fixed
panel, which is also where the magnifier and the list of points live. Under the
picture is a scrubber, because the frame to click is not always the first one -
a recording can open on a hand in front of the lens, and a court corner can be
stood on. The clicked pixels belong to the camera rather than to a frame, so
they stay put while the video is scrubbed and each point can be placed on
whichever frame shows it. Nothing here writes files: `pick_court_points` returns the pixels, or None if the
window was closed without saving, and the caller decides what to do with them.
"""

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from ..court import CALIBRATION_POINTS
from .court_diagram import (COURT_FILL, CURRENT_FILL, DONE_FILL, PANEL_BG,
                            CourtDiagram, describe_point)
from .frames import FRAME_SIZE, VideoFrames

# The side panel is fixed and the video takes everything else, so the picture
# is as large as the window allows with the diagram still beside it.
PANEL_WIDTH = 340
DIAGRAM_HEIGHT = 260
MAGNIFIER_SIZE = 180

# How much the magnifier enlarges. The reference points are line intersections
# and antenna tips a few pixels across, and a click is stored as a whole pixel,
# so being able to see the pixel being chosen is what makes the extra accuracy
# real rather than notional.
MAGNIFIER_ZOOM = 8


def pick_court_points(video_file, points=CALIBRATION_POINTS, title=None,
                      frame_number=0, initial=None):
    """Open the picker on one video and return one pixel per reference point.

    Args:
        video_file: the video to take the frame from.
        points: the 3D reference points to ask for, in order. Defaults to
            `court.CALIBRATION_POINTS`, which is the order the CSV files and
            solvePnP agree on.
        title: what to call this camera in the window title.
        frame_number: which frame to open on. The scrubber can be moved off
            it, and points clicked on different frames sit together.
        initial: pixels already on file, to be adjusted rather than re-clicked.

    Returns:
        A list of (x, y) integer pixels in 1920x1080 image coordinates, one per
        point in `points`, or None if the window was closed without saving.
    """
    frames = VideoFrames(video_file)
    try:
        return _PointPicker(frames, points, title or video_file, initial,
                            frame_number).run()
    finally:
        frames.close()


class _PointPicker:
    """The window. One instance per camera; `run` returns what was clicked."""

    def __init__(self, frames, points, title, initial=None, frame_number=0):
        self.frames = frames
        self.frame_number = frame_number
        self.image = frames.image(frame_number)
        self.points = list(points)
        self.placed = list(initial) if initial else [None] * len(self.points)
        if len(self.placed) != len(self.points):
            raise ValueError(
                f"got {len(self.placed)} starting pixels for "
                f"{len(self.points)} reference points")

        self.current = self._next_unplaced(-1) or 0
        self.result = None

        # Set by _render_frame, and needed to turn a click back into a pixel of
        # the source image.
        self.scale = 1.0
        self.offset = (0, 0)
        self.photo = None
        self.magnifier_photo = None

        self._build(title)

    # -- window ----------------------------------------------------------

    def _build(self, title):
        self.root = tk.Tk()
        self.root.title(f"Court calibration — {title}")
        self.root.minsize(900, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._cancel)
        self._fill_screen()

        panel = ttk.Frame(self.root, padding=10)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        panel.configure(width=PANEL_WIDTH)

        # The picture and its scrubber share what the panel leaves, the
        # scrubber taking a single row at the foot of it.
        left = ttk.Frame(self.root)
        left.pack(side="left", fill="both", expand=True)
        self._build_scrubber(left)

        self.video = tk.Canvas(left, bg="#111111", highlightthickness=0,
                               cursor="tcross")
        self.video.pack(fill="both", expand=True)
        self.video.bind("<Configure>", lambda event: self._render_frame())
        self.video.bind("<Button-1>", self._on_click)
        self.video.bind("<Motion>", self._on_motion)
        self.video.bind("<Leave>", lambda event: self._clear_magnifier())

        self._build_panel(panel)
        self._bind_keys()
        self._refresh()

    def _fill_screen(self):
        """Open as large as the screen allows.

        Every pixel not in the side panel is picture, and a click is only as
        accurate as the picture is big: at half size one screen pixel is two
        source pixels, and these points are line intersections.
        """
        try:
            self.root.attributes("-zoomed", True)
        except tk.TclError:
            # Not every window manager supports that; size it by hand instead.
            self.root.geometry(f"{self.root.winfo_screenwidth() - 80}"
                               f"x{self.root.winfo_screenheight() - 120}+20+20")

    def _build_scrubber(self, parent):
        """The row that moves the video, or nothing at all for a single frame.

        A frame's number is not recorded anywhere: the clicked pixels are a
        property of where the camera stands, so which frame each was clicked on
        does not matter to anything downstream. It only has to show the point.
        """
        self.slider = None
        if self.frames.count <= 1:
            return

        row = ttk.Frame(parent, padding=(6, 4))
        row.pack(side="bottom", fill="x")

        # A second's worth of frames, for the coarse step. A container that
        # does not report its rate gets a sane stride rather than a step of 0.
        self.second = int(round(self.frames.rate)) or 60

        for text, step in (("<<", -self.second), ("<", -1)):
            ttk.Button(row, text=text, width=3,
                       command=lambda step=step: self._step(step)).pack(side="left")

        # The readout is built before the slider and packed from the right, so
        # that setting the slider's starting value - which calls straight back
        # into _on_slider - has a label to write to.
        self.frame_label = ttk.Label(row, font=("TkFixedFont", 9), width=20,
                                     anchor="e")
        self.frame_label.pack(side="right", padx=(8, 0))
        self._show_frame_label(self.frame_number)

        for text, step in ((">>", self.second), (">", 1)):
            ttk.Button(row, text=text, width=3,
                       command=lambda step=step: self._step(step)).pack(side="right")

        self.slider = ttk.Scale(row, from_=0, to=self.frames.count - 1,
                                orient="horizontal", command=self._on_slider)
        self.slider.set(self.frame_number)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        # Only on release: dragging would otherwise seek on every pixel of
        # travel, and a seek into a long recording is the expensive move.
        self.slider.bind("<ButtonRelease-1>",
                         lambda event: self._show_frame(round(self.slider.get())))

    def _build_panel(self, panel):
        self.diagram_canvas = tk.Canvas(panel, height=DIAGRAM_HEIGHT, bg=PANEL_BG,
                                        highlightthickness=1,
                                        highlightbackground="#D1D5DB")
        # Expands into whatever the panel has left over: the diagram is the
        # thing being read at every click, so spare height belongs to it.
        self.diagram_canvas.pack(fill="both", expand=True)
        self.diagram_canvas.bind("<Configure>", lambda event: self._draw_diagram())
        self.diagram = CourtDiagram(self.diagram_canvas, self.points)

        ttk.Button(panel, text="Turn the diagram (r)",
                   command=self._rotate).pack(fill="x", pady=(4, 8))

        self.heading = ttk.Label(panel, font=("TkDefaultFont", 11, "bold"))
        self.heading.pack(fill="x")
        self.description = ttk.Label(panel, wraplength=PANEL_WIDTH - 24,
                                     font=("TkDefaultFont", 10))
        self.description.pack(fill="x", pady=(2, 8))

        self.listbox = tk.Listbox(panel, height=len(self.points),
                                  font=("TkFixedFont", 9), activestyle="none",
                                  exportselection=False)
        self.listbox.pack(fill="x")
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        self.magnifier = tk.Canvas(panel, width=MAGNIFIER_SIZE,
                                   height=MAGNIFIER_SIZE, bg="#111111",
                                   highlightthickness=1,
                                   highlightbackground="#D1D5DB")
        self.magnifier.pack(pady=(10, 4))
        self.cursor_label = ttk.Label(panel, font=("TkFixedFont", 9))
        self.cursor_label.pack(fill="x")

        # Packed bottom-up, so the actions keep the foot of the panel: a window
        # too short for everything then squeezes the magnifier rather than
        # hiding the way out.
        ttk.Label(panel, font=("TkDefaultFont", 8), foreground="#4B5563",
                  wraplength=PANEL_WIDTH - 24,
                  text=("Click the highlighted point. Pick a row to redo one,"
                        " arrow keys nudge it a pixel at a time.")).pack(
            side="bottom", fill="x", pady=(8, 0))
        ttk.Button(panel, text="Cancel (esc)", command=self._cancel).pack(
            side="bottom", fill="x", pady=(4, 0))
        self.save_button = ttk.Button(panel, text="Save (enter)", command=self._save)
        self.save_button.pack(side="bottom", fill="x", pady=(6, 0))

        buttons = ttk.Frame(panel)
        buttons.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Undo (u)", command=self._undo).pack(
            side="left", expand=True, fill="x")
        ttk.Button(buttons, text="Clear all", command=self._clear).pack(
            side="left", expand=True, fill="x")

    def _bind_keys(self):
        self.root.bind("<Escape>", lambda event: self._cancel())
        self.root.bind("<Return>", lambda event: self._save())
        self.root.bind("u", lambda event: self._undo())
        self.root.bind("<Control-z>", lambda event: self._undo())
        self.root.bind("r", lambda event: self._rotate())
        for key, dx, dy in (("Left", -1, 0), ("Right", 1, 0),
                            ("Up", 0, -1), ("Down", 0, 1)):
            self.root.bind(f"<{key}>",
                           lambda event, dx=dx, dy=dy: self._nudge(dx, dy))
            self.root.bind(f"<Shift-{key}>",
                           lambda event, dx=dx, dy=dy: self._nudge(dx * 10, dy * 10))

    def run(self):
        self.root.mainloop()
        return self.result

    # -- state -----------------------------------------------------------

    def _next_unplaced(self, after):
        """The next point still to click, searching on from `after` and wrapping."""
        count = len(self.points)
        for step in range(1, count + 1):
            index = (after + step) % count
            if self.placed[index] is None:
                return index
        return None

    def _on_click(self, event):
        pixel = self._to_source(event.x, event.y)
        if pixel is None:
            return

        self.placed[self.current] = pixel
        remaining = self._next_unplaced(self.current)
        if remaining is not None:
            self.current = remaining
        self._refresh()

    def _on_select(self, event):
        """A row picked in the list becomes the point being placed.

        Redrawing rebuilds the list and re-selects the current row, so this can
        be reached from the redraw rather than from a person. Doing nothing
        when the selection already agrees keeps that from recursing.
        """
        selection = self.listbox.curselection()
        if selection and selection[0] != self.current:
            self.current = selection[0]
            self._refresh()

    def _undo(self):
        """Clear the selected point, or the last one placed before it."""
        if self.placed[self.current] is not None:
            index = self.current
        else:
            index = next((i for i in range(len(self.points) - 1, -1, -1)
                          if self.placed[i] is not None), None)
        if index is None:
            return

        self.placed[index] = None
        self.current = index
        self._refresh()

    def _clear(self):
        self.placed = [None] * len(self.points)
        self.current = 0
        self._refresh()

    def _nudge(self, dx, dy):
        pixel = self.placed[self.current]
        if pixel is None:
            return

        self.placed[self.current] = (
            min(max(pixel[0] + dx, 0), FRAME_SIZE[0] - 1),
            min(max(pixel[1] + dy, 0), FRAME_SIZE[1] - 1))
        self._refresh()

    def _rotate(self):
        self.diagram.rotate()
        self._draw_diagram()

    def _step(self, delta):
        self._show_frame(self.frame_number + delta)

    def _on_slider(self, value):
        """Dragging only reads out where it has got to; the frame waits."""
        self._show_frame_label(round(float(value)))

    def _show_frame(self, number):
        """Put a different frame of the video up, keeping the placed points."""
        number = min(max(int(number), 0), self.frames.count - 1)
        if number == self.frame_number:
            self._show_frame_label(number)
            return

        try:
            self.image = self.frames.image(number)
        except IOError:
            # A frame the container claims but cannot decode: stay where we are
            # rather than leaving the window showing one frame and saying another.
            self._show_frame_label(self.frame_number)
            self.slider.set(self.frame_number)
            return

        self.frame_number = number
        self.slider.set(number)
        self._show_frame_label(number)
        self._render_frame()
        self._clear_magnifier()

    def _show_frame_label(self, number):
        self.frame_label.config(text=f"frame {number} / {self.frames.count - 1}")

    def _save(self):
        if any(pixel is None for pixel in self.placed):
            return
        self.result = list(self.placed)
        self.root.destroy()

    def _cancel(self):
        self.result = None
        self.root.destroy()

    # -- drawing ---------------------------------------------------------

    def _refresh(self):
        self._draw_diagram()
        self._draw_markers()
        self._draw_panel_text()

    def _draw_diagram(self):
        self.diagram.draw(self.placed, self.current)

    def _draw_panel_text(self):
        point = self.points[self.current]
        done = sum(pixel is not None for pixel in self.placed)

        self.heading.config(
            text=f"Point {self.current + 1} of {len(self.points)}"
                 f"   ({done} placed)")
        coordinate = "(" + ", ".join(f"{value:g}" for value in point) + ")"
        self.description.config(text=f"{describe_point(point)}\n{coordinate} m")

        self.listbox.delete(0, tk.END)
        for index, (reference, pixel) in enumerate(zip(self.points, self.placed)):
            mark = "✓" if pixel is not None else "·"
            coordinates = ",".join(f"{value:g}" for value in reference)
            shown = f"{pixel[0]:>4},{pixel[1]:>4}" if pixel else "   —,   —"
            self.listbox.insert(
                tk.END, f" {index + 1} {mark} ({coordinates:>12})  {shown}")
            if pixel is not None:
                self.listbox.itemconfig(index, foreground=DONE_FILL)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.current)

        self.save_button.state(
            ["!disabled"] if done == len(self.points) else ["disabled"])

    def _render_frame(self):
        """Fit the frame to the canvas, as large as it goes, and redraw markers."""
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
        self.video.tag_lower("frame")
        self._draw_markers()

    def _draw_markers(self):
        canvas = self.video
        canvas.delete("marker")

        for index, pixel in enumerate(self.placed):
            if pixel is None:
                continue

            x, y = self._to_canvas(*pixel)
            current = index == self.current
            color = CURRENT_FILL if current else DONE_FILL
            reach = 12 if current else 9

            # A cross rather than a filled dot: the point being marked is a
            # line intersection a couple of pixels across, and a dot would
            # cover the very thing that has to be checked.
            canvas.create_line(x - reach, y, x - 3, y, fill=color, width=2,
                               tags="marker")
            canvas.create_line(x + 3, y, x + reach, y, fill=color, width=2,
                               tags="marker")
            canvas.create_line(x, y - reach, x, y - 3, fill=color, width=2,
                               tags="marker")
            canvas.create_line(x, y + 3, x, y + reach, fill=color, width=2,
                               tags="marker")
            if current:
                canvas.create_oval(x - reach, y - reach, x + reach, y + reach,
                                   outline=color, width=2, tags="marker")
            canvas.create_text(x + reach + 7, y - reach - 2, text=str(index + 1),
                               fill=color, anchor="w", tags="marker",
                               font=("TkDefaultFont", 10, "bold"))

    def _on_motion(self, event):
        pixel = self._to_source(event.x, event.y)
        if pixel is None:
            self._clear_magnifier()
            return

        half = MAGNIFIER_SIZE // (2 * MAGNIFIER_ZOOM)
        crop = self.image.crop((pixel[0] - half, pixel[1] - half,
                                pixel[0] + half, pixel[1] + half))
        # NEAREST, so the pixel grid stays visible at this zoom rather than
        # being blurred into a guess.
        self.magnifier_photo = ImageTk.PhotoImage(
            crop.resize((MAGNIFIER_SIZE, MAGNIFIER_SIZE), Image.NEAREST))

        self.magnifier.delete("all")
        self.magnifier.create_image(0, 0, anchor="nw", image=self.magnifier_photo)
        middle = MAGNIFIER_SIZE // 2
        self.magnifier.create_line(0, middle, MAGNIFIER_SIZE, middle,
                                   fill=COURT_FILL)
        self.magnifier.create_line(middle, 0, middle, MAGNIFIER_SIZE,
                                   fill=COURT_FILL)
        self.cursor_label.config(text=f"cursor  {pixel[0]:>4}, {pixel[1]:>4} px")

    def _clear_magnifier(self):
        self.magnifier.delete("all")
        self.cursor_label.config(text="")

    # -- coordinates -----------------------------------------------------

    def _to_source(self, canvas_x, canvas_y):
        """Canvas position -> whole pixel of the 1920x1080 frame, or None."""
        x = round((canvas_x - self.offset[0]) / self.scale)
        y = round((canvas_y - self.offset[1]) / self.scale)
        if 0 <= x < FRAME_SIZE[0] and 0 <= y < FRAME_SIZE[1]:
            return x, y
        return None

    def _to_canvas(self, x, y):
        return (x * self.scale + self.offset[0], y * self.scale + self.offset[1])
