"""Colors shared by the trajectory figures.

The court is orange, as indoor volleyball courts usually are, and it acts as
the surface the data marks sit on rather than as a data color itself.

The four data colors were checked with the dataviz palette validator against
the effective court surface (#F0BA97, i.e. the orange at its render opacity
over white), all-pairs:

    Lightness band      PASS   all inside L 0.43-0.77
    Chroma floor        PASS   all >= 0.1
    CVD separation      PASS   worst pair dE 9.2 (deutan), 7.4 (tritan)
    Normal-vision floor PASS   worst pair dE 18.3
    Contrast vs surface WARN   SERVE is 2.18:1 against the court

The obvious choice of a red reception marker was rejected: red against the
orange court measured dE 14.2 for *normal* vision, below the floor of 15.

RAW_PATH was hard to place, because the cool half of the wheel is where the
fitted path already lives: every violet, indigo and teal tried against PATH
came out at dE 5.7-13.7 for normal vision, under the floor of 15, and the
teals additionally read as gray (chroma below 0.1). The ochre is far enough
from all three of the other data colors and, despite sharing the court's hue
family, sits dE 27.9 from the court surface itself, so it does not sink into
it. It clears the surface at 2.85:1, just under the 3:1 line, which is why the
raw series must keep both the legend entry naming it and its distinct
markers-only shape.

Because of the contrast WARN, the serve and reception markers must keep their
visible text labels and outline rings - they carry a distinct symbol and a
label, so they are never distinguished by hue alone. Don't drop those.

ATTACK_END is the fifth color, added for the attack figure, which marks three
contacts rather than two. The other two came free: an attack figure never draws
a serve or a reception, so SERVE and RECEPTION were already unused there and
are reused for the set and the attack contacts under their own names below.
Only the third contact needed a new color.

Finding one was not a matter of taste. The four existing colors are dense: a
sweep of the sRGB cube against all five checks, plus this project's own extra
rule that a data color must sit at least dE 15 from the court surface, ruled
out every violet, indigo, teal and slate tried by hand - #6D28D9, the obvious
purple, lands dE 0.3 from PATH under deutan simulation. The deep purple that
survived was re-validated as a full five-slot palette:

    Lightness band      PASS   all inside L 0.43-0.77
    Chroma floor        PASS   all >= 0.1
    CVD separation      PASS   worst pair dE 9.2 (deutan) - unchanged
    Normal-vision floor PASS   worst pair dE 15.7, ATTACK_END vs RECEPTION
    Contrast vs surface WARN   unchanged: still only SERVE and RAW_PATH

ATTACK_END itself is dE 44.2 from the court and clears it at 5.06:1, better
than any other data color here, and its worst separation from an existing one
is dE 15.7 normal / 11.3 deutan. Adding it introduced no new warning. If a
sixth is ever needed, run the validator rather than picking one - there is very
little room left.
"""

# Court surface and its markings.
COURT_FILL = "#E0762F"
COURT_LINE = "#FFFFFF"

# Used by the interactive figure only, where you can rotate under the court and
# want to see through it. The static PNG draws the court opaque instead:
# matplotlib composites a translucent plane over geometry it thinks is behind
# it, which tinted the blue trajectory purple.
COURT_FILL_OPACITY = 0.55

# Data marks.
PATH = "#1D4ED8"        # the reconstructed ball trajectory
RAW_PATH = "#A16207"    # the raw triangulated points behind it, before fitting
SERVE = "#059669"       # serve contact point
RECEPTION = "#9D174D"   # reception contact point

# The three contacts an attack figure marks. The first two are the serve and
# reception colors under names that say what they mean here - neither a serve
# nor a reception appears in an attack clip, so those slots were free.
SET_CONTACT = SERVE       # the setter's touch
ATTACK_CONTACT = RECEPTION  # where the attacker struck the ball
ATTACK_END = "#673091"    # where the attack finished; see the note above

# Where the pass ended, in a *reception* figure. The serve and reception
# already hold the first two contact colors there, and ATTACK_END's slot is
# free - a reception figure never draws an attack - so this takes it, as the
# attack figure's set took the serve's. No new color, so nothing to re-validate.
RECEPTION_END = ATTACK_END

# The top of the pass, in a reception figure. Not a contact but a measurement
# read off one, so it is drawn in the net's neutral ink, as a different symbol
# with a dashed drop line to the floor and a label giving the height: told
# apart by shape and text, adding no sixth data color to a palette with very
# little room left (see above).
APEX = "#3D3D3A"

# Ring drawn around the highlighted points so they lift off the court.
MARKER_OUTLINE = "#FFFFFF"

# The net and antennas: structure, not data, so a neutral ink rather than a
# data color. Dark, because the net stands in front of the page background and
# the plotly scene's pale panels, where the court's white would vanish; the
# mesh is drawn translucent so paths behind it stay visible.
NET = "#3D3D3A"
NET_OPACITY = 0.18

# Serve speed, for the player serve figure: light cyan to deep violet, faster
# reads darker. Built in OKLCH with lightness falling evenly from 0.76 to 0.30
# while the hue turns from 215 to 300, so it stays a sequential ramp - one
# direction of lightness, no rainbow - and keeps to the cool side of the wheel,
# opposite the orange court.
#
# It replaced blue steps 400-700 of the dataviz reference ramp, whose two ends
# were only dE 29 apart and read as much the same blue. This one spans dE 49.
# The light end is the constraint: it has to stay visible against both surfaces
# a path is seen on, the court (#F0BA97) and the scene's pale panels (#E5ECF6).
# Starting at L 0.82 gave dE 55 of range but a light end of equal luminance to
# the court (1.02:1) and only dE 15.6 from the panels; at 0.76 the light end is
# dE 19.8 from the court and 20.5 from the panels, both over this project's
# floor of 15, at the cost of a little range. Luminance contrast against the
# court is still low at that end (1.2:1), so the slowest paths are told apart
# from the floor by hue - which is why the ramp stays cool.
SPEED_SCALE = [
    [0.0, "#4DC3DD"],
    [0.25, "#3698CB"],
    [0.5, "#396BB3"],
    [0.75, "#3F3F91"],
    [1.0, "#3B0F67"],
]

# The contact and reception dots on the serve figure: text ink, told apart by
# shape and legend rather than hue, so they add no color to a figure whose
# color already means speed.
SERVE_POINT = "#3D3D3A"
