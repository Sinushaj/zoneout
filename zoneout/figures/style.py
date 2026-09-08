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

# Ring drawn around the highlighted points so they lift off the court.
MARKER_OUTLINE = "#FFFFFF"
