"""Colors shared by the trajectory figures.

The court is orange, as indoor volleyball courts usually are, and it acts as
the surface the data marks sit on rather than as a data color itself.

The three data colors were checked with the dataviz palette validator against
the effective court surface (#F0BA97, i.e. the orange at its render opacity
over white), all-pairs:

    Lightness band      PASS   all inside L 0.43-0.77
    Chroma floor        PASS   all >= 0.1
    CVD separation      PASS   worst pair dE 12.7 (deutan), 11.0 (tritan)
    Normal-vision floor PASS   worst pair dE 29.3
    Contrast vs surface WARN   SERVE is 2.18:1 against the court

The obvious choice of a red reception marker was rejected: red against the
orange court measured dE 14.2 for *normal* vision, below the floor of 15.

Because of the contrast WARN, the serve and reception markers must keep their
visible text labels and outline rings - they carry a distinct symbol and a
label, so they are never distinguished by hue alone. Don't drop those.
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
SERVE = "#059669"       # serve contact point
RECEPTION = "#9D174D"   # reception contact point

# Ring drawn around the highlighted points so they lift off the court.
MARKER_OUTLINE = "#FFFFFF"
