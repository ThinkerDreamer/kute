"""Outlines the strokes of kute's tray icons and writes the three SVGs.

GTK recolours a symbolic icon by filling every shape, so the icons must have no strokes.
Run from the repository root: `python3 assets/icons/outline.py assets/icons` (needs shapely).
"""
import math, sys
from shapely.geometry import LineString
from shapely.ops import unary_union
from shapely.geometry.polygon import orient

def cubic(p0, p1, p2, p3, n=48):
    out = []
    for i in range(1, n + 1):
        t = i / n; u = 1 - t
        out.append((u**3*p0[0]+3*u*u*t*p1[0]+3*u*t*t*p2[0]+t**3*p3[0],
                    u**3*p0[1]+3*u*u*t*p1[1]+3*u*t*t*p2[1]+t**3*p3[1]))
    return out

def curve(start, segs):
    pts = [start]
    for c1, c2, end in segs:
        pts += cubic(pts[-1], c1, c2, end)
    return LineString(pts)

outer = curve((1.6, 7.8), [((1.6,3.8),(4.2,0.9),(7.6,0.9)), ((11,0.9),(13.2,3.5),(13.2,7)),
    ((13.2,10.2),(11.2,11.5),(10.2,12.9)), ((9.2,14.4),(9.2,16.6),(7.8,18.4)),
    ((7,19.5),(5.8,20),(4.7,20)), ((3.5,20),(2.5,19.4),(2,18.4))])
inner = curve((5.2, 7.9), [((5.2,6.2),(6.2,4.9),(7.6,4.9)), ((9,4.9),(9.8,6),(9.8,7.3)),
    ((9.8,8.8),(8.2,9.4),(7.8,10.8))])
# The stand: the lower half of a circle round (17.3, 15), r 3.8, then the post and the foot.
arc = LineString([(17.3 - 3.8*math.cos(math.pi*i/96), 15 + 3.8*math.sin(math.pi*i/96)) for i in range(97)])
post = LineString([(17.3, 18.8), (17.3, 20.9)])
foot = LineString([(15, 20.9), (19.6, 20.9)])

def stroke(line, width):
    return line.buffer(width / 2, quad_segs=24, cap_style="round", join_style="round")

ear = unary_union([stroke(outer, 1.6), stroke(inner, 1.6)])
stand = unary_union([stroke(arc, 1.4), stroke(post, 1.4), stroke(foot, 1.4)])

def path_d(geom):
    geom = geom.simplify(0.01)
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    parts = []
    for p in polys:
        p = orient(p)
        for ring in [p.exterior, *p.interiors]:
            cs = list(ring.coords)[:-1]
            fmt = lambda c: f"{round(c[0],2):g} {round(c[1],2):g}"
            parts.append("M" + fmt(cs[0]) + "L" + " ".join(fmt(c) for c in cs[1:]) + "Z")
    return "".join(parts)

STYLE = ('<style id="current-color-scheme" type="text/css">'
         '.ColorScheme-Text{color:#232629;}.ColorScheme-NegativeText{color:#da4453;}</style>')
MIC = '<rect x="15.2" y="8.4" width="4.2" height="9.2" rx="2.1" fill="currentColor" class="{cls}"/>'

def icon(mic_class, faded):
    body = [
        f'<path fill="currentColor" class="ColorScheme-Text" d="{path_d(ear)}"/>',
        MIC.replace("{cls}", mic_class),
        f'<path fill="currentColor" class="ColorScheme-Text" d="{path_d(stand)}"/>',
    ]
    if faded:
        body = ['<g opacity="0.5">', *body, '</g>']
    return "\n".join(['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 22 22" width="22" height="22">',
                      "  " + STYLE, *("  " + b for b in body), "</svg>", ""])

out = sys.argv[1]
open(f"{out}/kute-listening-symbolic.svg", "w").write(icon("ColorScheme-Text", False))
open(f"{out}/kute-recording-symbolic.svg", "w").write(icon("ColorScheme-NegativeText error", False))
open(f"{out}/kute-paused-symbolic.svg", "w").write(icon("ColorScheme-Text", True))
