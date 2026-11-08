#!/usr/bin/env python3
"""
Figure 7 — Four-Dimensional Temporal Hypergraph Structure
Manuscript figure (BMC MIDM) — knowledge-graph style layout with cross-dimension links and an infographic stats panel.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, Polygon, Rectangle, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe
import numpy as np
import os

# ========== Font ==========
import matplotlib.font_manager as fm
for _fp in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        break
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'

# ========== Data ==========
N_PATIENTS = 253
DIMS = {
    'History':   {'n': 12,  'types': 'Symptom / Diagnosis',   'color': '#3B7DD8', 'light': '#D6E6FB', 'assoc': 1313, 'icon': 'H'},
    'Evidence':  {'n': 144, 'types': 'Lab / Imaging',          'color': '#2CA02C', 'light': '#D5F5D5', 'assoc': 1535, 'icon': 'E'},
    'Treatment': {'n': 9,   'types': 'Medication',             'color': '#D62728', 'light': '#FDDCDC', 'assoc': 775,  'icon': 'T'},
    'Severity':  {'n': 17,  'types': 'Score / Severity',       'color': '#9467BD', 'light': '#EDE3F6', 'assoc': 636,  'icon': 'S'},
}
TOTAL_ENTITIES = 182
TOTAL_HYPEREDGES = 1378
TOTAL_ASSOC = sum(d['assoc'] for d in DIMS.values())
WINDOW = 5
MEAN_DAYS = 2.9
COVERAGE = 58.9

BASE = os.environ.get('EXPERIMENT_BASE', os.getcwd())
OUTPUT_PATH = os.environ.get('FIG_OUT', f'{BASE}/Figure7_4D_Hypergraph_Structure.png')
DPI = 300

C_BG = '#FAFBFC'
C_DARK = '#2C2C2C'
C_MID = '#555555'
C_LIGHT = '#999999'
C_WHITE = '#FFFFFF'

# ========== Helper: smooth bezier connector ==========
def draw_bezier(ax, p1, p2, color, alpha=0.25, lw=0.6, zorder=0, curvature=0.15):
    """Draw a smooth bezier curve between two points."""
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = np.sqrt(dx**2 + dy**2)
    # Control point offset perpendicular to the line
    if dist > 0:
        nx = -dy / dist
        ny = dx / dist
        offset = dist * curvature
        ctrl_x = mid_x + nx * offset
        ctrl_y = mid_y + ny * offset
    else:
        ctrl_x, ctrl_y = mid_x, mid_y
    
    verts = [
        (p1[0], p1[1]),
        (ctrl_x, ctrl_y),
        (p2[0], p2[1]),
    ]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)
    patch = matplotlib.patches.PathPatch(path, facecolor='none', edgecolor=color,
                                          alpha=alpha, linewidth=lw, zorder=zorder)
    ax.add_patch(patch)

def draw_bezier_pair(ax, p1, p2, color, alpha=0.25, lw=0.6, zorder=0, curvature=0.15):
    """Draw symmetric bezier curves between two points (both directions for visual balance)."""
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 0.1:
        return
    nx = -dy / dist
    ny = dx / dist
    offset = dist * curvature
    
    # Curve bending one way
    ctrl_x1 = mid_x + nx * offset
    ctrl_y1 = mid_y + ny * offset
    verts1 = [(p1[0], p1[1]), (ctrl_x1, ctrl_y1), (p2[0], p2[1])]
    codes1 = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path1 = Path(verts1, codes1)
    ax.add_patch(matplotlib.patches.PathPatch(path1, facecolor='none', edgecolor=color,
                  alpha=alpha, linewidth=lw, zorder=zorder))
    
    # Curve bending the other way
    ctrl_x2 = mid_x - nx * offset
    ctrl_y2 = mid_y - ny * offset
    verts2 = [(p1[0], p1[1]), (ctrl_x2, ctrl_y2), (p2[0], p2[1])]
    codes2 = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path2 = Path(verts2, codes2)
    ax.add_patch(matplotlib.patches.PathPatch(path2, facecolor='none', edgecolor=color,
                  alpha=alpha, linewidth=lw, zorder=zorder))


# ========== CREATE FIGURE ==========
fig = plt.figure(figsize=(16, 7.5), dpi=DPI, facecolor=C_WHITE)

# ==================== PANEL (a): Network-style Hypergraph ====================
ax_a = fig.add_axes([0.02, 0.06, 0.52, 0.90])
ax_a.set_xlim(-5.8, 5.8)
ax_a.set_ylim(-5.8, 5.8)
ax_a.set_aspect('equal')
ax_a.set_facecolor(C_BG)
ax_a.axis('off')

ax_a.text(0, 5.5, '(a)  Per-Patient 4D Temporal Hypergraph',
          fontsize=11, fontweight='bold', color=C_DARK, ha='center', va='center')

# Central patient node
center = (0, 0)
ax_a.add_patch(Circle(center, 0.55, facecolor=C_DARK, edgecolor='#1A1A1A', linewidth=2.5, zorder=30))
ax_a.add_patch(Circle(center, 0.40, facecolor='#3D3D3D', edgecolor='none', zorder=31))
ax_a.text(0, 0.05, 'Patient', ha='center', va='center', fontsize=7.5, color='white', fontweight='bold', zorder=32)
ax_a.text(0, -0.22, 'Node', ha='center', va='center', fontsize=6, color='#BBBBBB', zorder=32)

# Dimension positions & radii (proportional to sqrt(real_n) for visual balance)
dim_centers = {
    'History':   (-3.8, 2.6),
    'Evidence':  (3.8, 2.6),
    'Treatment': (-3.8, -2.6),
    'Severity':  (3.8, -2.6),
}
# Cloud radii — uniform, compact to avoid timeline overlap
dim_radii = {'History': 1.25, 'Evidence': 1.25, 'Treatment': 1.25, 'Severity': 1.25}

# Number of visual nodes per dimension (roughly proportional to real count, capped for clarity)
vis_nodes = {'History': 16, 'Evidence': 11, 'Treatment': 7, 'Severity': 9}

def generate_structured_positions(cx, cy, r, n):
    """
    Place n nodes in concentric rings with Fibonacci-spiral-inspired angular offset.
    Rings: nodes distributed across 2-3 rings proportional to sqrt density.
    Within each ring: even angular spacing with golden-ratio rotation offset.
    """
    positions = []
    np.random.seed(hash((cx, cy, n)) % 2**31)
    
    if n <= 6:
        rings = [(0.50, n)]
    elif n <= 10:
        rings = [(0.42, n // 2), (0.78, n - n // 2)]
    else:
        n1 = n // 3
        n2 = (n - n1) // 2
        n3 = n - n1 - n2
        rings = [(0.30, n1), (0.58, n2), (0.85, n3)]
    
    golden = (1 + np.sqrt(5)) / 2
    global_idx = 0
    
    for ring_idx, (ring_r, ring_n) in enumerate(rings):
        if ring_n == 0:
            continue
        # Golden-ratio start angle for each ring to avoid alignment
        start_angle = ring_idx * np.pi / golden
        for i in range(ring_n):
            angle = start_angle + (i / ring_n) * 2 * np.pi
            # Slight jitter for natural organic feel (but much less than before)
            angle += np.random.uniform(-0.06, 0.06)
            rr = ring_r + np.random.uniform(-0.03, 0.03)
            ex = cx + r * rr * np.cos(angle)
            ey = cy + r * rr * np.sin(angle)
            positions.append((ex, ey))
            global_idx += 1
    
    return positions

# Generate entity positions
np.random.seed(42)
entity_data = {}
for dim_name in DIMS:
    cx, cy = dim_centers[dim_name]
    r = dim_radii[dim_name]
    n = vis_nodes[dim_name]
    entity_data[dim_name] = generate_structured_positions(cx, cy, r, n)

# ===== STEP 1: Intra-dimension mesh connections (knowledge graph style) =====
for dim_name, positions in entity_data.items():
    color = DIMS[dim_name]['color']
    
    # Build minimum spanning-tree-like connections: connect each node to its 2-3 nearest neighbors
    for i, p1 in enumerate(positions):
        dists = [(j, np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)) 
                 for j, p2 in enumerate(positions) if j != i]
        dists.sort(key=lambda x: x[1])
        # Connect to 2-3 nearest neighbors
        for j, d in dists[:3]:
            p2 = positions[j]
            draw_bezier(ax_a, p1, p2, color, alpha=0.35, lw=0.7, zorder=0, curvature=0.12)

# ===== STEP 2: Cross-dimension connections (knowledge graph style) =====
# Connect entities across related dimensions with subtle curves
cross_pairs = [
    ('History', 'Evidence'),    # symptoms relate to lab/imaging findings
    ('Evidence', 'Severity'),   # lab results inform severity
    ('Treatment', 'Severity'),  # treatment decisions relate to severity
    ('History', 'Treatment'),   # history influences treatment
]

for dim_a, dim_b in cross_pairs:
    pos_a = entity_data[dim_a]
    pos_b = entity_data[dim_b]
    color_a = DIMS[dim_a]['color']
    color_b = DIMS[dim_b]['color']
    # Blend colors
    blended = '#8888AA'
    
    n_connections = min(len(pos_a), len(pos_b), 5)
    # Randomly select some nodes from each dimension to connect
    indices_a = list(np.random.choice(len(pos_a), n_connections, replace=False))
    indices_b = list(np.random.choice(len(pos_b), n_connections, replace=False))
    
    for ia, ib in zip(indices_a, indices_b):
        p1 = pos_a[ia]
        p2 = pos_b[ib]
        dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        if dist < 6.0:  # Only draw if not too far
            draw_bezier(ax_a, p1, p2, blended, alpha=0.18, lw=0.6, zorder=0, curvature=0.25)

# ===== STEP 3: Draw dimension boundary clouds =====
for dim_name, d_info in DIMS.items():
    cx, cy = dim_centers[dim_name]
    r = dim_radii[dim_name]
    cloud = Ellipse((cx, cy), r*2 + 0.3, r*2 + 0.3, facecolor=d_info['light'],
                    edgecolor=d_info['color'], linewidth=0.8, alpha=0.35,
                    linestyle=(0, (3, 2)), zorder=0)
    ax_a.add_patch(cloud)

# ===== STEP 4: Draw hyperedges (fan polygons from patient to day's entities) =====
day_layers = [(0, 0.28, 1.3), (2, 0.17, 1.0), (4, 0.09, 0.7)]
for day, alpha, lw in day_layers:
    for dim_name, positions in entity_data.items():
        color = DIMS[dim_name]['color']
        n_active = max(1, int(len(positions) * (0.55 - day * 0.08)))
        indices = list(np.random.choice(len(positions), n_active, replace=False))
        pts = [positions[i] for i in indices]
        if len(pts) < 2:
            continue
        pts_arr = np.array(pts)
        angles = np.arctan2(pts_arr[:,1] - center[1], pts_arr[:,0] - center[0])
        pts_sorted = pts_arr[np.argsort(angles)]
        poly_pts = np.vstack([center, pts_sorted])
        poly = Polygon(poly_pts, facecolor=color, edgecolor=color,
                       alpha=alpha*0.45, linewidth=lw*0.5, linestyle='-', zorder=2)
        ax_a.add_patch(poly)
        # Radial connector lines
        for pt in pts:
            ax_a.plot([center[0], pt[0]], [center[1], pt[1]], color=color,
                      alpha=alpha*0.3, linewidth=lw*0.3, zorder=2)

# ===== STEP 5: Draw entity nodes (on top) =====
for dim_name, positions in entity_data.items():
    color = DIMS[dim_name]['color']
    for (ex, ey) in positions:
        ax_a.add_patch(Circle((ex, ey), 0.1, facecolor=color, edgecolor='white', linewidth=0.8, zorder=10))

# Re-draw patient on very top
ax_a.add_patch(Circle(center, 0.55, facecolor=C_DARK, edgecolor='#1A1A1A', linewidth=2.5, zorder=30))
ax_a.add_patch(Circle(center, 0.40, facecolor='#3D3D3D', edgecolor='none', zorder=31))
ax_a.text(0, 0.05, 'Patient', ha='center', va='center', fontsize=7.5, color='white', fontweight='bold', zorder=32)
ax_a.text(0, -0.22, 'Node', ha='center', va='center', fontsize=6, color='#BBBBBB', zorder=32)

# ===== STEP 6: Dimension labels =====
for dim_name, (cx, cy) in dim_centers.items():
    d_info = DIMS[dim_name]
    r = dim_radii[dim_name]
    label_y = cy + r + 0.48
    ax_a.add_patch(Circle((cx, label_y + 0.08), 0.2, facecolor=d_info['color'], edgecolor='none', zorder=10))
    ax_a.text(cx, label_y + 0.08, d_info['icon'], ha='center', va='center', fontsize=7,
              color='white', fontweight='bold', zorder=11)
    ax_a.text(cx + 0.35, label_y + 0.08, dim_name, fontsize=8, fontweight='bold',
              color=d_info['color'], va='center', zorder=10)
    ax_a.text(cx + 0.35, label_y - 0.18, f'n={d_info["n"]} entities', fontsize=6.5,
              color=C_LIGHT, va='center', zorder=10)

# ===== STEP 7: Timeline =====
timeline_y = -4.5
ax_a.plot([-5.2, 5.2], [timeline_y, timeline_y], color='#CCCCCC', lw=1.5, zorder=3)
ax_a.annotate('', xy=(5.5, timeline_y), xytext=(-5.5, timeline_y),
              arrowprops=dict(arrowstyle='->', color=C_MID, lw=1.8))
for d in range(WINDOW):
    dx = -4.3 + d * 2.15
    ax_a.plot([dx, dx], [timeline_y-0.18, timeline_y+0.18], color=C_MID, lw=1.5, zorder=4)
    ax_a.text(dx, timeline_y + 0.35, f'Day {d}', ha='center', fontsize=7.5, color=C_MID, fontweight='bold')
ax_a.text(0.5, timeline_y - 0.5, f'{WINDOW}-day temporal window  |  mean {MEAN_DAYS:.1f} active days  |  {COVERAGE}% coverage',
          ha='center', fontsize=7, color=C_LIGHT, style='italic')

# Hyperedges legend
legend_y2 = -5.2
for i, (lbl, clr) in enumerate([('E_history', '#3B7DD8'), ('E_evidence', '#2CA02C'),
                                 ('E_treatment', '#D62728'), ('E_severity', '#9467BD')]):
    lx = -3.5 + i * 2.3
    ax_a.add_patch(Rectangle((lx, legend_y2 - 0.1), 0.25, 0.25, facecolor=clr, edgecolor='none', alpha=0.7, zorder=5))
    ax_a.text(lx + 0.4, legend_y2 + 0.02, lbl, fontsize=6.5, color=C_MID, va='center')

# Annotation
ax_a.text(-4.5, 4.85, 'Fan = hyperedges\nCurves = entity co-occurrence\nDashed lines = cross-dim links',
          fontsize=6.5, color=C_LIGHT, style='italic', va='top', ha='left')


# ==================== PANEL (b): Cohort Summary ====================
ax_b = fig.add_axes([0.56, 0.06, 0.42, 0.90])
ax_b.set_xlim(0, 10)
ax_b.set_ylim(0, 10)
ax_b.set_facecolor(C_WHITE)
ax_b.axis('off')

ax_b.text(0.05, 9.6, '(b)  Cohort Summary', fontsize=11, fontweight='bold', color=C_DARK)
ax_b.text(0.05, 9.3, f'{N_PATIENTS} patients  |  {TOTAL_ENTITIES} entities  |  {TOTAL_HYPEREDGES:,} hyperedges  |  {TOTAL_ASSOC:,} associations',
          fontsize=7, color=C_LIGHT)

# ---- Row 1: Entity bars ----
y1 = 8.6
ax_b.text(0.05, y1 + 0.35, 'Entity Vocabulary', fontsize=9, fontweight='bold', color=C_DARK)

dim_names = list(DIMS.keys())
bar_h = 0.36
bar_gap = 0.52
bar_x = 0.15
bar_max_w = 7.8
max_n = max(d['n'] for d in DIMS.values())

for i, (name, d_info) in enumerate(DIMS.items()):
    yi = y1 - i * bar_gap
    w = (d_info['n'] / max_n) * bar_max_w
    ax_b.add_patch(FancyBboxPatch((bar_x, yi - bar_h/2), bar_max_w, bar_h,
                    boxstyle="round,pad=0.04", facecolor='#F0F0F0', edgecolor='none', zorder=0))
    ax_b.add_patch(FancyBboxPatch((bar_x, yi - bar_h/2), w, bar_h,
                    boxstyle="round,pad=0.04", facecolor=d_info['color'], edgecolor='none', alpha=0.9, zorder=1))
    ax_b.text(bar_x + w + 0.12, yi, f'{d_info["n"]}', fontsize=8, fontweight='bold',
              color=d_info['color'], va='center')
    ax_b.text(bar_x + 0.12, yi + 0.13, name, fontsize=7, fontweight='bold', color=C_DARK, va='bottom')
    ax_b.text(bar_x + 0.12, yi - 0.13, d_info['types'], fontsize=6, color=C_LIGHT, va='top')

y_total = y1 - 4 * bar_gap - 0.08
ax_b.text(bar_x, y_total, f'Total: {TOTAL_ENTITIES} entities  |  Embedding V={TOTAL_ENTITIES}, d=32',
          fontsize=7, color=C_MID, fontweight='bold')

# ---- Row 2: Associations ----
y2 = y_total - 0.85
ax_b.text(0.05, y2 + 0.25, 'Entity-Patient Associations', fontsize=9, fontweight='bold', color=C_DARK)

assoc_y = y2 - 0.35
assoc_bar_h = 0.42
cumulative_x = bar_x
for name, d_info in DIMS.items():
    assoc_w = (d_info['assoc'] / TOTAL_ASSOC) * bar_max_w
    ax_b.add_patch(FancyBboxPatch((cumulative_x, assoc_y - assoc_bar_h/2), assoc_w, assoc_bar_h,
                    boxstyle="round,pad=0.03", facecolor=d_info['color'], edgecolor='white',
                    linewidth=1.5, alpha=0.88, zorder=2))
    if d_info['assoc'] > 800:
        ax_b.text(cumulative_x + assoc_w/2, assoc_y, f'{d_info["assoc"]:,}',
                  ha='center', va='center', fontsize=6.5, fontweight='bold', color='white', zorder=3)
    cumulative_x += assoc_w
ax_b.text(cumulative_x + 0.15, assoc_y, f'{TOTAL_ASSOC:,}', fontsize=8, fontweight='bold', color=C_DARK, va='center')
ax_b.text(cumulative_x + 0.15, assoc_y - 0.18, 'total', fontsize=6, color=C_LIGHT, va='center')

# ---- Row 3: KPI Cards + Pipeline ----
y3 = assoc_y - 1.2

card_w = 1.9
card_h = 0.9
card_gap = 0.22
card_start_x = 0.05
card_data_kpi = [
    ('182', 'Entities', f'{TOTAL_ENTITIES} nodes', '#3B7DD8'),
    ('1,378', 'Hyperedges', f'per-day per-dim', '#D62728'),
    (f'{COVERAGE}%', 'Coverage', f'{WINDOW}-day window', '#2CA02C'),
]
for i, (value, title, subtitle, color) in enumerate(card_data_kpi):
    cx = card_start_x + i * (card_w + card_gap)
    ax_b.add_patch(FancyBboxPatch((cx, y3 - card_h), card_w, card_h,
                    boxstyle="round,pad=0.06", facecolor=C_WHITE, edgecolor='#E0E0E0', linewidth=0.8, zorder=2))
    ax_b.add_patch(FancyBboxPatch((cx + 0.06, y3 - 0.08), card_w - 0.12, 0.04,
                    boxstyle="round,pad=0.02", facecolor=color, edgecolor='none', alpha=0.8, zorder=3))
    ax_b.text(cx + card_w/2, y3 - 0.38, value, fontsize=15, fontweight='bold',
              color=color, ha='center', va='center', zorder=4)
    ax_b.text(cx + card_w/2, y3 - 0.62, title, fontsize=6.5, fontweight='bold',
              color=C_MID, ha='center', va='center', zorder=4)
    ax_b.text(cx + card_w/2, y3 - 0.80, subtitle, fontsize=5.5,
              color=C_LIGHT, ha='center', va='center', zorder=4)

# Pipeline
pipe_x = 6.2
pipe_y_top = y3 - 0.1
step_h = 0.28
step_w = 3.2
step_gap = 0.08
pipe_steps = [
    ('1', '8 Clinical Documents', '#E8E8E8', C_DARK),
    ('2', 'LLM Extraction (4-D Prompts)', '#D5F5D5', '#2CA02C'),
    ('3', 'Entity Vocabulary (V=182)', '#D6E6FB', '#3B7DD8'),
    ('4', 'Per-Day Per-Dim Hyperedges (1,378)', '#FDDCDC', '#D62728'),
    ('5', 'Temporal Hypergraph G', '#EDE3F6', '#9467BD'),
]
for i, (num, text, bg, border) in enumerate(pipe_steps):
    sy = pipe_y_top - i * (step_h + step_gap)
    ax_b.add_patch(FancyBboxPatch((pipe_x, sy - step_h), step_w, step_h,
                    boxstyle="round,pad=0.04", facecolor=bg, edgecolor=border, linewidth=0.8, alpha=0.9, zorder=2))
    ax_b.text(pipe_x + 0.12, sy - step_h/2, num, fontsize=5.5, fontweight='bold',
              color=border, va='center', zorder=3)
    ax_b.text(pipe_x + step_w/2, sy - step_h/2, text, fontsize=5.8,
              color=C_DARK, ha='center', va='center', fontweight='bold', zorder=3)
    if i < len(pipe_steps) - 1:
        ax_b.annotate('', xy=(pipe_x + step_w/2, sy - step_h - step_gap + 0.02),
                      xytext=(pipe_x + step_w/2, sy - step_h - 0.01),
                      arrowprops=dict(arrowstyle='->', color='#CCCCCC', lw=0.8))

# ---- Row 4: Icon legend + footer ----
y4 = pipe_y_top - len(pipe_steps) * (step_h + step_gap) - 0.5
icon_x_positions = [0.15, 2.5, 4.85, 7.2]
for i, (name, d_info) in enumerate(DIMS.items()):
    ix = icon_x_positions[i]
    ax_b.add_patch(Circle((ix + 0.15, y4), 0.16, facecolor=d_info['color'], edgecolor='none', zorder=2))
    ax_b.text(ix + 0.15, y4, d_info['icon'], ha='center', va='center', fontsize=6.5,
              color='white', fontweight='bold', zorder=3)
    ax_b.text(ix + 0.38, y4, f'{name}', fontsize=7, fontweight='bold', color=d_info['color'], va='center')
    ax_b.text(ix + 0.38, y4 - 0.17, d_info['types'], fontsize=5.5, color=C_LIGHT, va='center')

ax_b.text(5.0, y4 - 0.7, 'Figure 7: Four-Dimensional Temporal Hypergraph Structure  |  4D-HyperGraph',
          ha='center', fontsize=6, color='#BBBBBB', style='italic')

# ========== Save ==========
fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight', facecolor=C_WHITE,
            edgecolor='none', pad_inches=0.2)
plt.close(fig)

print(f'Saved: {OUTPUT_PATH}')
print(f'Stats: Entities={TOTAL_ENTITIES} | Hyperedges={TOTAL_HYPEREDGES} | Assoc={TOTAL_ASSOC} | Coverage={COVERAGE}%')
