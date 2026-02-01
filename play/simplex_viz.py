import matplotlib.pyplot as plt
import numpy as np

def simplex_to_cartesian(coords):
    """Convert simplex coordinates (a, b, c) to 2D Cartesian for plotting."""
    a, b, c = coords
    x = 0.5 * (2 * b + c)
    y = (np.sqrt(3) / 2) * c
    return x, y

# Create figure
fig, ax = plt.subplots(figsize=(8, 7))

# Draw the simplex triangle
triangle_vertices = np.array([
    simplex_to_cartesian((1, 0, 0)),
    simplex_to_cartesian((0, 1, 0)),
    simplex_to_cartesian((0, 0, 1)),
    simplex_to_cartesian((1, 0, 0)),
])
ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], 'k-', linewidth=2)

# Draw grid lines
for i in range(1, 10):
    t = i / 10
    p1 = simplex_to_cartesian((t, 1-t, 0))
    p2 = simplex_to_cartesian((t, 0, 1-t))
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', linewidth=0.5, alpha=0.4)
    p1 = simplex_to_cartesian((1-t, t, 0))
    p2 = simplex_to_cartesian((0, t, 1-t))
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', linewidth=0.5, alpha=0.4)
    p1 = simplex_to_cartesian((1-t, 0, t))
    p2 = simplex_to_cartesian((0, 1-t, t))
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', linewidth=0.5, alpha=0.4)

# Feasible region vertices (a, b, c)
# Given b vs c head-to-head is (0.3, 0.7):
# - vote_b <= 0.3 (b's max share when all b-preferrers have b as first choice)
# - vote_c <= 0.7 (c's max share when all c-preferrers have c as first choice)
region_vertices = [
    (1, 0, 0),      # all votes to a
    (0.7, 0.3, 0),  # a=0.7, b=0.3 (b maxed), c=0
    (0, 0.3, 0.7),  # a=0, b=0.3 (b maxed), c=0.7 (c maxed)
    (0.3, 0, 0.7),  # a=0.3, b=0, c=0.7 (c maxed)
]

# Convert to cartesian and draw filled region
region_xy = np.array([simplex_to_cartesian(v) for v in region_vertices])
polygon = plt.Polygon(region_xy, alpha=0.4, facecolor='#3498db', edgecolor='#2980b9', linewidth=2)
ax.add_patch(polygon)

# Mark vertices
for v in region_vertices:
    x, y = simplex_to_cartesian(v)
    ax.scatter(x, y, c='#e74c3c', s=100, zorder=5, edgecolors='black', linewidth=1.5)
    ax.annotate(f'{v}', (x, y), xytext=(8, 8), textcoords='offset points', fontsize=9)

# Label simplex vertices
ax.text(-0.08, -0.05, 'A (1,0,0)', ha='center', fontsize=11, fontweight='bold')
ax.text(1.08, -0.05, 'B (0,1,0)', ha='center', fontsize=11, fontweight='bold')
ax.text(0.5, np.sqrt(3)/2 + 0.05, 'C (0,0,1)', ha='center', fontsize=11, fontweight='bold')

ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Feasible 3-Way Election Region\n(given B vs C head-to-head is 0.3 : 0.7)',
             fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/Users/sethlupo/code/mcm2026/play/simplex_viz.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved to simplex_viz.png")
