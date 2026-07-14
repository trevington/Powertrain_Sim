import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay

# Locate CSV robustly (script folder, parent, repo root)
candidates = [
    Path(__file__).resolve().parent / 'tools\\emrax208_efficiency_map.csv',
    Path(__file__).resolve().parent.parent / 'tools\\emrax208_efficiency_map.csv',
    Path(__file__).resolve().parent / 'data' / 'tools\\emrax208_efficiency_map.csv',
]
csv_path = next((p for p in candidates if p.exists()), None)
if csv_path is None:
    raise FileNotFoundError(f"tools\\emrax208_efficiency_map.csv not found (tried: {candidates})")

data = pd.read_csv(csv_path, header=None)
if data.empty:
    raise ValueError(f"No data in {csv_path}")

x = data.iloc[:, 0].values
y = data.iloc[:, 1].values
c = data.iloc[:, 2].values

# Grid resolution (increase for finer map)
nx, ny = 100, 100
xi = np.linspace(x.min(), x.max(), nx)
yi = np.linspace(y.min(), y.max(), ny)
xi_grid, yi_grid = np.meshgrid(xi, yi)
points_grid = np.vstack([xi_grid.ravel(), yi_grid.ravel()]).T

# Mask outside convex hull using Delaunay
tri = Delaunay(np.vstack([x, y]).T)
mask = tri.find_simplex(points_grid) >= 0
mask = mask.reshape(xi_grid.shape)

# RBF parameters: set epsilon proportional to data scale, allow smooth tuning
scale = np.mean([np.ptp(x), np.ptp(y)])
epsilon = scale / 10.0 if scale > 0 else 1.0
rbf = Rbf(x, y, c, function='multiquadric', epsilon=epsilon, smooth=0.2)

ci = rbf(xi_grid, yi_grid)

# Apply mask: set outside-hull to nan so contours don't extrapolate
ci[~mask] = np.nan

# Optional gentle smoothing on the interior only
ci_smooth = gaussian_filter(np.nan_to_num(ci, nan=np.nanmean(c)), sigma=2)
ci_smooth[~mask] = np.nan

# Plot shaded topographical map with contours and sample points
plt.figure(figsize=(10, 7))
im = plt.imshow(ci_smooth, extent=(x.min(), x.max(), y.min(), y.max()),
                origin='lower', cmap='terrain', aspect='auto')
contours = plt.contour(xi_grid, yi_grid, ci_smooth, levels=12, colors='k', linewidths=0.6, alpha=0.6)
plt.clabel(contours, inline=True, fmt='%.3f', fontsize=8)
plt.scatter(x, y, c=c, cmap='terrain', edgecolors='k', s=40, zorder=3)
plt.colorbar(im, label='Efficiency')
plt.title('Topographical Efficiency Map (RBF + mask + smoothing)')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()

# Save before show
out = Path(__file__).resolve().parent / 'efficiency_map.png'
plt.savefig(out, dpi=300)
plt.show()

# Compute simple leave-one-out RMSE to quantify interpolation accuracy
errs = []
for i in range(len(x)):
    xi_train = np.delete(x, i)
    yi_train = np.delete(y, i)
    ci_train = np.delete(c, i)
    rbf_loocv = Rbf(xi_train, yi_train, ci_train,
                    function='multiquadric', epsilon=epsilon, smooth=0.2)
    pred = rbf_loocv(x[i], y[i])
    errs.append((pred - c[i])**2)
rmse = float(np.sqrt(np.mean(errs)))
print(f"LOO RMSE: {rmse:.5f}")

# Print a recommended next step
print("Try adjusting 'smooth' and 'epsilon' in Rbf for better fit; use more sample points where RMSE is high.")

