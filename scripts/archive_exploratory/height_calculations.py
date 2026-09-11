import numpy as np

nf = np.array([-0.006948, 0.999966, 0.004543], dtype=float)
df = 1.477840

nc = np.array([0.001841, 0.999997, -0.001799], dtype=float)
dc = -1.584690

nf /= np.linalg.norm(nf)
nc /= np.linalg.norm(nc)

# Common vertical direction.
n = nf + nc
n /= np.linalg.norm(n)

# Orient upward.
if n[1] < 0:
    n = -n

# Re-express each plane's offset along common normal.
# For a plane n_i . x + d_i = 0, a point on it is
# at signed coordinate -d_i / (n . n_i) along common n.
hf = -df / np.dot(n, nf)
hc = -dc / np.dot(n, nc)

height = hc - hf

print("Common vertical normal:", n)
print("Floor coordinate:", hf)
print("Ceiling coordinate:", hc)
print("Height:", height)
print("Height cm:", height * 100)
