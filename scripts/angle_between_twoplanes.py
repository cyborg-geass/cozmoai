import numpy as np

nf = np.array([-0.006948, 0.999966, 0.004543])
nc = np.array([0.001841, 0.999997, -0.001799])

nf /= np.linalg.norm(nf)
nc /= np.linalg.norm(nc)

angle = np.degrees(
    np.arccos(
        np.clip(np.dot(nf, nc), -1.0, 1.0)
    )
)

print(angle)
