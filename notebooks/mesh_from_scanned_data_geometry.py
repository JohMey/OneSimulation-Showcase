"""Input data of the OneSimulation example "Simulation mesh from scanned data".

Writes mesh_from_scanned_data.npy: a segmented CT image of 64 x 48 x 48 voxels of 0.5 mm, label 0
for air, 1 for an ellipsoidal bone (half axes 14, 10 and 10 mm) and 2 for a cylindrical implant
(radius 3 mm, 20 mm long) inside it, as a segmentation program would export it. Every voxel takes
the label of its centre.
"""

import numpy as np

spacing, shape = 0.5e-3, (64, 48, 48)
centres = (
    np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1) + 0.5
) * spacing
x, y, z = (centres - np.array(shape) * spacing / 2).transpose(3, 0, 1, 2)
labels = np.zeros(shape, dtype=np.uint8)
labels[(x / 0.014) ** 2 + (y / 0.010) ** 2 + (z / 0.010) ** 2 < 1.0] = 1
labels[(np.hypot(y, z) < 0.003) & (np.abs(x) < 0.010)] = 2
np.save("mesh_from_scanned_data.npy", labels)
