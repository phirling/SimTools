import numpy as np
import areeepo as ap
import matplotlib.pyplot as plt

L = 10.0
extent1 = np.zeros((3,2))
extent1[:,1] = L

# Deliberate positions: a few ordered lines
pos1 = np.zeros((50, 3))
pos1[:20, 0] = np.linspace(0,L,20)
pos1[20:40,0] = 0.0
pos1[40:50,0] = np.linspace(L/2., L, 10)

pos1[:20, 1] = 0.0
pos1[20:40,1] = np.linspace(0,L,20)
pos1[40:50,1] = 2.5

pos1[:20, 2] = 7.5
pos1[20:40,2] = 2.5
pos1[40:50,2] = 5.0

# The lines have strong value
vals1 = 50*np.ones(50)

# Random points for NN to work
nrdm = 256
pos2 = np.random.uniform(0,L,(nrdm,3))
vals2 = np.random.uniform(0,1.0 / nrdm,(nrdm,))

# Stack them
POS = np.vstack((pos1,pos2))
VAL = np.hstack((vals1,vals2))


modes = ['xy', 'xz','yz']


fig, ax = plt.subplots(3,3, figsize=(8,8))
for i, md in enumerate(modes):
    d1,d2,d3 = ap.mode2dim(md)

    projH, bs, e2d = ap.project_with_histogram(POS, VAL, axis=d3, bins=20, extent=extent1, density = True)
    projNN, bs, e2d = ap.project_with_NN(POS, VAL, axis=d3, bins=20, extent=extent1)

    ax[i,0].scatter(pos1[:, d1], pos1[:, d2])
    ax[i,0].set_xlim(extent1[d1,0],extent1[d1,1])
    ax[i,0].set_ylim(extent1[d2,0],extent1[d2,1])

    ax[i,1].imshow(projH.T, extent = e2d, origin='lower')
    ax[i,2].imshow(projNN.T, extent = e2d, origin='lower')

fig.tight_layout()

plt.show()