import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib

fig = plt.figure(figsize=(22, 6.0))
fig.patch.set_facecolor("#0f0f0f")

col_titles = [
    "Orijinal (Cover)",
    "Stego (Product)",
    "Fark Haritasi  (x20 amplifikasyon)",
    "Quadtree Havuzu  (sari = secili)",
]

outer = gridspec.GridSpec(
    1,
    4,
    figure=fig,
    hspace=0.06,
    wspace=0.04,
    left=0.01,
    right=0.99,
    top=0.85,
    bottom=0.01,
)

panels = [
    (np.random.rand(512,512), "gray", False),
    (np.random.rand(512,512), "gray", False),
    (np.random.rand(512,512), "hot", True),
    (np.random.rand(512,512,3), None, False),
]

for col_idx, (data, cmap, add_cbar) in enumerate(panels):
    ax = fig.add_subplot(outer[0, col_idx])
    ax.set_title(col_titles[col_idx], fontsize=13, fontweight="bold", color="white", pad=15)
    ax.set_xticks([])
    ax.set_yticks([])
    if cmap is None:
        im = ax.imshow(data, interpolation="nearest")
    else:
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")

fig.suptitle(
    "Steganografi Gorsel Karsilastirmasi - Adaptive Quadtree + Discrete ABC + LSB-Matching\nKullanici gorsel cift: cover.pgm | stego.pgm",
    fontsize=14,
    fontweight="bold",
    color="white",
    y=0.98,
)
fig.savefig("test.png", facecolor=fig.get_facecolor(), dpi=100)
