import seaborn as sns
import pandas as pd
import h5py
from SOAPify import loadRefs
from SOAPify import simpleSOAPdistance as SOAPdistance
import numpy as np
from .referenceSaponificator import radiusInfo


def distanceVisualizer(rcuts: "list[radiusInfo]", referencesFile: str, kind: str):
    for rcut in rcuts:
        print(f"Drawing distance graph for {rcut.name} (rcut={rcut.rcut})")
        data, legend = loadRefs(
            h5py.File(referencesFile, "r"),
            [
                f"Bulk/{rcut.name}",
                f"Ico5083/{rcut.name}",
                f"Th4116/{rcut.name}",
                f"Dhfat3049/{rcut.name}",
            ],
        )

        distanceMatrix = np.zeros((len(data), len(data)))

        for i in range(len(data)):
            # distanceMatrix[i, i] = 0
            for j in range(i + 1, len(data)):
                distanceMatrix[j, i] = distanceMatrix[i, j] = SOAPdistance(
                    data[i], data[j]
                )
                # if np.isnan(distanceMatrix[j, i]):
                #    distanceMatrix[j, i] = distanceMatrix[i, j] = 0.0
                # print(distanceMatrix[j, i])

        df = pd.DataFrame(distanceMatrix, index=legend, columns=legend)
        outFname = f"distmat_{kind}Classification_{rcut.name}"
        # df.to_csv(f"{outFname}.csv")
        center = df.values[np.tril_indices(df.shape[0], k=1)].mean()

        g = sns.clustermap(
            df,
            center=center,
            cmap="Spectral_r",
            method="average",
            # row_colors=get_colors(df.index),
            dendrogram_ratio=(0.4, 0.4),
            cbar_pos=(1.02, 0.15, 0.03, 0.45),
            linewidths=0.75,
            figsize=(8, 8),
        )
        g.ax_col_dendrogram.remove()
        # _ = g.ax_heatmap.set_xticks([])
        g.savefig(f"{outFname}.png", bbox_inches="tight")
        # g.savefig(f"{outFname}.pdf", bbox_inches="tight")
