# noqa: INP001 D100
from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("perf.csv")
data = (data / data["Original"].mean().max()) * 100

print(data)

fig = plt.figure(figsize=(10, 8), dpi=100)
fig.subplots_adjust(left=0.25, top=0.85)
axes = plt.subplot()

axes.barh(
    data.columns,
    data.mean(),
    color=[
        "#1f6acc" if value > data.mean().min() else "#24a348" for value in data.mean()
    ],
    height=0.7,
)
axes.errorbar(
    data.mean(),
    data.columns,
    xerr=data.std(),
    ls="None",
    color="black",
    linewidth=1,
    capsize=6,
)

CENTER = 50

for i, name in enumerate(data.columns):
    plt.text(
        x=25 if data[name].min() > CENTER else 75,
        y=i,
        s=f"{data[name].mean():.1f}%",
        ha="center",
        va="center",
        bbox={"facecolor": "white", "alpha": 0.8},
    )

plt.grid()
plt.title(
    "Gilbert\n( 5qubits | 32x32 state | FS mode | 2M iters | 1K corrs )",
    fontsize=18,
    pad=20,
)
plt.xlabel("Work time [% of original]", fontsize=14, labelpad=20)
plt.ylabel("Backend configuration", fontsize=14, labelpad=20)

plt.savefig("perf.jpg")
