"""Patch notebook: update to 3 speeds (0.1, 0.5, 1.0 mm/s) — no file paths modified."""
import json, os

NB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "stage1_results_v2.ipynb",
)

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_map = {cell["id"]: cell for cell in nb["cells"]}

# ── a003: update speed comment ────────────────────────────────────────────────
a003 = cell_map["a003"]
src = "".join(a003["source"])
new_src = src.replace(
    "# Speeds tested: 0.1, 0.5, 1.0, 2.0, 4.0 mm/s  |  Max depth: 4.0 mm",
    "# Speeds tested: 0.1, 0.5, 1.0 mm/s  |  Max depth: 4.0 mm",
)
if new_src == src:
    print("a003: WARNING — comment not found, skipping")
else:
    a003["source"] = [line + "\n" for line in new_src.splitlines()]
    if a003["source"] and a003["source"][-1] == "\n":
        a003["source"][-1] = ""
    print("a003: speeds comment updated to 3 speeds")

# ── a014: update §7 markdown ──────────────────────────────────────────────────
cell_map["a014"]["source"] = [
    "## §7 Load Hysteresis Data\n",
    "\n",
    "Each session folder is one slab (n=3 per blend). The CSV groups data by `speed_mm_s`\n",
    "(3 speeds: 0.1, 0.5, 1.0 mm/s, max depth 4.0 mm). For each speed,\n",
    "one load–unload cycle is recorded with one force sample at every 0.1 mm step.\n",
    "HI is computed per speed via trapezoid integration of force vs. depth,\n",
    "then averaged across speeds to give the slab-mean HI.\n",
    "Note: speeds above ~0.5 mm/s showed no significant curve separation due to\n",
    "serial command latency on the Ender 3 Z-axis; 1.0 mm/s retained as upper bound.",
]
print("a014: markdown updated to 3 speeds")

# ── a018: update §9 markdown ──────────────────────────────────────────────────
cell_map["a018"]["source"] = [
    "## §9 Hysteresis — Loop Curves\n",
    "\n",
    "One subplot per blend (2×2). Three speed curves overlaid per subplot\n",
    "(0.1, 0.5, 1.0 mm/s), loading and unloading as separate same-color traces.\n",
    "Each curve is the mean across n=3 slabs.\n",
    "|HI| annotation shows mean ± SD of absolute hysteresis index across slabs.",
]
print("a018: markdown updated")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Saved: {NB_PATH}")
