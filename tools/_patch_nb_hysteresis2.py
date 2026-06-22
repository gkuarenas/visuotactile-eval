"""Patch notebook: update speed/depth comments from 3.5 to 4.0."""
import json, os, re

NB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "stage1_results_v2.ipynb",
)

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_map = {cell["id"]: cell for cell in nb["cells"]}

# ── a003: update comment line in HYSTERESIS_SESSIONS block ───────────────────
a003 = cell_map["a003"]
src = "".join(a003["source"])
new_src = src.replace(
    "# Speeds tested: 0.1, 0.5, 1.0, 2.0, 3.5 mm/s  |  Max depth: 3.5 mm",
    "# Speeds tested: 0.1, 0.5, 1.0, 2.0, 4.0 mm/s  |  Max depth: 4.0 mm",
)
if new_src == src:
    print("a003: WARNING — comment not found")
else:
    a003["source"] = [line + "\n" for line in new_src.splitlines()]
    if a003["source"] and a003["source"][-1] == "\n":
        a003["source"][-1] = ""
    print("a003: speeds/depth comment updated")

# ── a014: update §7 markdown ──────────────────────────────────────────────────
a014 = cell_map["a014"]
a014["source"] = [
    "## §7 Load Hysteresis Data\n",
    "\n",
    "Each session folder is one slab (n=3 per blend). The CSV groups data by `speed_mm_s`\n",
    "(5 speeds: 0.1, 0.5, 1.0, 2.0, 4.0 mm/s, max depth 4.0 mm). For each speed,\n",
    "one load–unload cycle is recorded with inline force samples at every 0.1 mm step.\n",
    "HI is computed per speed via trapezoid integration of force vs. depth,\n",
    "then averaged across speeds to give the slab-mean HI.",
]
print("a014: markdown updated")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Saved: {NB_PATH}")
