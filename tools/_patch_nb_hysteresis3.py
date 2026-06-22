"""Patch §9 markdown cell only — no file paths modified."""
import json, os

NB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "stage1_results_v2.ipynb",
)

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_map = {cell["id"]: cell for cell in nb["cells"]}

cell_map["a018"]["source"] = [
    "## §9 Hysteresis — Loop Curves\n",
    "\n",
    "One subplot per blend (2×2). Five closed loop curves overlaid per subplot\n",
    "(loading 0→max then unloading max→0 as a single solid line), each speed\n",
    "in a distinct scienceplots color. Each curve is the mean across n=3 slabs.\n",
    "|HI| annotation shows mean ± SD of absolute hysteresis index across slabs.",
]

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("a018 updated. Saved.")
