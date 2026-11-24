import json
from pathlib import Path

p = Path("member3_model_training.ipynb")
n = json.loads(p.read_text(encoding="utf-8"))

print("NOTEBOOK:", p.resolve())
print("TOTAL CELLS:", len(n["cells"]))

for i, c in enumerate(n["cells"]):
    source = "".join(c.get("source", []))
    outputs = len(c.get("outputs", []))
    print(f"\n--- CELL {i} | {c['cell_type']} | lines: {len(source.splitlines())} | outputs: {outputs} ---")
    print(source[:1800])
