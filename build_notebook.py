"""Convert vibration_analysis.py (percent format) into vibration_analysis.ipynb.

Cells starting with '# %% [markdown]' become Markdown cells (leading '# ' is
stripped); '# %%' starts a code cell. The module docstring is dropped.
"""
from pathlib import Path

import nbformat as nbf

src = Path(__file__).with_name("vibration_analysis.py").read_text()
blocks = src.split("\n# %%")[1:]
nb = nbf.v4.new_notebook()
for block in blocks:
    head, _, body = block.partition("\n")
    if head.strip() == "[markdown]":
        text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#")
                         for line in body.strip().splitlines())
        nb.cells.append(nbf.v4.new_markdown_cell(text))
    else:
        nb.cells.append(nbf.v4.new_code_cell(body.strip("\n")))
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                             "name": "python3"}
nbf.write(nb, Path(__file__).with_name("vibration_analysis.ipynb"))
print(f"wrote notebook with {len(nb.cells)} cells")
