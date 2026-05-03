"""
Execute MemEIC_Pipeline_Debug.ipynb programmatically and save with outputs.

Usage:
    python execute_notebook.py

Produces:
    MemEIC_Pipeline_Debug_executed.ipynb — notebook with all outputs embedded.
"""

import sys
import time
import traceback
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

INPUT_NB  = "MemEIC_Pipeline_Debug.ipynb"
OUTPUT_NB = "MemEIC_Pipeline_Debug_executed.ipynb"
KERNEL    = "memeic"          # registered via: python -m ipykernel install --user --name memeic
TIMEOUT   = 1200              # 20 min per cell (model loading + forward passes are heavy)

def main():
    print("=" * 70)
    print("  MemEIC Notebook Executor")
    print("=" * 70)

    # ── 1. Load notebook ────────────────────────────────────────────
    print(f"\n[1/4] Loading notebook: {INPUT_NB}")
    nb = nbformat.read(INPUT_NB, as_version=4)

    total_cells = len(nb.cells)
    code_cells = sum(1 for c in nb.cells if c.cell_type == "code")
    md_cells = total_cells - code_cells
    print(f"       Total cells: {total_cells} ({code_cells} code, {md_cells} markdown)")

    # ── 2. Set kernel metadata ──────────────────────────────────────
    print(f"\n[2/4] Setting kernel to '{KERNEL}'")
    nb.metadata["kernelspec"] = {
        "display_name": "Python (memeic)",
        "language": "python",
        "name": KERNEL,
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }

    # ── 3. Execute cell-by-cell with error handling ─────────────────
    print(f"\n[3/4] Executing notebook (timeout={TIMEOUT}s per cell)...")
    print(f"       Kernel: {KERNEL}")
    print()

    client = NotebookClient(
        nb,
        timeout=TIMEOUT,
        kernel_name=KERNEL,
        resources={"metadata": {"path": "."}},  # CWD = notebook directory
    )

    start_time = time.time()
    errors = []
    executed_code = 0

    # Start the kernel once — reuse for all cells
    with client.setup_kernel():
        for idx, cell in enumerate(nb.cells):
            cell_num = idx + 1

            if cell.cell_type != "code":
                # Markdown cells: nothing to execute
                continue

            executed_code += 1
            source_preview = cell.source.split("\n")[0][:60]
            print(f"  [{cell_num:>2d}/{total_cells}] Running code cell {executed_code}/{code_cells}: "
                  f"{source_preview}...")

            cell_start = time.time()
            try:
                client.execute_cell(cell, idx)
                elapsed = time.time() - cell_start
                # Count output types
                out_types = [o.get("output_type", "?") for o in cell.get("outputs", [])]
                print(f"           ✓ Done in {elapsed:.1f}s  "
                      f"(outputs: {len(out_types)} — {', '.join(set(out_types)) if out_types else 'none'})")

            except CellExecutionError as e:
                elapsed = time.time() - cell_start
                ename = getattr(e, "ename", "Error")
                evalue = getattr(e, "evalue", str(e))
                short_err = f"{ename}: {evalue}"[:120]
                print(f"           ✗ FAILED in {elapsed:.1f}s — {short_err}")
                errors.append({"cell": cell_num, "error": short_err})
                # Error is already captured in cell outputs by nbclient — continue

            except Exception as e:
                elapsed = time.time() - cell_start
                short_err = f"{type(e).__name__}: {str(e)}"[:120]
                print(f"           ✗ UNEXPECTED ERROR in {elapsed:.1f}s — {short_err}")
                errors.append({"cell": cell_num, "error": short_err})

    total_time = time.time() - start_time

    # ── 4. Save executed notebook ───────────────────────────────────
    print(f"\n[4/4] Saving executed notebook: {OUTPUT_NB}")
    nbformat.write(nb, OUTPUT_NB)

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  EXECUTION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total time:     {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Cells executed: {executed_code}/{code_cells} code cells")
    print(f"  Errors:         {len(errors)}")

    if errors:
        print(f"\n  Failed cells:")
        for err in errors:
            print(f"    Cell {err['cell']}: {err['error']}")

    print(f"\n  Output saved to: {OUTPUT_NB}")
    print(f"{'=' * 70}")

    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
