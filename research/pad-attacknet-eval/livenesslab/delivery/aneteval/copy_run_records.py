"""Copia nel pacchetto della valutazione C5 (delivery/pad-attacknet-eval/runs/<run id>/) i record pubblicabili dei comandi
dei run, come per C2 (delivery/pad-arch-eval/runs/): per ogni tentativo di ciascuno dei sei run e per ogni check-estimator
della radice dei run, status.json, run.log e public/ (nulla di restricted/), più conditions.txt, il file delle condizioni
scritto da delivery/anetrun/launch.zsh (<run o check-estimator>-attempt<N>-conditions.txt nella cartella dei log del
lancio), copiato così com'è. Dopo i sei run, dalla radice del repository:

  .venv/bin/python -I -B delivery/aneteval/copy_run_records.py --runs-root results/anetrun --conditions-dir <cartella dei log del lancio>

Prima di scrivere: ogni file delle condizioni presente; nessun testo vietato nelle copie (PAC.output_problems, senza i
manifest: percorsi e testi personali, non gli identificativi di soggetto o di frame; ridondante per run.log e public/,
già scansionati dal runner con gli ingressi reali). Non sovrascrive una copia diversa senza --force. pad_anet_eval.py
check-inputs ed evaluate verificano poi le copie byte per byte (evaluate con la scansione sui manifest caricati)."""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import nuaa_folds_run as FR  # noqa: E402
import pad_anet_eval as V  # noqa: E402
import pad_anet_run as NR  # noqa: E402
import pad_arch_comparison as PAC  # noqa: E402

PACKAGE_RUNS = ROOT / "delivery/pad-attacknet-eval/runs"
TAG_RUN = {t: r for r, t in NR.RUN_TAG.items()}


def label_of(run_id: str) -> str:
    """L'etichetta del lancio (launch.zsh) di una cartella: <nome del run>-attempt<N> o check-estimator-attempt<N>."""
    m = NR.RUN_ID_RE.match(run_id)
    if m:
        return f"{TAG_RUN[f'{m.group(1)}-p{m.group(2)}']}-attempt{int(m.group(3)) if m.group(3) else 1}"
    m = NR.EST_ID_RE.match(run_id)
    if m:
        return f"check-estimator-attempt{int(m.group(1)) if m.group(1) else 1}"
    raise SystemExit(f"{run_id}: not a run id of the runs or of check-estimator")


def plan(runs_root: Path, conditions_dir: Path) -> dict:
    """Che cosa copiare: per cartella (run id) i file (origine, destinazione relativa). Ogni file delle condizioni deve
    esserci e ogni testo deve passare la scansione delle uscite pubbliche."""
    out = {}
    for f in V.record_folders(runs_root):
        cond = conditions_dir / f"{label_of(f.name)}-conditions.txt"
        if not cond.is_file():
            raise SystemExit(f"{f.name}: the conditions file {cond.name} is missing in the folder of the launch logs")
        files = [(f / n, n) for n in V.record_files(f)] + [(cond, "conditions.txt")]
        for src, rel in files:
            if probs := PAC.output_problems(src.read_bytes().decode("utf-8", "replace"), FR.Inputs()):
                raise SystemExit(f"{f.name}/{rel}: text that a publishable output must not hold: " + "; ".join(probs))
        out[f.name] = files
    return out


def copy_all(runs_root: Path, conditions_dir: Path, dest: Path, force: bool = False) -> int:
    """Le copie in dest/<run id>/ (dopo il piano e i controlli); una copia esistente diversa è rifiutata senza force."""
    todo = plan(runs_root, conditions_dir)
    for rid, files in todo.items():
        for src, rel in files:
            dst = dest / rid / rel
            if dst.exists() and dst.read_bytes() != src.read_bytes() and not force:
                raise SystemExit(f"{rid}/{rel} exists and differs: pass --force to rewrite it")
    for rid, files in todo.items():
        for src, rel in files:
            dst = dest / rid / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    return sum(len(v) for v in todo.values())


def main() -> int:
    a = sys.argv[1:]
    if "--runs-root" not in a or "--conditions-dir" not in a:
        print("usage: copy_run_records.py --runs-root results/anetrun --conditions-dir <folder of the launch logs> [--force]")
        return 1
    root = Path(a[a.index("--runs-root") + 1])
    root = root if root.is_absolute() else ROOT / root
    cdir = Path(a[a.index("--conditions-dir") + 1]).resolve()
    n = copy_all(root, cdir, PACKAGE_RUNS, "--force" in a)
    print(f"run records: {len(V.record_folders(root))} folders copied to {PACKAGE_RUNS.relative_to(ROOT)} ({n} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
