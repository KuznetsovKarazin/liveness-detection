"""
Esegue uno script come su una macchina senza TensorFlow (strumento dei log «without TensorFlow» di questo pacchetto).

Avvolge ogni finder di sys.meta_path: per i moduli nascosti (tensorflow e i suoi sottomoduli; con --hide anche altri
nomi di primo livello, per esempio cv2) find_spec dà None, quindi importlib.util.find_spec(...) è None, l'import fallisce
con ModuleNotFoundError e nessuna voce entra in sys.modules; per tutti gli altri moduli decide il finder originale.
Poi esegue lo script con runpy come __main__, con la sua cartella in testa a sys.path (python -I non ce la mette).
Non modifica nulla su disco e non tocca l'installazione.

  python -I -B tools/run_without_tensorflow.py scripts/check_pad_arch_comparison.py
  python -I -B tools/run_without_tensorflow.py --hide tensorflow,cv2 scripts/pad_arch_comparison.py plan --trial-run ../trial --out-dir <cartella nuova>
"""
import runpy
import sys
from pathlib import Path


class Hide:
    """Un finder di sys.meta_path che non trova i moduli nascosti e delega tutto il resto al finder avvolto."""

    def __init__(self, inner, names):
        self.inner = inner
        self.names = names

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.names:
            return None
        f = getattr(self.inner, "find_spec", None)
        return f(name, path, target) if f else None

    def __getattr__(self, k):
        return getattr(self.inner, k)


def main() -> None:
    args = sys.argv[1:]
    names = {"tensorflow"}
    if args[:1] == ["--hide"]:
        if len(args) < 2:
            sys.exit("usage: run_without_tensorflow.py [--hide name[,name]] script [arguments]")
        names = {x for x in args[1].split(",") if x}
        args = args[2:]
    if not args:
        sys.exit("usage: run_without_tensorflow.py [--hide name[,name]] script [arguments]")
    if any(m.split(".")[0] in names for m in sys.modules):
        sys.exit(f"already imported: {sorted(m for m in sys.modules if m.split('.')[0] in names)}")
    sys.meta_path[:] = [Hide(f, names) for f in sys.meta_path]
    script = args[0]
    sys.argv = args
    sys.path.insert(0, str(Path(script).resolve().parent))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
