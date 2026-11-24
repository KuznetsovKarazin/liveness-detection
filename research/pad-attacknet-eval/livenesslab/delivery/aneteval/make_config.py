"""Genera delivery/aneteval/pad_anet_eval.json con gli SHA-256 attuali dei file fissati e i testi della configurazione
della valutazione C5. Prima dei sei run i run sono «RUNS_PENDING»; dopo i run, con --runs-root, per ciascuno dei sei run
si registrano l'ultimo tentativo nella radice dei run e gli SHA-256 del suo status.json e dei due SHA256SUMS (letti dai
file, mai scritti a mano):

  .venv/bin/python -I -B delivery/aneteval/make_config.py
  .venv/bin/python -I -B delivery/aneteval/make_config.py --runs-root results/anetrun --force [--reference <file>]

Con --runs-root si ferma se un campo diverso da runs differisce dalla configurazione di riferimento (published_config).
Non sovrascrive una configurazione diversa senza --force; stampa lo SHA-256 della configurazione scritta."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import pad_anet_eval as V  # noqa: E402
import pad_anet_run as NR  # noqa: E402

OUT = ROOT / "delivery/aneteval/pad_anet_eval.json"
A = "a choice of the author"
COMPARISON = "delivery/anetcmp/pad_anet_comparison.json"
RUNCFG = "delivery/anetrun/pad_anet_run.json"
PROTOCOL = "delivery/pad-attacknet-run/pr34_amended/PROTOCOL.md"
C2_UNITS = "delivery/pad-arch-eval-results/runs/20261121-ARCHEVAL-332fbdf/public/units.csv"
C2_RESTRICTED = "results/archeval/20261121-ARCHEVAL-332fbdf/restricted"
C2_RESTRICTED_SUMS = "d69c29e632a6fbe1cec42b166fa378e8ed38030353ae6c19c25ab7adbf0e42a7"     # dalla configurazione della PR A, c2_results


def sha(p) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def runs_record(runs_root: Path) -> dict:
    """Per ciascuno dei sei run l'ultimo tentativo nella radice dei run e gli SHA-256 dei suoi file di stato."""
    out = {}
    for r in NR.RUNS:
        rid = V.latest_attempt(r, runs_root)
        f = runs_root / rid
        st = json.loads((f / "status.json").read_text(encoding="utf-8"))
        out[r] = {"run_id": rid, "status_sha256": hashlib.sha256((f / "status.json").read_bytes()).hexdigest(),
                  "public_sha256sums_sha256": st["public_sha256sums_sha256"], "restricted_sha256sums_sha256": st["restricted_sha256sums_sha256"]}
    return out


def doc(data_sha=sha, code_sha=sha, runs=V.RUNS_PENDING, RECORDS="delivery/pad-attacknet-eval/runs", COMPARISON=COMPARISON, RUNCFG=RUNCFG, PROTOCOL=PROTOCOL, C2_UNITS=C2_UNITS,
        C2_RESTRICTED=C2_RESTRICTED, C2_RESTRICTED_SUMS=C2_RESTRICTED_SUMS) -> dict:
    sha = data_sha
    return {
        "schema": V.CONFIG_SCHEMA,
        "task": ("The evaluation on the NUAA test of the C5 comparison pre-registered in PR #34 (PROTOCOL.md §6, amended in PR #37): the "
                 "evaluable new models of AttackNetV1, AttackNetV2_1 and AttackNetV2_2 of the six runs and the 48 C2 references (LivenessNet "
                 "and ResNet-18, their test scores of the C2 evaluation 20261121-ARCHEVAL-332fbdf), with the definitions of PR A "
                 "(pad_anet_definitions): primary analysis at patience 4 and threshold 0.5, descriptive conditions, rankings, "
                 "denominators, descriptive comparison with ResNet-18, guardian per network"),
        "authorisation": ("PROTOCOL.md §11: the evaluation code and configuration are published, with the link on the C5 card, before the "
                          "NUAA test of the new models is read; evaluate runs once, after the six runs and after this configuration records "
                          "them"),
        "operations": ("after the six runs: copy_run_records.py --runs-root results/anetrun --conditions-dir <folder of the launch logs> "
                       "copies the records of the runs into this package; make_config.py --runs-root results/anetrun records the last "
                       "attempt of each run and stops if any other field differs from the reference configuration: that of the last "
                       "commit (git show HEAD), or the file given with --reference (the configuration of the published head of this pull "
                       "request, saved with git show <commit>:delivery/aneteval/pad_anet_eval.json); the comparison guarantees that only "
                       "runs changed since the reference, not that the reference was published; commit; the completed configuration and "
                       "package published (a commit of this pull request or a new pull request) and linked on the C5 card; only then "
                       "check-inputs and evaluate, launched from a zsh with setopt nobgnice through delivery/aneteval/launch.zsh "
                       "(ANETEVAL_LOG_DIR=<a folder outside the repository> [ANETEVAL_ATTEMPT=1] nohup zsh delivery/aneteval/launch.zsh "
                       "check-inputs | evaluate > <a log file outside the repository> 2>&1 &), which writes a conditions file per command; "
                       "the launch script stops without launching the command if it is not run from the root of the repository, if "
                       "ANETEVAL_LOG_DIR is missing or inside the repository, if the command is neither check-inputs nor evaluate, if "
                       "evaluate has no ANETEVAL_ATTEMPT, or if the conditions file of the same label already exists (exit 9); it also stops, "
                       "with the reason in the conditions file, if its nice value is not 0, if pmset -g batt does not say AC Power "
                       "(evaluate only), or if check-machine of the runs or check-inputs of the evaluation fails (exit 1); the launch script never passes "
                       "--second-reading: a second reading, if ever needed, is launched by hand with --second-reading under after_stop; "
                       "evaluate once; the config_sha256 and code_sha256 recorded by evaluate are those of the "
                       "published files (PROTOCOL.md §11); one heavy process at a time"),
        "comparison": {"path": COMPARISON, "sha256": sha(COMPARISON)},
        "run_configuration": {"path": RUNCFG, "sha256": sha(RUNCFG)},
        "protocol": {"path": PROTOCOL, "sha256": sha(PROTOCOL)},
        "c2_evaluation": {"run_id": "20261121-ARCHEVAL-332fbdf", "units": {"path": C2_UNITS, "sha256": sha(C2_UNITS)},
                          "restricted_folder": C2_RESTRICTED, "restricted_sha256sums_sha256": C2_RESTRICTED_SUMS},
        "code": {f: code_sha(f) for f in V.CODE_FILES},
        "runs": runs,
        "run_records": {"folder": RECORDS, "files": list(V.RECORD_FILES),
                        "rule": ("as for C2 (delivery/pad-arch-eval/runs/): the published records of every attempt of the six runs and of every "
                                 "check-estimator of the runs root, status.json, run.log and public/ (nothing of restricted/), with conditions.txt, "
                                 "the conditions file written by delivery/anetrun/launch.zsh, copied unchanged by "
                                 "delivery/aneteval/copy_run_records.py; check-inputs and evaluate check them byte for byte against the runs "
                                 "root and scan them for text a publishable output must not hold (check-inputs and copy_run_records.py without "
                                 "the manifests: personal paths and texts, not subject or frame identifiers; evaluate with the manifests "
                                 "loaded; for run.log and public/ the scan repeats that of the runner, made with the real inputs)")},
        "attempt_rule": ("amended PROTOCOL.md §6 (k)/(n): for each of the six runs only the last attempt is evaluated, which must be PASS with "
                         "phase 1 complete for the 12 positions; the configuration records its run id and the SHA-256 of its status.json and "
                         "of its two SHA256SUMS, and evaluate checks that no later attempt exists in the runs root"),
        "after_stop": (f"{A}, as in C2: no automatic resume; an evaluation that stops keeps its folder and outcome; test_read_started is "
                       "recorded in status.json (false at the start, rewritten true before the fold schema and any test frame); a new attempt "
                       "after an attempt with test_read_started, or one left RUNNING, or with a non-empty restricted/, or an unreadable one, "
                       "is a second "
                       "reading of the test, refused unless --second-reading, given only after the outcome has been reported on the C5 card "
                       "and recorded in status.json and in the results; after the reading of the test the restricted test scores already "
                       "computed are kept, with their SHA256SUMS, also for FAIL and INCOMPLETE"),
        "dev_reproduction": ("before any test frame of the new models: each evaluable model reloaded (checkpoint SHA-256 checked) and checked "
                             "against the registry, its dev scored in both BN states (the saved re-estimated statistics installed) and compared "
                             "with the dev scores saved by the run (pad_arch_run.evaluator_agreement: tolerances of PR #27); one model out: "
                             "MISMATCH, the test is not read"),
        "c2_reproduction": ("before any test frame of the new models: from the 48 C2 test score files (each equal to the anchored restricted "
                            "SHA256SUMS of the C2 evaluation) the primary analysis (pad_arch_definitions.primary_analysis), the 8 conditions "
                            "and units.csv of C2 are recomputed and compared for exact equality with results.json and units.csv of the C2 "
                            "evaluation (+0.0998 [0.0256, 0.1632], 926 valid resamples); a difference: FAIL; after the analysis the "
                            "descriptive comparison of LivenessNet with ResNet-18 must be the exact opposite of the C2 gaps and contrast"),
        "test": ("after every dev reproduction and a re-verification: test_read_started written to status.json; the fold schema; the "
                 "frame identifiers of the 48 C2 files compared with the test keys of their fold; per fold the test frames assembled once and scored by "
                 "its evaluable models in both BN states; the 48 C2 references are not re-evaluated (their saved test scores are used)"),
        "non_finite_test": ("PROTOCOL.md §6 (n): a model with non-finite test scores in either BN state is reported with the count of such "
                            "frames; the evaluation stops before the primary analysis (outcome NON_FINITE_TEST, the restricted test scores kept in "
                            "restricted/ with their SHA256SUMS, no public result); no model is excluded after its test scores are computed; a new "
                            "attempt is a second reading (after_stop)"),
        "analysis": ("pad_anet_definitions.primary_analysis and descriptive_analysis on the units of the 8 conditions (the guardian inside: "
                     "an AttackNet with fewer than 12 pairs evaluable in both BN states is 'not computed' with the reason of its not "
                     "evaluable models)"),
        "cap": {"hours": 2.0, "unit": V.CAP_UNIT,
                "rule": (f"{A}: three times the evaluation hours of plan/run_plan.md of PR A (0.59 h), rounded up to a multiple of 0.5 h; "
                         "checked before the dev reproduction of each model, before the test assembly of each fold, before the test scoring "
                         "of each model and before the analysis; at the cap or by SIGUSR1: INCOMPLETE, no result")},
        "outputs": {"run": [V.STATUS, V.LOG], "public": list(V.PUBLIC_OUTPUTS), "restricted": [f"<run>_<model>{V.TEST_SUFFIX}", "SHA256SUMS"]},
        "choices_of_the_author": {
            "thresholds_source": f"{A}: the thresholds of the new models from their saved dev scores (pad_arch_eval.load_dev), as in C2",
            "c2_reproduction_gate": f"{A}: the reproduction of C2 from the 48 files compared for exact equality before the test (c2_reproduction)",
            "c2_units_bytes": f"{A}: the units.csv of C2 recomputed and compared byte for byte",
            "non_finite_test_outcome": f"{A}: the outcome name NON_FINITE_TEST (exit code 6) for the stop of PROTOCOL.md §6 (n)",
            "reused_thresholds": f"{A}: the thresholds of the 48 references from their C2 test score files (threshold_dev_eer of the file)",
            "units_order": f"{A}: the units of each condition in (fold, seed) order, the order of C2 units",
            "cap": f"{A}: the cap of evaluate (cap.rule)",
            "attempts": f"{A}: --attempt N required, a new attempt in a new folder; the official output root {V.DEFAULT_OUT_ROOT}",
            "outputs": (f"{A}: the names and columns of the outputs; table.csv: per network and BN state the mean over the units of the "
                        "per-unit APCER, BPCER and ACER of the primary condition, with units and denominators"),
            "runs_record": f"{A}: the six runs recorded after the runs by make_config.py --runs-root (RUNS_PENDING before)",
            "second_reading": f"{A}: the option --second-reading and its rule (after_stop)",
            "run_records": f"{A}: the records of the runs published in this package, as for C2, with conditions.txt (run_records)",
            "launch_conditions": f"{A}: the launch script delivery/aneteval/launch.zsh and its conditions file, as for the runs (operations)",
            "test_one_class": (f"{A}: the form of the count of PROTOCOL.md §6 (f) on the test: per network x patience x BN state x threshold, "
                               "the models (units) deciding one class for every test frame, in results.json (test_one_class)")},
        "test_read": V.TEST_READ,
    }


def published_differences(new: dict, published: dict) -> list:
    """I campi della configurazione nuova diversi da quelli della configurazione pubblicata, salvo runs (l'unico campo che
    si completa dopo i run)."""
    keys = sorted(set(new) | set(published))
    return [k for k in keys if k != "runs" and new.get(k) != published.get(k)]


def published_config(path=None) -> dict:
    """La configurazione di riferimento: quella dell'ultimo commit (git show HEAD:delivery/aneteval/pad_anet_eval.json), o
    il file dato con --reference (per esempio la configurazione della testa pubblicata della PR, salvata con git show).
    Il confronto garantisce che solo runs è cambiato rispetto al riferimento, non che il riferimento sia stato pubblicato."""
    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    r = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:delivery/aneteval/pad_anet_eval.json"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("the configuration of the last commit cannot be read (git show HEAD:delivery/aneteval/pad_anet_eval.json)")
    return json.loads(r.stdout)


def main() -> int:
    runs = V.RUNS_PENDING
    d = None
    if "--runs-root" in sys.argv:
        root = Path(sys.argv[sys.argv.index("--runs-root") + 1])
        runs = runs_record(root if root.is_absolute() else ROOT / root)
        d = doc(runs=runs)
        ref_path = sys.argv[sys.argv.index("--reference") + 1] if "--reference" in sys.argv else None
        ref = published_config(ref_path)
        what = f"the reference {Path(ref_path).name}" if ref_path else "the configuration of the last commit (git show HEAD)"
        if diff := published_differences(d, ref):
            print(f"refused: the fields {', '.join(diff)} differ from {what} (only runs may be completed after the runs)")
            return 1
        print(f"compared with {what}: only runs differs")
    text = json.dumps(d or doc(runs=runs), indent=1, ensure_ascii=False) + "\n"
    if OUT.exists() and OUT.read_text(encoding="utf-8") != text and "--force" not in sys.argv:
        print(f"{OUT.name} exists and differs: pass --force to rewrite it")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)} {hashlib.sha256(text.encode('utf-8')).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
