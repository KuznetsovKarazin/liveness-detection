# M2 writers maintenance: safe outputs of `rgb_manifests.py` and `nuaa_manifests.py`, protected roots of `export_m2_rgb.py`

Task M2, request of 22 October 2026 on the M2 card, after the merge of PR #8 (`c83d17a`): secure the earlier M2
writers (`write_all` of `rgb_manifests.py` and `nuaa_manifests.py`) and verify and correct the protection of the roots
in `export_m2_rgb.py`, in a separate PR, before the next use of the earlier exports. Maintenance only: no new
scientific activity, **no data, weights, training or inference**; every test uses synthetic files in temporary
folders. The accepted packages and the historical results are unchanged and were not regenerated.

## Delivery status

This folder (`research/m2-writers-maintenance/`) is the content of the PR. The accepted packages
`research/m2-nuaa-manifests/`, `research/m2-rgb-manifests/` and `research/m2-c2-pilot-impl/` keep their copies as
delivered, with the defect described below; **the copies in this folder are the ones to use for future exports and
runs** of the M2 manifest commands.

## Defects addressed

Reproduced on synthetic fixtures in temporary folders (accepted copies → this folder):

| case | accepted copies | this folder |
|---|---|---|
| `write_all`, `<name>.tmp` pre-created as a symbolic link to an external file (`rgb_manifests.py`, `nuaa_manifests.py`) | external file overwritten; the output becomes the link | external file intact; output a new regular file |
| same, `<name>.tmp` as a hard link | external file overwritten | external file intact |
| `build`, a manifest pre-created as a symbolic or hard link to an external file (`write_manifest` wrote in place) | external file overwritten | external file intact (symbolic link refused and removed as a link; hard link replaced by name) |
| `check-overlap --out` is a symbolic link to an external folder | 4 reports written in the external folder | refused, nothing written |
| `check-overlap --rules <out>/overlap_report.json` | rules file removed by the invalidation of stale outputs | refused before any change, rules intact |
| `export_m2_rgb.py` run from a package layout with `--m2 research/m1-metrics` | the sibling package is deleted (\*) | refused: inside the work tree with the accepted packages |
| `check-overlap --self-test` (RGB) on manifests without `pooled_train` | `KeyError` traceback, exit 1, the four reports left with `passed: true` | explicit refusal (`self-test FAIL: … missing …`), exit 1, reports rewritten as FAIL, Markdown reports removed |
| `check-overlap --self-test` (NUAA) without the needed manifests | exit 1 with the reports left with `passed: true` | as above |
| an input named like a unique temporary of an output (`--rules <out>/.overlap_report.json.savedrules.tmp`; coordinator's review of 23 October) | accepted by the input check, then removed by the invalidation (`missing rules`, exit 1) | refused before any change, input byte-identical (same name rule for check and clean-up) |
| `<out>/overlap_report.md` replaced by a folder after a successful run (coordinator's review of 23 October) | invalidation stopped half-way: private JSON removed, public reports left PASS | refused before any change (folder, FIFO, socket at an output or temporary name), the earlier outputs left byte-identical |
| another spelling or alias of a clean-up name (`<out>/.OVERLAP_REPORT.JSON.savedrules.tmp`, `Overlap_Report.json`, `alias/..`; attack review of 23 October) | on a case-insensitive filesystem the check compared names with case and the clean-up removed the real file | refused before any change: identity with the real entries the clean-up would remove, and names compared without case |
| a manifest (or a file read by `report`, or a data-folder entry) that is a link to a clean-up name | accepted, its target removed by the clean-up | every top-level entry of an input folder (not recursive) and every file read is checked the same way; refused before any change |
| an output that cannot be removed (read-only output folder, immutable or append-only file) | partial invalidation, PASS left | refused before any change, nothing removed |
| a normally failing self-test (a faulty copy not rejected as expected) | `overlap_self_test.*` FAIL, but the four overlap reports stayed PASS | both overlap reports rewritten as FAIL (stage `self-test`), Markdown reports removed |
| any unexpected error of `build`, `check-overlap` or `report` (also `fail()` inside the self-test and Ctrl-C), also after writing its outputs | traceback; outputs already written stay (a PASS next to the failure) | explicit `FAIL` line; exit 1 (Ctrl-C: re-raised after the clean-up, Python's traceback and exit 130); the outputs of the command and the downstream ones are removed, `check-overlap` leaves only explicit FAIL reports; an error during the validation of the output folder removes nothing |

(\*) Latent: the published `research/m2-rgb-manifests/livenesslab/` stops earlier because four documents of the
development repository are missing (`required inputs missing`); on a scratch copy of the group tree at `c83d17a`, with
those documents added, the accepted export deleted `research/m1-metrics/` before running its first step. The root was
`Path(__file__).parents[1]`, i.e. the nested `livenesslab/`, and `path_guard.py` protects only inside the root it receives.

Sibling gaps closed in the same modules: `check-overlap` and `report` had no check of the output folder (only `build`
refused the data, weights and delivered results folders, and only when the output was inside them, not when it
contained them); the NUAA commands did not refuse an output folder holding the outputs of the CASIA-FASD/CelebA-Spoof
tranche (same report names); `export_m2_rgb.py` did not protect the data, weights and delivered results folders when
moved with the `LIVENESSLAB_*_DIR` environment variables.

## What changed

- **`scripts/m2_writers.py`** (new, standard library plus `path_guard.py`):
  - `write_all`: every temporary file is created with `tempfile.mkstemp` in the output folder (new name, `O_CREAT | O_EXCL`,
    no link followed), with the permissions `write_text` would give, then renamed with `os.replace`, which replaces the
    name and never the file a link pointed to. An output that is a symbolic link or not a regular file, and an output
    folder that is a symbolic link, are refused. On any exception no file of the group and no temporary file is left;
    links are removed as links, never their targets.
  - One name rule for the input check and the clean-up (coordinator's review of 23 October): `owned_name`/`owned_paths`
    give every path the clean-up of an output may touch (the output, the fixed `<name>.tmp` of the accepted copies, the
    unique `.<name>.*.tmp`); `touched_by` refuses any input (as written and resolved) that is one of them, or is inside a
    folder the command rewrites, before any change: by identity (device and inode) with the real entries the clean-up
    would remove, so another spelling on a case-insensitive filesystem or an alias is recognised, and by name without
    case for entries that do not exist yet; every top-level entry of an input folder (not recursive; for example a manifest that is a link) and
    every file a command reads (`report`: summary, aggregate, overlap report, self-test, manifests) is checked too. A preflight in `output_problem` refuses a folder, FIFO, socket or other
    non-regular object at any of those paths (and inside `<out>/manifests` for `build`), an output folder that does not
    accept writes (`os.access`), an immutable or append-only file (`st_flags`: `UF/SF_IMMUTABLE`, `UF/SF_APPEND`) and, on
    Windows, a read-only file, before anything is removed.
  - `invalidate_all`, used by the commands for the initial clean-up: if the invalidation stops half-way (including
    `SystemExit`), the clean-up is completed one name at a time (other outputs removed, JSON that cannot be removed
    overwritten with a FAIL document when the folder accepts writes) and the command stops with an explicit refusal
    (`the invalidation of the previous outputs stopped …`). Every later `SystemExit` is a deliberate FAIL that writes its
    own FAIL artefacts after a complete clean-up, so `main` keeps treating it as a refusal.
  - `invalidate_each`: the clean-up after a failure removes one output at a time; a JSON report that cannot be removed
    is overwritten with a FAIL document when the folder still accepts writes, otherwise it stays as it is and, like any
    other file left behind, is named (`could not invalidate <name> (…): left behind`).
  - `invalidate`: removes stale outputs and their temporaries (the fixed `<name>.tmp` of the accepted copies and the
    unique `.<name>.*.tmp`) without following links; a folder with an output name is refused.
  - `output_problem`, called by every writing command before any change: the output folder (and `<out>/manifests` for
    `build`) is not a symbolic link or a file; it is not inside, and does not contain, the work tree that holds the
    accepted packages (`outer_roots`: the git work tree containing `livenesslab/` when it is not its root, and every
    ancestor whose `research/` contains it, also without git, the same rule as `c2_pilot.py`); `path_guard.py` rules
    (filesystem root, home, repository root and their ancestors, protected folders; inside the development repository
    only `results/m2_rgb/` or `results/m2/`); it is not, does not contain and is not inside the data, weights and
    delivered results folders, also when moved with the environment variables; it is not equal to or inside an input
    folder; no input (rules, manifests folder, template, data and weights folders) is one of the paths the command
    writes or removes, or inside them. The manifests folder may stay inside the output folder (default layout
    `<out>/manifests`): the commands write only the declared names directly in the output folder.
- **`scripts/rgb_manifests.py`, `scripts/nuaa_manifests.py`**: `invalidate`, `write_all`, `write_manifest` and
  `write_rows` delegate to `m2_writers.py` (same bytes; manifests keep LF line ends); new `outputs_problem` called by
  `build`, `check-overlap` and `report` before any change; the self-test keeps its faulty copies and its outputs in
  separate temporary folders. The self-test is run under `except BaseException`: when it fails normally, raises any
  exception, calls `fail()` (`SystemExit`) or is interrupted (`KeyboardInterrupt`), `self_test_failure` rewrites both
  overlap reports as FAIL (stage `self-test`, the error text only in the restricted report), removes the Markdown
  reports and, when the self-test wrote nothing, writes a FAIL `overlap_self_test.json`; then `self-test FAIL: …` and
  exit 1, or the interruption is re-raised (exit 130). The RGB self-test refuses explicitly when a manifest of the
  dataset is missing (`self_test_faults`); the NUAA one raises the same refusal instead of exiting with the reports at
  PASS. `main` catches every exception except `SystemExit` (which `fail()` and refusals use); `unexpected_failure` acts
  only on the output folder validated by the command (`VALIDATED`, recorded by `outputs_problem` after the checks, so
  an error during the validation removes nothing): it removes the outputs of the command and the downstream ones
  (`build`: all; `check-overlap`: overlap reports, self-test and short reports; `report`: short reports), writes the two
  explicit FAIL reports for `check-overlap` (stage `error`), prints `<command> FAIL: unexpected error …` and exits 1, or
  prints `<command> FAIL: interrupted` and re-raises the interruption. `nuaa_manifests.py` also refuses an output folder holding the outputs of the
  CASIA-FASD/CelebA-Spoof tranche (`RGB_OUTPUTS`).
- **`scripts/export_m2_rgb.py`**: `export_paths_problem` (new, 23 October): the folders the export deletes and recreates
  (`--m2`, public, restricted) must not be symbolic links, and `--uploaded-sums` must not be inside them (it would be
  deleted before being read); refused before any change. `deletion_problem` also refuses every folder inside, or containing, the work tree with
  the accepted packages and every folder that is, contains or is inside the (possibly moved) data, weights and delivered
  results folders. Decision: the export is not refused as a whole outside the development repository; from a package
  layout every output inside the group tree is refused and scratch folders outside it stay allowed (a real export still
  needs the development repository: documents and data). `reference_state` and `reference_report` (the identity check of the shared files, moved out of `main` and tested
  end to end): `nuaa_manifests.py` is accepted as the
  copy of the NUAA package only with its delivered SHA-256 or with the exact SHA-256 of the corrected file plus an AST
  check (below). `CODE` lists `m2_writers.py` and its two checks; the export also runs `check_m2_writers.py`.
- **Tests**: `scripts/check_m2_writers.py` (new) and `scripts/check_m2_writers_mutations.py` (new); in
  `check_rgb_manifests.py` and `check_nuaa_manifests.py` the assertion on the fixed temporary names is replaced by one on
  unique names; in `check_export_m2_rgb.py` the identity of the shared files with the NUAA package accepts
  `nuaa_manifests.py` only through that rule; in `check_rgb_mutations.py` and `check_nuaa_mutations.py` the four
  mutations of `write_all`/`invalidate` (and one of the `--out` check of `check-overlap`) are rewritten on the new code,
  same counts.

## Changes declared against the accepted copies

`m2_writers.MAINTENANCE` records, for each changed module, the SHA-256 of the accepted copy, the exact SHA-256 of the
corrected file and the top-level functions or constants touched by this maintenance. Outside those (and the added
`import m2_writers as mw`) the AST of the corrected file must equal that of the accepted copy, read from the sibling
package and verified by its SHA-256; `check_m2_writers.py` proves it and that any other change is refused.

| file | accepted copy (SHA-256) | corrected file (SHA-256) | declared functions or constants |
|---|---|---|---|
| `scripts/rgb_manifests.py` | `61e8d1a6…` (`m2-rgb-manifests`, `m2-c2-pilot-impl`) | `4036a3fa…` | `invalidate`, `write_all`, `write_manifest`, `write_rows`, `outputs_problem` (new), `VALIDATED` (new), `cmd_build`, `self_test_faults`, `run_self_test`, `cmd_check_overlap`, `self_test_failure` (new), `cmd_report`, `main`, `unexpected_failure` (new) |
| `scripts/nuaa_manifests.py` | `bc1ce7b1…` (`m2-nuaa-manifests`, `m2-rgb-manifests`, `m2-c2-pilot-impl`) | `a719ce2e…` | the same except `self_test_faults`, plus `overlap_run` and `RGB_OUTPUTS` (new) |
| `scripts/export_m2_rgb.py` | `fa5b1eea…` (`m2-rgb-manifests`) | `86273885…` | `CODE`, `deletion_problem`, `export_paths_problem` (new), `reference_state` (new), `reference_report` (new), `main` |

Outputs of normal runs are unchanged: synthetic runs of the accepted copies and of this folder (`check-overlap` PASS,
preliminary FAIL and rule FAIL, `build` FAIL with its summary and manifests, NUAA `check-overlap` PASS and FAIL, `build`
without data, `write_manifest`) give identical files, after replacing only timestamps and temporary folder names. Two
behaviours change on purpose: a normally failing self-test now turns both overlap reports to FAIL (stage `self-test`)
and removes the Markdown reports (before, they stayed PASS next to the failed self-test); an unexpected error (or an
interruption) now removes the outputs of the command and the downstream ones and, for `check-overlap`, writes explicit
FAIL reports (before: a traceback and the outputs already written). The self-test on a world without the needed manifests
is one such case (crash before, explicit refusal now).

## Affected accepted packages (unchanged)

| package | copies with the known defect | effect |
|---|---|---|
| `research/m2-nuaa-manifests/` | `nuaa_manifests.py` | its `build`, `check-overlap`, `report` |
| `research/m2-rgb-manifests/` | `rgb_manifests.py`, `nuaa_manifests.py`, `export_m2_rgb.py` | its commands and export |
| `research/m2-c2-pilot-impl/` | `rgb_manifests.py`, `nuaa_manifests.py` | none on the C2 path: `c2_pilot.py` only reads and validates manifests with them and uses its own `write_all` (corrected on 21 October) |

The defect acts only when a link was created beforehand at an output or temporary name in the output folder (or the
output folder itself is a link); the delivered results were not re-examined or regenerated here. `select_threshold.py` (frozen, shared by several packages) and
the other writers of the repository were audited and are not changed here.

## Verification without data or weights

From `research/m2-writers-maintenance/livenesslab/` (Python 3.11 or later with the packages of `requirements.txt`;
data, weights and results folders are not needed):

```bash
python scripts/check_m2_writers.py
python scripts/check_m2_writers_mutations.py
python scripts/check_rgb_manifests.py
python scripts/check_nuaa_manifests.py
python scripts/check_export_m2_rgb.py
python scripts/check_rgb_mutations.py
python scripts/check_nuaa_mutations.py
```

Expected last lines:

- `check_m2_writers.py`: `tests run: 43, failures: 0, errors: 0, skipped: 0`, `RESULT: PASS`
- `check_m2_writers_mutations.py`: `mutations killed: 104/104`, `control mutations survived: 1/1`, `RESULT: PASS` (in each
  mutated copy the pinned SHA-256 of the mutated file is re-pinned, so every mutation is killed by behaviour; the
  control mutation changes only a comment and must survive)
- `check_rgb_manifests.py`: `tests run: 57, failures: 0, errors: 0, skipped: 2`, `RESULT: PASS` (skipped: the two tests on the restricted manifests)
- `check_nuaa_manifests.py`: `tests run: 69, failures: 0, errors: 0, skipped: 7`, `RESULT: PASS` (skipped: the tests on the restricted manifests)
- `check_export_m2_rgb.py`: `tests run: 29, failures: 0, errors: 0, skipped: 2`, `RESULT: PASS` (skipped: development-only exports and restricted register drafts)
- `check_rgb_mutations.py`: `mutations killed: 58/58`, `RESULT: PASS`
- `check_nuaa_mutations.py`: `mutations killed: 24/24`, `RESULT: PASS`

Every skip is a declared missing input, printed with its reason; a skip is never counted as a killed mutation. Inside
the temporary copy of `check_rgb_mutations.py` the export suite reports 3 skips instead of 2: the copy is not next to
`m2-nuaa-manifests/`, so the comparison of the shared files is skipped there (declared missing input); the suite run
from this folder performs it.
`check_m2_writers.py` needs the sibling accepted packages (`../../m2-nuaa-manifests/`, `../../m2-rgb-manifests/`) for
the declaration check, as in `research/` of the group repository; the mutation check refuses to run without them.

## Test logs

`test_logs/` holds the output of the seven commands above, run from this layout (scratch copy of `research/` at
`c83d17a` with this folder, in a git work tree); `check_m2_writers_mutations.txt` comes from the run in the development
repository on the same code (identical files, the output prints no path).

## Not done, limits

- Run on macOS (arm64, Python 3.11.16, NumPy 1.26.4, scikit-learn 1.3.2), also on a case-sensitive APFS volume and in a
  reduced environment (`env -i`, `LC_ALL=C`, unusual time zone, `umask 077`, temporary folder with a space). Also run on
  Linux from this layout (`python:3.12` container, aarch64, NumPy 2.3.5, scikit-learn 1.8.0): `SHA256SUMS` 41/41,
  `check_m2_writers.py` 43 tests and 0 skipped, `check_rgb_manifests.py` 57 (2 skipped), `check_nuaa_manifests.py` 69
  (7 skipped), `check_export_m2_rgb.py` 29 (2 skipped), mutations 104/104 with `control mutations survived: 1/1`, 58/58,
  24/24, work tree clean afterwards (the previous version, 32 tests and 84/84, was also run on Linux x86_64 by the
  coordinator, review of 23 October). The
  simulated parts of the flag and permission tests (`os.lstat` reporting `UF_IMMUTABLE`/`UF_APPEND`, `os.access`
  refusing writes) run on every platform; the real parts (`chflags uchg` where `os.chflags` exists, a folder with mode
  555 when not running as root) are extra assertions inside the same tests, run only where available, never a skip, so
  the suite reports `skipped: 0` everywhere. The Windows read-only branch is not run.
- Interruptions: a Ctrl-C during the command is handled (outputs cleaned, then the interruption is re-raised, so Python
  prints its `KeyboardInterrupt` traceback and exits with 130); a second Ctrl-C during that clean-up can interrupt it and
  leave outputs behind (a limit of Python signal handling); a Ctrl-C followed by an error inside the clean-up of the
  self-test exits with 1, with no PASS left unless the output folder refuses writes (see below). If an output cannot be
  removed during the clean-up (for example a permission error), a JSON report is overwritten with a FAIL document when
  the folder still accepts writes; if the folder itself refuses writes (e.g. a read-only output folder) the JSON
  reports stay as they are, possibly with `passed: true`, and each one is named in a `could not invalidate <name> (…):
  left behind` line followed by the clean-up failure line; the exit code is still 1.
- A process that changes the output folder concurrently, between the checks and the writes, is not addressed beyond
  the exclusive temporary files and `os.replace`.
- The accepted packages and their results stay as delivered; no export was rerun.
- The other writers of the repository (`select_threshold.py`, frozen, and scripts outside M2) were audited and not
  changed in this PR.

## Files and integrity

`SHA256SUMS` lists every file of this folder except itself (`shasum -a 256 -c SHA256SUMS`): this `README.md`,
`test_logs/` and `livenesslab/` with the code copied from the development repository: `requirements.txt`,
`src/tesi_app/{__init__,paths}.py`, `scripts/{m2_writers,rgb_manifests,nuaa_manifests,export_m2_rgb,path_guard,
select_threshold,package_rules,export_helpers}.py`, the checks `scripts/{check_m2_writers,check_m2_writers_mutations,
check_rgb_manifests,check_nuaa_manifests,check_export_m2_rgb,check_rgb_mutations,check_nuaa_mutations}.py`,
`scripts/m2_inventory/` (six files, unchanged), `delivery/m2/{nuaa_overlap_rules.json,report.md.tmpl}` and
`delivery/m2/rgb/` (README template, two rule files, two report templates, naming schema and transformations,
unchanged). Unchanged shared copies are byte-identical to those of the accepted packages.
