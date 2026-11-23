"""Mutation check: each mutation, applied alone to a temporary copy of the scripts, must make check_synthetic.py
fail. Mutations 1-24 are those of PROTOCOL.md section 7; 25-56 were added after the independent code reviews.
Invented data only; the files of this folder are never modified."""
import sys

sys.dont_write_bytecode = True

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

FILES = ('session_compare.py', 'session_summary.py', 'check_synthetic.py', 'session_config.json')
COMPARE, SUMMARY = 'session_compare.py', 'session_summary.py'
GROUP_OF = "group_of = {v: g for g, values in cfg['groups'].items() for v in values}\n"
THRESHOLDS = 'thresholds = {g: cut[state, col] for g in GROUPS}'
UP = 'up = sum(1 for c in signs if c > 0)'
DOWN = 'down = sum(1 for c in signs if c < 0)'
MEANS_CELL = "cells += [pp(values['sessions_1_2_mean']), pp(values['session_3_mean']),"
PP_ROUND = 'cents = int(round(value * 100, 2) * 100)'
UNITS_CELL = "f\"{r[metric+'_units_up']}/{r[metric+'_units_zero']}/{r[metric+'_units_down']}\""
UNITS_SWAPPED = "f\"{r[metric+'_units_down']}/{r[metric+'_units_zero']}/{r[metric+'_units_up']}\""
NOTE = '{expected[f, "photograph", GROUPS[0]][0]:,} and {expected[f, "photograph", GROUPS[1]][0]:,}'
NOTE_SWAPPED = '{expected[f, "photograph", GROUPS[1]][0]:,} and {expected[f, "photograph", GROUPS[0]][0]:,}'
RATES = ("        row[PREFIX[g]+'_apcer'] = rate(c['photograph_errors'], c['n_photograph'])\n"
         "        row[PREFIX[g]+'_bpcer'] = rate(c['bona_fide_errors'], c['n_bona_fide'])\n")
RATES_SWAPPED = ("        row[PREFIX[g]+'_apcer'] = rate(c['bona_fide_errors'], c['n_bona_fide'])\n"
                 "        row[PREFIX[g]+'_bpcer'] = rate(c['photograph_errors'], c['n_photograph'])\n")


def remove(*names):
    """The functions return at once, at every call site (session_summary.py uses the checks of session_compare.py)."""
    return ('remove', names)


def remove_check(n):
    names = [f'check_c{n}'] + ([f'check_c{n}_file'] if n in (6, 12) else [])
    return (f'remove check C{n}', COMPARE, remove(*names))


MUTATIONS = [remove_check(n) for n in range(1, 13)] + [
    # PROTOCOL.md section 7, continued.
    ('APCER with < instead of <=', COMPARE,
     [('int(np.count_nonzero(s[y==1]<=t))', 'int(np.count_nonzero(s[y==1]<t))')]),
    ('BPCER with >= instead of >', COMPARE,
     [('int(np.count_nonzero(s[y==0]>t))', 'int(np.count_nonzero(s[y==0]>=t))')]),
    ('threshold 0.5 instead of the one of score_metrics.csv', COMPARE,
     [(THRESHOLDS, 'thresholds = {g: 0.5 for g in GROUPS}')]),
    ('threshold of session 3 taken from the other BN state', COMPARE,
     [(THRESHOLDS, 'thresholds = {GROUPS[0]: cut[state, col], GROUPS[1]: cut[STATES[1 - si], col]}')]),
    ('threshold of session 3 taken from the other target', COMPARE,
     [(THRESHOLDS, 'thresholds = {GROUPS[0]: cut[state, col], '
                   'GROUPS[1]: cut[state, TARGET_COLS[1 - TARGET_COLS.index(col)]]}')]),
    ('sessions 1-2 and session 3 swapped', COMPARE,
     [(GROUP_OF, GROUP_OF.replace('{v: g for', '{v: GROUPS[1 - GROUPS.index(g)] for'))]),
    ('session 03 read as sessions 1-2', COMPARE, [(GROUP_OF, GROUP_OF + "    group_of['03'] = GROUPS[0]\n")]),
    ('session 01 read as session 3', COMPARE, [(GROUP_OF, GROUP_OF + "    group_of['01'] = GROUPS[1]\n")]),
    ('frame-weighted mean instead of weight 1/12', SUMMARY,
     [('diff_mean = sum(diffs, Fraction(0)) / len(diffs)',
       'diff_mean = Fraction(sum(e3s), sum(n3s)) - Fraction(sum(e12s), sum(n12s))')]),
    ('development BPCER recomputed as (200 - k)/200', COMPARE,
     [("b=1.0-np.searchsorted(genuine,thr,side='right')/len(genuine)",
       "b=(len(genuine)-np.searchsorted(genuine,thr,side='right'))/len(genuine)")]),
    ('difference sessions 1-2 minus session 3', SUMMARY,
     [('diffs = [a3 - a12 for a12, a3 in zip(r12, r3)]', 'diffs = [a12 - a3 for a12, a3 in zip(r12, r3)]')]),
    ('unit with zero difference counted as up', SUMMARY, [(UP, UP.replace('c > 0', 'c >= 0'))]),
    # Independent code review, M1-M13 (session_summary.py).
    ('M1 table: group means swapped', SUMMARY,
     [(MEANS_CELL, "cells += [pp(values['session_3_mean']), pp(values['sessions_1_2_mean']),")]),
    ('M2 table: no conversion x 100', SUMMARY, [(PP_ROUND, 'cents = int(round(value, 2) * 100)')]),
    ('M3 table: rounding on the float', SUMMARY, [(PP_ROUND, 'cents = int(round(float(value) * 100, 2) * 100)')]),
    ('M4 table: difference without sign', SUMMARY,
     [("pp(values['mean_difference'], signed=True),", "pp(values['mean_difference'], signed=False),")]),
    ('M5 table: BN state labels swapped', SUMMARY,
     [("r['target'][:-3], STATE_LABEL[r['bn_state']]]",
       "r['target'][:-3], STATE_LABEL[STATES[1 - STATES.index(r['bn_state'])]]]")]),
    ('M6 table: APCER and BPCER columns swapped', SUMMARY,
     [("        for metric in ('apcer', 'bpcer'):\n            exact = r['_'+metric]\n",
       "        for metric in ('bpcer', 'apcer'):\n            exact = r['_'+metric]\n")]),
    ('M7 table: units up and down swapped', SUMMARY, [(UNITS_CELL, UNITS_SWAPPED)]),
    ('M8 table: groups swapped in the denominator note', SUMMARY, [(NOTE, NOTE_SWAPPED)]),
    ('M9 figure: metric of the panel swapped', SUMMARY,
     [("exact = by[p, net, col, state]['_'+metric]",
       "exact = by[p, net, col, state]['_'+('bpcer' if metric == 'apcer' else 'apcer')]")]),
    ('M10 figure: offset by seed instead of fold', SUMMARY,
     [("panel['points'].append((x + OFFSETS[f],", "panel['points'].append((x + OFFSETS[seed - 41],")]),
    ('M11 figure: mean dash at the session 3 mean', SUMMARY,
     [("float(100 * exact['exact']['mean_difference'])", "float(100 * exact['exact']['session_3_mean'])")]),
    ('M12 existing output files not refused', SUMMARY, remove('check_outputs')),
    ('M13 units up and down swapped in the count', SUMMARY,
     [(UP, UP.replace('c > 0', 'c < 0')), (DOWN, DOWN.replace('c < 0', 'c > 0'))]),
    # Independent code review, M14-M31 (session_compare.py).
    ('M14 C5 identifier fields not checked', COMPARE,
     [('        if not subject == match.group(2) == match.group(3):', '        if False:')]),
    ('M15 C5 class prefix not checked', COMPARE,
     [("        if (match.group(1) == 'attack') != (int(label) == 1):", '        if False:')]),
    ('M16 C5 form of sample_id not checked', COMPARE, [('        if match is None:', '        if False:')]),
    ('M17 C6 identifier counts not checked', COMPARE,
     [('(identifiers is not None and identifiers != expected[cell][1])', 'False')]),
    ('M18 C6 SHA-256 of expected_counts.csv not checked', COMPARE,
     [("    if digest != cfg['expected_counts']['sha256']:\n"
       "        raise CheckFailed('C6', 'SHA-256 of expected_counts.csv differs from the configuration')\n", '')]),
    ('M19 C1 unique and expected keys not checked', COMPARE,
     [('    if len(keys) != 96 or len(set(keys)) != 96 or set(keys) != set(itertools.product(\n'
       '            PATIENCES, FOLDS, SEEDS, NETWORKS, STATES)):', '    if len(keys) != 96:')]),
    ('M20 C2 third BN state not checked', COMPARE,
     [("    if role == 'test' and not {r.get('bn_state') for r in file_rows} <= set(STATES):\n"
       "        raise CheckFailed('C2', 'unexpected BN state in a test file', **key)\n", '')]),
    ('M21 C3 development pairing not checked', COMPARE,
     [("    if refs.setdefault(('dev', fold), dev_pair) != dev_pair:\n"
       "        raise CheckFailed('C3', 'development frames, identifiers or labels differ across conditions', **key)\n",
       '')]),
    ('M22 C4 one hash per state not checked', COMPARE,
     [("    if len(hashes) != 1:\n"
       "        raise CheckFailed('C4', 'test and development checkpoint hashes differ', **key)\n", '')]),
    ('M23 C7 development APCER not compared', COMPARE,
     [("[('threshold',t),('dev_apcer',da[idx]),('dev_bpcer',db[idx])]", "[('threshold',t),('dev_bpcer',db[idx])]")]),
    ('M24 C12 mean in (seed, fold) order', COMPARE,
     [("(p, net, state, col)), key=lambda r: (r['fold'], r['seed']))",
       "(p, net, state, col)), key=lambda r: (r['seed'], r['fold']))")]),
    ('M25 C12 mean as a plain sum', COMPARE,
     [("mean = float(np.array([r['pooled_'+metric] for r in units]).mean())",
       "mean = sum(r['pooled_'+metric] for r in units) / 12")]),
    ('M26 C2 scores up to 2 accepted', COMPARE,
     [('scores.min() < 0 or scores.max() > 1:', 'scores.min() < 0 or scores.max() > 2:')]),
    ('M27 thresholds of both groups from the other BN state', COMPARE,
     [(THRESHOLDS, 'thresholds = {g: cut[STATES[1 - si], col] for g in GROUPS}')]),
    ('M28 classes swapped', COMPARE,
     [("CLASSES = (('photograph', 1), ('bona fide', 0))", "CLASSES = (('photograph', 0), ('bona fide', 1))")]),
    ('M29 APCER and BPCER swapped in the rows', COMPARE, [(RATES, RATES_SWAPPED)]),
    ('M30 C7 selection with < instead of <=', COMPARE,
     [('idx=np.where(da<=target)[0][-1]', 'idx=np.where(da<target)[0][-1]')]),
    ('M31 C7 against the row of another seed', COMPARE,
     [('check_c7(dy, ds, sm[p, f, seed, net, state], key)', 'check_c7(dy, ds, sm[p, f, SEEDS[0], net, state], key)')]),
    # Confirmation of the code review.
    ('C3 count of 400 development frames not checked', COMPARE, [(' or len(dev_pair[2]) != 400', '')]),
]


def mutate(text, change):
    if isinstance(change, tuple) and change[0] == 'remove':
        for name in change[1]:
            head = f'def {name}('
            if text.count(head) != 1:
                raise SystemExit(f'Mutation harness: {head} not found exactly once')
            end = text.index('\n', text.index(head)) + 1
            text = text[:end] + '    return\n' + text[end:]
        return text
    for old, new in change:
        if text.count(old) != 1:
            raise SystemExit(f'Mutation harness: target text not found exactly once: {old}')
        text = text.replace(old, new)
    return text


def synthetic(folder):
    """Run check_synthetic.py; return whether it passed and the line of its first failure (or its last line)."""
    proc = subprocess.run([sys.executable, '-I', '-B', str(folder/'check_synthetic.py'), '--package', str(folder)],
                          capture_output=True, text=True, timeout=1800)
    lines = (proc.stdout + proc.stderr).strip().splitlines() or ['']
    failure = next((line for line in lines if line.startswith('Synthetic check FAILED')), lines[-1])
    return proc.returncode == 0, failure


def own_case(number, failure):
    """A removal of check Cn (mutations 1-12) counts as killed by its own stop case only if the first failure of
    check_synthetic.py is a stop case of Cn."""
    return re.match(rf'Synthetic check FAILED: stop case \d+ C{number} \(', failure) is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    package = parser.parse_args().package.resolve()
    for _, target, change in MUTATIONS:  # every target text exists once before anything runs
        mutate((package/target).read_text(encoding='utf-8'), change)
    with TemporaryDirectory() as temp:
        folder = Path(temp)/'baseline'
        folder.mkdir()
        for name in FILES:
            shutil.copy(package/name, folder/name)
        passed, last = synthetic(folder)
        if not passed:
            raise SystemExit('Unmutated scripts fail check_synthetic.py: ' + last)
        print('unmutated: check_synthetic.py passed', flush=True)
        killed, own = [], []
        for number, (label, target, change) in enumerate(MUTATIONS, 1):
            folder = Path(temp)/f'mutation-{number:02d}'
            folder.mkdir()
            for name in FILES:
                shutil.copy(package/name, folder/name)
            text = mutate((folder/target).read_text(encoding='utf-8'), change)
            compile(text, target, 'exec')
            (folder/target).write_text(text, encoding='utf-8')
            passed, failure = synthetic(folder)
            killed.append(not passed)
            verdict = 'killed' if not passed else 'SURVIVED'
            if number <= 12:
                own.append(not passed and own_case(number, failure))
                verdict += ' by its own stop case' if own[-1] else ' NOT by its own stop case'
            print(f'mutation {number:2d} ({label}): {verdict} | {failure[:200]}', flush=True)
    print(f'mutations killed: {sum(killed)}/{len(MUTATIONS)}')
    print(f'removal mutations killed by their own stop case: {sum(own)}/12')
    sys.exit(0 if all(killed) and sum(own) == 12 else 1)


if __name__ == '__main__':
    main()
