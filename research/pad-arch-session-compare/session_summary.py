"""Equal-weight summary of session_counts.csv: one CSV, one table and one figure. Reads public files only.

Inputs: session_counts.csv written by session_compare.py, score_metrics.csv and paired_summary.csv of pull request
#36, the registered session_config.json and expected_counts.csv. Checks the public inputs first and repeats checks
C1, C6 (frames only: session_counts.csv has no identifier counts), C8, C9, C10 and C12 with the functions of
session_compare.py; then builds session_summary.csv, session_table.md and session_differences.pdf and writes the three
files only when all of them have been built.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import io
import itertools
import os
import tempfile
import warnings
from fractions import Fraction
from pathlib import Path


def _load_compare():
    path = Path(__file__).with_name('session_compare.py')
    spec = importlib.util.spec_from_file_location('session_compare', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sc = _load_compare()
CheckFailed = sc.CheckFailed
GROUPS, PREFIX, STATES, NETWORKS, PATIENCES, TARGET_COLS = (sc.GROUPS, sc.PREFIX, sc.STATES, sc.NETWORKS,
                                                            sc.PATIENCES, sc.TARGET_COLS)
OUTPUTS = ('session_summary.csv', 'session_table.md', 'session_differences.pdf')
STATE_LABEL = {'ordinary': 'stored', 're-estimated': 're-estimated'}
STATE_COLOR = {'ordinary': '#204f71', 're-estimated': '#b34929'}  # colours of det_curves.pdf in pull request #36
PANELS = tuple(itertools.product(PATIENCES, NETWORKS))  # order of Table A3: p4 LivenessNet, p4 ResNet-18, p12 ...
POSITIONS = tuple(itertools.product(TARGET_COLS, STATES))
OFFSETS = {1: -.18, 2: -.06, 3: .06, 4: .18}  # fixed horizontal offset per fold, no random jitter
METRIC_FIELDS = {'apcer': ('photograph_errors', 'n_photograph'), 'bpcer': ('bona_fide_errors', 'n_bona_fide')}
INT_FIELDS = ('patience', 'fold', 'seed') + tuple(PREFIX[g]+'_'+name for g in GROUPS for name in sc.COUNT_FIELDS)
RATE_FIELDS = tuple(PREFIX[g]+'_'+m for g in GROUPS for m in ('apcer', 'bpcer')) + ('pooled_apcer', 'pooled_bpcer')
SUMMARY_FIELDS = ('defined_units', 'sessions_1_2_mean', 'session_3_mean', 'mean_difference', 'min_difference',
                  'max_difference', 'units_up', 'units_zero', 'units_down')
SUMMARY_COLUMNS = (('patience', 'network', 'target', 'bn_state', 'highlighted', 'units')
                   + tuple(m+'_'+name for m in ('apcer', 'bpcer') for name in SUMMARY_FIELDS))
TABLE_HEAD = ['Patience', 'Detector', 'Target (%)', 'BN state'] + [
    f'{metric} {name}' for metric in ('APCER', 'BPCER')
    for name in ('(%) sessions 1–2', '(%) session 3', '(%) difference', 'units up/zero/down')]


def combinations():
    """The 16 combinations in the order of Table A3 of the draft: patience, detector, target, then BN state."""
    return list(itertools.product(PATIENCES, NETWORKS, TARGET_COLS, STATES))


def parse_counts(path):
    try:
        rows = sc.read(path)
    except (OSError, UnicodeDecodeError):
        raise CheckFailed('counts', 'session_counts.csv cannot be read') from None
    if not rows or tuple(rows[0]) != sc.COUNTS_COLUMNS:
        raise CheckFailed('counts', 'session_counts.csv does not have the expected columns')
    parsed = []
    try:
        for r in rows:
            row = dict(r)
            for name in INT_FIELDS:
                row[name] = int(r[name])
            for name in RATE_FIELDS:
                row[name] = float(r[name]) if r[name] != '' else None
            parsed.append(row)
    except ValueError:
        raise CheckFailed('counts', 'session_counts.csv has an unreadable value') from None
    return parsed


def observed_from_rows(rows):
    """Denominators per fold x class x group; every row of a fold must carry the same ones."""
    observed = {}
    for row in rows:
        for (name, _), n_field in zip(sc.CLASSES, ('n_photograph', 'n_bona_fide')):
            for g in GROUPS:
                cell, value = (row['fold'], name, g), row[PREFIX[g]+'_'+n_field]
                if observed.setdefault(cell, (value, None)) != (value, None):
                    raise CheckFailed('C6', 'denominators differ across rows of a fold', fold=row['fold'])
    return observed


def summarise_metric(units, metric):
    """Paired difference session 3 - sessions 1-2 per fold-seed unit, mean with weight 1/12 in exact arithmetic,
    minimum, maximum and signs decided on integers. If any of the twelve units is undefined (a denominator 0), the
    combination reports the number of defined units only (PROTOCOL.md, undefined quantities)."""
    e_field, n_field = METRIC_FIELDS[metric]
    e12s = [u['sessions_1_2_'+e_field] for u in units]
    n12s = [u['sessions_1_2_'+n_field] for u in units]
    e3s = [u['session_3_'+e_field] for u in units]
    n3s = [u['session_3_'+n_field] for u in units]
    defined = sum(1 for n12, n3 in zip(n12s, n3s) if n12 > 0 and n3 > 0)
    out = {metric+'_'+name: None for name in SUMMARY_FIELDS}
    out[metric+'_defined_units'] = defined
    out['_'+metric] = None
    if len(units) != 12 or defined != 12:
        return out
    r12 = [Fraction(e, n) for e, n in zip(e12s, n12s)]
    r3 = [Fraction(e, n) for e, n in zip(e3s, n3s)]
    diffs = [a3 - a12 for a12, a3 in zip(r12, r3)]
    signs = [e3 * n12 - e12 * n3 for e12, n12, e3, n3 in zip(e12s, n12s, e3s, n3s)]
    mean12 = sum(r12, Fraction(0)) / len(r12)
    mean3 = sum(r3, Fraction(0)) / len(r3)
    diff_mean = sum(diffs, Fraction(0)) / len(diffs)
    up = sum(1 for c in signs if c > 0)
    zero = sum(1 for c in signs if c == 0)
    down = sum(1 for c in signs if c < 0)
    exact = {'sessions_1_2_mean': mean12, 'session_3_mean': mean3, 'mean_difference': diff_mean,
             'min_difference': min(diffs), 'max_difference': max(diffs)}
    for name, value in exact.items():
        out[metric+'_'+name] = float(value)
    out.update({metric+'_units_up': up, metric+'_units_zero': zero, metric+'_units_down': down})
    out['_'+metric] = {'exact': exact, 'points': [(u['fold'], u['seed'], d) for u, d in zip(units, diffs)]}
    return out


def summarise(rows, cfg):
    results = []
    for p, net, col, state in combinations():
        units = sorted((r for r in rows if (r['patience'], r['network'], r['target'], r['bn_state']) ==
                        (p, net, col, state)), key=lambda r: (r['fold'], r['seed']))
        result = {'patience': p, 'network': net, 'target': col, 'bn_state': state,
                  'highlighted': 'yes' if (p, col) == (cfg['highlight']['patience'], cfg['highlight']['target'])
                  else 'no', 'units': len(units)}
        for metric in ('apcer', 'bpcer'):
            result.update(summarise_metric(units, metric))
        results.append(result)
    return results


def pp(value, signed=False):
    """Percentage points with two decimals, rounded once from the exact rational value, half to even."""
    cents = int(round(value * 100, 2) * 100)
    text = f'{abs(cents) // 100}.{abs(cents) % 100:02d}'
    return ('-' if cents < 0 else '+' if signed and cents > 0 else '') + text


def table(results, expected):
    lines = ['| ' + ' | '.join(TABLE_HEAD) + ' |', '|' + '---|' * len(TABLE_HEAD)]
    for r in results:
        cells = [str(r['patience']), r['network'], r['target'][:-3], STATE_LABEL[r['bn_state']]]
        for metric in ('apcer', 'bpcer'):
            exact = r['_'+metric]
            if exact is None:
                cells += [f"not reported ({r[metric+'_defined_units']} of 12 units defined)", '', '', '']
                continue
            values = exact['exact']
            cells += [pp(values['sessions_1_2_mean']), pp(values['session_3_mean']),
                      pp(values['mean_difference'], signed=True),
                      f"{r[metric+'_units_up']}/{r[metric+'_units_zero']}/{r[metric+'_units_down']}"]
        if r['highlighted'] == 'yes':
            cells = [f'**{c}**' if c else c for c in cells]
        lines.append('| ' + ' | '.join(cells) + ' |')
    denominators = '; '.join(
        f'fold {f}: photograph {expected[f, "photograph", GROUPS[0]][0]:,} and {expected[f, "photograph", GROUPS[1]][0]:,}, '
        f'bona fide {expected[f, "bona fide", GROUPS[0]][0]:,} and {expected[f, "bona fide", GROUPS[1]][0]:,}'
        for f in sc.FOLDS)
    caption = (
        'Test APCER and BPCER (%) of sessions 1–2 and of session 3 at the development thresholds of pull request '
        '#36 (development APCER target 5% or 10%), and the paired difference session 3 − sessions 1–2. Each cell '
        'is the mean of twelve fold–seed units with equal weight; the thresholds are transferred from the '
        'development scores of each checkpoint and BN state, as in Table A3, and are the same for the two '
        'groups. Units up/zero/down: fold–seed units with a positive, zero or negative difference, decided on '
        'integer counts. Every value is rounded separately to two decimals from its exact rational value (half to '
        'even), so the difference shown can differ by 0.01 from the difference of the two values shown; full '
        'values are in session_summary.csv. Rows in bold: highlighted outcome (patience 4, target 5%; author\'s '
        'choice).')
    note = ('Test frames per fold, sessions 1–2 and session 3: ' + denominators +
            '. The two groups contain different sets of manifest identifiers (PROTOCOL.md, limits).')
    return '\n'.join([caption, ''] + lines + ['', note, ''])


def figure_data(results):
    """Everything the figure shows, panel by panel: 2 rows (APCER, BPCER difference) x 4 columns (PANELS)."""
    by = {(r['patience'], r['network'], r['target'], r['bn_state']): r for r in results}
    panels = []
    for i, metric in enumerate(('apcer', 'bpcer')):
        for j, (p, net) in enumerate(PANELS):
            panel = {'row': i, 'column': j, 'metric': metric, 'patience': p, 'network': net,
                     'title': f'{net}, patience {p}',
                     'ylabel': f'{metric.upper()} difference (pp)\nsession 3 − sessions 1–2',
                     'xticklabels': [f'{col[:-3]}%\n{STATE_LABEL[state]}' for col, state in POSITIONS],
                     'points': [], 'means': []}
            for x, (col, state) in enumerate(POSITIONS):
                exact = by[p, net, col, state]['_'+metric]
                if exact is None:
                    continue
                color = STATE_COLOR[state]
                for f, seed, d in exact['points']:
                    panel['points'].append((x + OFFSETS[f], float(100 * d), color))
                panel['means'].append((x, float(100 * exact['exact']['mean_difference']), color))
            panels.append(panel)
    return panels


def draw(panels):
    """Draw the panels of figure_data and return the PDF bytes (no date metadata)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(PANELS), figsize=(11, 5.4), sharey='row')
    for panel in panels:
        ax = axes[panel['row'], panel['column']]
        ax.axhline(0, color='0.35', lw=.8)
        for x, y, color in panel['points']:
            ax.plot(x, y, 'o', ms=3.2, color=color, alpha=.75, mec='none')
        for x, y, color in panel['means']:
            ax.plot([x - .3, x + .3], [y, y], color=color, lw=2.2)
        ax.set_xticks(range(len(panel['xticklabels'])), panel['xticklabels'], fontsize=7)
        ax.set_xlim(-.6, len(panel['xticklabels']) - .4)
        ax.grid(axis='y', alpha=.22)
        if panel['row'] == 0:
            ax.set_title(panel['title'], fontsize=9)
        if panel['column'] == 0:
            ax.set_ylabel(panel['ylabel'], fontsize=8)
    fig.text(.5, .005, 'Development APCER target and BN state; dots: twelve fold–seed units; dash: equal-weight mean',
             ha='center', fontsize=7.5)
    fig.tight_layout(rect=(0, .03, 1, 1))
    buffer = io.BytesIO()
    fig.savefig(buffer, format='pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    return buffer.getvalue()


def check_outputs(out):
    if out.exists() and not out.is_dir() or any((out / name).exists() for name in OUTPUTS):
        raise CheckFailed('output', 'an output file already exists')


def run(args, cfg):
    # Public inputs first, as in session_compare.py.
    check_outputs(args.out)
    sc.check_c1(args.metrics, cfg)
    sc.check_c12_file(args.paired, cfg)
    sc.check_c6_file(args.config, cfg)
    expected = sc.load_expected(args.config, cfg)
    sm = {sc.metrics_key(r): r for r in sc.read(args.metrics)}
    rows = parse_counts(args.counts)
    sc.check_c10(rows=rows)
    sc.check_c6(expected, observed_from_rows(rows))
    sc.check_c8(rows, sm)
    sc.check_c9(rows, sm, sc.fold_totals(expected))
    sc.check_c12(args.paired, cfg, rows)
    results = summarise(rows, cfg)
    text = table(results, expected)
    pdf = draw(figure_data(results))
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out) as temp:
        temp = Path(temp)
        sc.write(temp/'session_summary.csv', [{k: r[k] for k in SUMMARY_COLUMNS} for r in results])
        (temp/'session_table.md').write_text(text, encoding='utf-8')
        (temp/'session_differences.pdf').write_bytes(pdf)
        for name in OUTPUTS:
            os.replace(temp/name, args.out/name)
    return len(results)


def main():
    warnings.simplefilter('ignore')  # a library warning would print source paths into the public log
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('counts', 'metrics', 'paired', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('session_config.json'),
                        help='used by check_synthetic.py only; the registered run reads the file next to this script')
    parser.add_argument('--traceback', type=Path, default=None,
                        help='restricted file that receives the full traceback of a failure')
    args = parser.parse_args()
    try:
        data = args.config.read_bytes()
        print('configuration sha256:', hashlib.sha256(data).hexdigest(), flush=True)
        n = run(args, sc.load_config(data))
    except CheckFailed as error:
        print(error.line() + '. No output written.', flush=True)
        sc.save_traceback(args.traceback)
        sys.exit(sc.EXIT_CHECK)
    except Exception as error:
        print(f'FAILED: unexpected error ({type(error).__name__}). No output written.', flush=True)
        sc.save_traceback(args.traceback)
        sys.exit(sc.EXIT_UNEXPECTED)
    print(f'{n} combinations summarised; session_summary.csv, session_table.md and session_differences.pdf '
          'written; checks C1, C6, C8, C9, C10 and C12 passed. Aggregate values only.')


if __name__ == '__main__':
    main()
