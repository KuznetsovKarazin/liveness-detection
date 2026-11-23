"""Exercise session_compare.py and session_summary.py end to end on invented scores only.

No file of the study is read. In a temporary directory the script invents 48 test files, 96 development files, a
score_metrics.csv and a paired_summary.csv (made with functions copied from score_analysis.py) and an
expected_counts.csv; the configuration is the registered session_config.json with the SHA-256 values of the invented
files. It runs both scripts, checks every count, every cell of the table and every value of the figure against an
independent computation, one operating point and one summary built by hand, the rule for undefined quantities, and
then stop cases for every check, each built so that this check fails and no other.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

NETWORKS = ('LivenessNet','ResNet-18')
STATES = ('ordinary','re-estimated')
SEEDS = (42,43,44)
PATIENCES = (4, 12)
FOLDS = (1, 2, 3, 4)
TARGETS = (.05, .10)
COLS = ('5pct', '10pct')
GROUPS = ('sessions 1-2', 'session 3')
PREFIX = {'sessions 1-2': 'sessions_1_2', 'session 3': 'session_3'}
CLASSES = ('photograph', 'bona fide')
EXIT_CHECK = 4

# Invented test frames per fold and (class, group): the twelve units have four different sets of denominators.
SIZES = {1: (6, 8, 5, 7), 2: (5, 9, 6, 4), 3: (7, 6, 4, 8), 4: (4, 10, 7, 5)}
CELLS = (('photograph', 'sessions 1-2'), ('photograph', 'session 3'),
         ('bona fide', 'sessions 1-2'), ('bona fide', 'session 3'))
# Invented identifiers, cycled within a cell: X is in both classes and both groups, Y is a photograph identifier in
# both groups, Z is bona fide in sessions 1-2 only and photograph in session 3 only, W is bona fide in session 3 only.
ID_PLAN = dict(zip(CELLS, ('XY', 'XYZ', 'XZ', 'XW')))
# Hand-built combination: patience 4, LivenessNet, stored statistics, 5% target. Errors per (fold, seed):
# photograph sessions 1-2, photograph session 3, bona fide sessions 1-2, bona fide session 3.
HAND_CONDITION = (4, 'LivenessNet', 'ordinary')
HAND = {(1, 42): (3, 4, 1, 3), (1, 43): (1, 6, 4, 1), (1, 44): (5, 2, 0, 0),
        (2, 42): (2, 7, 3, 2), (2, 43): (4, 1, 1, 4), (2, 44): (0, 0, 6, 0),
        (3, 42): (7, 6, 2, 1), (3, 43): (3, 5, 0, 8), (3, 44): (6, 2, 4, 4),
        (4, 42): (2, 5, 0, 5), (4, 43): (1, 9, 7, 0), (4, 44): (3, 9, 2, 1)}
# Signs decided by hand from HAND and SIZES (e3 * n12 against e12 * n3): units up, zero, down (not symmetric).
HAND_SIGNS = {'apcer': (5, 4, 3), 'bpcer': (4, 2, 6)}
# The hand-built operating point: patience 4, LivenessNet, fold 1, seed 42, stored, 5% target, sessions 1-2 then
# session 3: n_photograph, n_bona_fide, photograph_errors, bona_fide_errors, photograph_ties, bona_fide_ties.
HAND_POINT = {'sessions 1-2': (6, 5, 3, 1, 2, 2), 'session 3': (8, 7, 4, 3, 2, 2)}
COUNT_FIELDS = ('n_photograph', 'n_bona_fide', 'photograph_errors', 'bona_fide_errors',
                'photograph_ties', 'bona_fide_ties')
POSITIONS = ('L', 'E5', 'M', 'E10', 'H')
# What the table and the figure must show (PROTOCOL.md section 6), written out independently of the scripts.
TABLE_HEAD = ('| Patience | Detector | Target (%) | BN state | APCER (%) sessions 1–2 | APCER (%) session 3 | '
              'APCER (%) difference | APCER units up/zero/down | BPCER (%) sessions 1–2 | BPCER (%) session 3 | '
              'BPCER (%) difference | BPCER units up/zero/down |')
STATE_NAME = {'ordinary': 'stored', 're-estimated': 're-estimated'}
FIGURE_PANELS = ((4, 'LivenessNet'), (4, 'ResNet-18'), (12, 'LivenessNet'), (12, 'ResNet-18'))
FIGURE_POSITIONS = (('5pct', 'ordinary'), ('5pct', 're-estimated'), ('10pct', 'ordinary'), ('10pct', 're-estimated'))
FIGURE_TICKS = ['5%\nstored', '5%\nre-estimated', '10%\nstored', '10%\nre-estimated']
FIGURE_OFFSET = {1: -.18, 2: -.06, 3: .06, 4: .18}
FIGURE_COLOR = {'ordinary': '#204f71', 're-estimated': '#b34929'}


# --- Copied from score_analysis.py (commit fe87b3d), independent of the scripts under test. ---

def read(path):
    with path.open(newline='',encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def profile(y, scores):
    """Ascending thresholds, strict attack score > threshold. Preserve tied scores."""
    if not np.isfinite(scores).all() or not np.isin(y,[0,1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Each score set must be finite and contain both classes')
    if np.min(scores)<0 or np.max(scores)>1:
        raise ValueError('Expected attack probabilities in [0,1]')
    attacks=np.sort(scores[y==1])
    genuine=np.sort(scores[y==0])
    thr=np.r_[np.nextafter(np.min(scores),-np.inf),np.unique(scores)]
    a=np.searchsorted(attacks,thr,side='right')/len(attacks)
    b=1.0-np.searchsorted(genuine,thr,side='right')/len(genuine)
    return thr,a,b


def errors(y,s,t):
    return np.asarray([np.mean(s[y==1]<=t),np.mean(s[y==0]>t)])


# --- Invented data. ---

def expect(condition, message):
    if not condition:
        raise SystemExit('Synthetic check FAILED: ' + message)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frames(f):
    frames, pic = [], 1000 * f
    for (cls, group), n in zip(CELLS, SIZES[f]):
        plan = ID_PLAN[cls, group]
        for i in range(n):
            subject = f'9{f}0{"XYZW".index(plan[i % len(plan)]) + 1}'
            session = '03' if group == 'session 3' else ('01' if i % 2 == 0 else '02')
            pic += 1
            prefix = 'attack' if cls == 'photograph' else 'real'
            frames.append({'sample_id': f'{prefix}/{subject}_{subject}_00_01_{session}_{pic}.jpg',
                           'subject_id': subject, 'session_id': session, 'label': int(cls == 'photograph'),
                           'cls': cls, 'group': group})
    return frames


def dev_frames(f):
    frames = []
    for label in (0, 1):
        for i in range(200):
            subject = f'8{f}0{i % 4 + 1}'
            session = '01' if i % 2 == 0 else '02'
            prefix = 'attack' if label else 'real'
            frames.append({'sample_id': f'{prefix}/{subject}_{subject}_00_01_{session}_{5000 + 200 * label + i}.jpg',
                           'subject_id': subject, 'session_id': session, 'label': label})
    return frames


def dev_scores(idx):
    """200 bona fide then 200 photograph scores. The 5% threshold is the highest low bona fide score and the 10%
    threshold the highest of the second block of photograph scores; k bona fide frames at or below either threshold
    is 190, 189 or 188, so that 1.0 - k/200 and (200 - k)/200 can differ. Both thresholds differ in every
    checkpoint-state (offset by idx)."""
    off = 0.0005 * idx
    n_low = 190 - idx % 3
    bona = np.r_[np.linspace(0.05, 0.40 + off, n_low), np.linspace(0.85, 0.95, 200 - n_low)]
    attack = np.r_[np.linspace(0.10, 0.39, 10), np.linspace(0.50 + off, 0.55 + off, 10),
                   np.linspace(0.60 + off, 0.99, 180)]
    return np.r_[bona, attack]


def hand_positions(frames, f, seed):
    """Photograph errors at the 5% threshold alternate E5 (tie) and L; bona fide non-errors alternate E5 and L."""
    errors_by_cell = dict(zip(CELLS, HAND[f, seed]))
    positions, seen = [], {}
    for frame in frames:
        cell = (frame['cls'], frame['group'])
        i = seen.get(cell, 0)
        seen[cell] = i + 1
        e = errors_by_cell[cell]
        if frame['cls'] == 'photograph':
            positions.append(('E5', 'L')[i % 2] if i < e else ('M', 'E10', 'H')[(i - e) % 3])
        else:
            positions.append(('M', 'E10', 'H')[i % 3] if i < e else ('E5', 'L')[(i - e) % 2])
    return positions


def refresh(base):
    """Write the configuration with the SHA-256 values of the invented files, as they are now."""
    cfg = json.loads((base/'config'/'session_config.json').read_text(encoding='utf-8'))
    cfg['metrics_sha256'] = sha256_file(base/'public'/'score_metrics.csv')
    cfg['ps_sha256'] = sha256_file(base/'public'/'paired_summary.csv')
    cfg['expected_counts']['sha256'] = sha256_file(base/'config'/cfg['expected_counts']['file'])
    (base/'config'/'session_config.json').write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')


def build(base, template):
    rng = np.random.default_rng(20261122)
    for name in ('test', 'dev4', 'dev12', 'public', 'config'):
        (base/name).mkdir(parents=True)
    truth, sm_rows, idx = {}, [], 0
    for p, devname in zip(PATIENCES, ('dev4', 'dev12')):
        for f in FOLDS:
            frames, dframes = test_frames(f), dev_frames(f)
            y = np.array([fr['label'] for fr in frames])
            dy = np.array([fr['label'] for fr in dframes])
            for seed in SEEDS:
                for net in NETWORKS:
                    name = f'{net}_fold{f}_seed{seed}'
                    checksum = hashlib.sha256(f'invented-{p}-{name}'.encode()).hexdigest()
                    test_rows = []
                    for state in STATES:
                        ds = dev_scores(idx)
                        idx += 1
                        cuts, da, db = profile(dy, ds)
                        cut = {}
                        for target in TARGETS:
                            cut[target] = float(cuts[np.where(da <= target)[0][-1]])
                        expect(cut[.05] < cut[.10] < .6, 'invented development thresholds')
                        values = {'L': cut[.05] / 2, 'E5': cut[.05], 'M': (cut[.05] + cut[.10]) / 2,
                                  'E10': cut[.10], 'H': (cut[.10] + 1) / 2}
                        if (p, net, state) == HAND_CONDITION:
                            positions = hand_positions(frames, f, seed)
                        else:
                            positions = [POSITIONS[k] for k in rng.integers(0, len(POSITIONS), len(frames))]
                        s = np.array([values[q] for q in positions])
                        # score_metrics.csv as score_analysis.py lines 225-234 (AUROC and EER are placeholders).
                        row = {'patience': p, 'fold': f, 'seed': seed, 'network': net, 'bn_state': state,
                               'test_auroc': 0.5, 'test_eer': 0.5}
                        for target in TARGETS:
                            i = np.where(da <= target)[0][-1]
                            t = float(cuts[i])
                            rates = errors(y, s, t)
                            col = str(int(100*target))
                            for key, val in [('threshold', t), ('dev_apcer', da[i]), ('dev_bpcer', db[i]),
                                             ('test_apcer', rates[0]), ('test_bpcer', rates[1])]:
                                row[col+'pct_'+key] = float(val)
                        sm_rows.append(row)
                        truth[p, f, seed, net, state] = (frames, s, {c: cut[t] for c, t in zip(COLS, TARGETS)})
                        common = {'network': net, 'fold': f, 'train_seed': seed, 'bn_state': state}
                        write(base/devname/f'{name}_dev_{state}.csv', [
                            {**common, 'role': 'dev', 'sample_id': fr['sample_id'], 'subject_id': fr['subject_id'],
                             'session_id': fr['session_id'], 'label': fr['label'], 'score_attack': float(v),
                             'prediction_05': int(v > .5), 'checkpoint_sha256': checksum}
                            for fr, v in zip(dframes, ds)])
                        test_rows += [
                            {**common, 'patience': p, 'role': 'test', 'sample_id': fr['sample_id'],
                             'subject_id': fr['subject_id'], 'session_id': fr['session_id'], 'label': fr['label'],
                             'score_attack': float(v), 'prediction_05': int(v > .5), 'checkpoint_sha256': checksum}
                            for fr, v in zip(frames, s)]
                    write(base/'test'/f'p{p}_{name}_test_scores.csv', test_rows)
    write(base/'public'/'score_metrics.csv', sm_rows)
    # paired_summary.csv as score_analysis.py lines 280-291.
    summary=[]
    for patience in (4,12):
        for network in NETWORKS:
            rows=sorted((r for r in sm_rows if r['patience']==patience and r['network']==network),
                        key=lambda r:(r['fold'],r['seed']))
            for metric in ['test_auroc','test_eer','5pct_test_apcer','5pct_test_bpcer','10pct_test_apcer','10pct_test_bpcer']:
                o=np.array([r[metric] for r in rows if r['bn_state']=='ordinary'])
                a=np.array([r[metric] for r in rows if r['bn_state']=='re-estimated'])
                summary.append({'patience':patience,'network':network,'metric':metric,'paired_units':len(o),
                                'stored_mean':float(o.mean()),'reestimated_mean':float(a.mean()),
                                'mean_change':float((a-o).mean())})
    write(base/'public'/'paired_summary.csv', summary)
    cfg = json.loads(template.read_text(encoding='utf-8'))
    expected_rows = []
    for f in FOLDS:
        frames = test_frames(f)
        for cls, group in CELLS:
            cell = [fr for fr in frames if (fr['cls'], fr['group']) == (cls, group)]
            expected_rows.append({'fold': f, 'class': cls, 'group': group, 'frames': len(cell),
                                  'identifiers': len({fr['subject_id'] for fr in cell})})
    write(base/'config'/cfg['expected_counts']['file'], expected_rows)
    (base/'config'/'session_config.json').write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    refresh(base)
    return truth


# --- Running the scripts. ---

def run(script, args):
    return subprocess.run([sys.executable, '-I', '-B', str(script)] + [str(a) for a in args],
                          capture_output=True, text=True, timeout=600)


def compare_args(base, out):
    return ['--test', base/'test', '--dev4', base/'dev4', '--dev12', base/'dev12',
            '--metrics', base/'public'/'score_metrics.csv', '--paired', base/'public'/'paired_summary.csv',
            '--config', base/'config'/'session_config.json', '--out', out]


def summary_args(base, counts, out):
    return ['--counts', counts, '--metrics', base/'public'/'score_metrics.csv',
            '--paired', base/'public'/'paired_summary.csv', '--config', base/'config'/'session_config.json',
            '--out', out]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Independent verification. ---

def own_counts(frames, s, t, group):
    """Independent recount for one group, with the decision rule of score_analysis.py."""
    mask = np.array([fr['group'] == group for fr in frames])
    y = np.array([fr['label'] for fr in frames])[mask]
    v = s[mask]
    return (int((y == 1).sum()), int((y == 0).sum()), int(((y == 1) & (v <= t)).sum()),
            int(((y == 0) & (v > t)).sum()), int(((y == 1) & (v == t)).sum()), int(((y == 0) & (v == t)).sum()))


def check_counts(rows, truth, sm):
    expect(len(rows) == 192, 'session_counts.csv does not have 192 rows')
    for row in rows:
        k = (int(row['patience']), int(row['fold']), int(row['seed']), row['network'], row['bn_state'])
        frames, s, cut = truth[k]
        t = cut[row['target']]
        expect(row['threshold_sessions_1_2'] == row['threshold_session_3'] == repr(t) ==
               sm[k][row['target']+'_threshold'], 'thresholds written')
        pooled = np.zeros(4, dtype=int)
        for g in GROUPS:
            counts = own_counts(frames, s, t, g)
            got = tuple(int(row[PREFIX[g]+'_'+name]) for name in COUNT_FIELDS)
            expect(got == counts, f'counts of {g} differ from the independent recount')
            expect(float(row[PREFIX[g]+'_apcer']) == counts[2] / counts[0] and
                   float(row[PREFIX[g]+'_bpcer']) == counts[3] / counts[1], 'group rates')
            pooled += np.array(counts[:4])
        expect(float(row['pooled_apcer']) == pooled[2] / pooled[0] == float(sm[k][row['target']+'_test_apcer']),
               'pooled APCER')
        expect(float(row['pooled_bpcer']) == pooled[3] / pooled[1] == float(sm[k][row['target']+'_test_bpcer']),
               'pooled BPCER')
    point = [r for r in rows if (r['patience'], r['fold'], r['seed'], r['network'], r['bn_state'], r['target']) ==
             ('4', '1', '42', 'LivenessNet', 'ordinary', '5pct')]
    expect(len(point) == 1, 'hand-built operating point missing')
    for g in GROUPS:
        got = tuple(int(point[0][PREFIX[g]+'_'+name]) for name in COUNT_FIELDS)
        expect(got == HAND_POINT[g], f'hand-built operating point, {g}')
    frames, s, cut = truth[4, 1, 42, 'LivenessNet', 'ordinary']
    y = np.array([fr['label'] for fr in frames])
    expect(((y == 1) & (s < cut['5pct'])).sum() != ((y == 1) & (s <= cut['5pct'])).sum() and
           ((y == 0) & (s >= cut['5pct'])).sum() != ((y == 0) & (s > cut['5pct'])).sum(),
           'the hand-built operating point does not depend on ties')
    tied = sum(1 for r in rows if all(int(r[PREFIX[g]+'_'+c+'_ties']) > 0 for g in GROUPS
                                      for c in ('photograph', 'bona_fide')))
    expect(tied >= 2, 'ties in both classes and both groups at fewer than two operating points')
    differing = sum(1 for r in sm.values() for c in COLS if float(r[c+'_dev_bpcer']) !=
                    (200 - round((1 - float(r[c+'_dev_bpcer'])) * 200)) / 200)
    expect(differing > 0, 'no development BPCER where 1.0 - k/200 differs from (200 - k)/200')
    for f in FOLDS:
        frames = test_frames(f)
        for cls in CLASSES:
            ids = {g: {fr['subject_id'] for fr in frames if (fr['cls'], fr['group']) == (cls, g)} for g in GROUPS}
            expect(ids[GROUPS[0]] & ids[GROUPS[1]] and ids[GROUPS[0]] ^ ids[GROUPS[1]],
                   'identifiers in one group only and in both')
    # The order and the function of the twelve-unit mean of check C12 must matter on these data, so that computing
    # it in (seed, fold) order or with a plain sum would fail.
    by_order, by_sum = 0, 0
    for p in PATIENCES:
        for net in NETWORKS:
            for col in COLS:
                for state in STATES:
                    units = [r for r in rows if (r['patience'], r['network'], r['target'], r['bn_state']) ==
                             (str(p), net, col, state)]
                    for metric in ('apcer', 'bpcer'):
                        fs = [float(r['pooled_'+metric]) for r in sorted(units, key=lambda r: (int(r['fold']), int(r['seed'])))]
                        sf = [float(r['pooled_'+metric]) for r in sorted(units, key=lambda r: (int(r['seed']), int(r['fold'])))]
                        reference = float(np.array(fs).mean())
                        by_order += float(np.array(sf).mean()) != reference
                        by_sum += sum(fs) / 12 != reference
    expect(by_order > 0 and by_sum > 0, 'the order and the function of the C12 mean do not matter on these data')


def independent_exact(rows, p, net, col, state, metric):
    e, n = ('photograph_errors', 'n_photograph') if metric == 'apcer' else ('bona_fide_errors', 'n_bona_fide')
    units = sorted((r for r in rows if (r['patience'], r['network'], r['target'], r['bn_state']) ==
                    (str(p), net, col, state)), key=lambda r: (int(r['fold']), int(r['seed'])))
    e12 = [int(u['sessions_1_2_'+e]) for u in units]
    n12 = [int(u['sessions_1_2_'+n]) for u in units]
    e3 = [int(u['session_3_'+e]) for u in units]
    n3 = [int(u['session_3_'+n]) for u in units]
    d = [Fraction(a, b) - Fraction(c, q) for a, b, c, q in zip(e3, n3, e12, n12)]
    return {'units': len(units), 'folds_seeds': [(int(u['fold']), int(u['seed'])) for u in units], 'd': d,
            'sessions_1_2_mean': sum(Fraction(a, b) for a, b in zip(e12, n12)) / 12,
            'session_3_mean': sum(Fraction(a, b) for a, b in zip(e3, n3)) / 12,
            'mean_difference': sum(d) / 12, 'min_difference': min(d), 'max_difference': max(d),
            'signs': (sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d)),
            'frame_weighted': Fraction(sum(e3), sum(n3)) - Fraction(sum(e12), sum(n12))}


def pp_ref(value, signed=False):
    """Independent percentage-point formatting: exact value x 100, two decimals, half to even."""
    with localcontext() as ctx:
        ctx.prec = 80
        v = (Decimal(value.numerator) * 100 / Decimal(value.denominator)).quantize(Decimal('0.01'),
                                                                                    rounding=ROUND_HALF_EVEN)
    text = f'{abs(v):.2f}'
    return ('-' if v < 0 else '+' if signed and v > 0 else '') + text


def check_summary(out, rows):
    summary = read(out/'session_summary.csv')
    expect(len(summary) == 16, 'session_summary.csv does not have 16 rows')
    order = [(r['patience'], r['network'], r['target'], r['bn_state']) for r in summary]
    expect(order == [(str(p), n, c, s) for p in PATIENCES for n in NETWORKS for c in COLS for s in STATES],
           'order of Table A3')
    expect([r['highlighted'] for r in summary].count('yes') == 4 and all(
        (r['highlighted'] == 'yes') == ((r['patience'], r['target']) == ('4', '5pct')) for r in summary),
        'highlighted rows')
    for r in summary:
        for metric in ('apcer', 'bpcer'):
            ref = independent_exact(rows, int(r['patience']), r['network'], r['target'], r['bn_state'], metric)
            expect(r['units'] == '12' and r[metric+'_defined_units'] == '12', 'units')
            for name in ('sessions_1_2_mean', 'session_3_mean', 'mean_difference', 'min_difference',
                         'max_difference'):
                expect(float(r[metric+'_'+name]) == float(ref[name]),
                       f'{metric} {name} differs from the independent value')
            got = tuple(int(r[metric+'_units_'+s]) for s in ('up', 'zero', 'down'))
            expect(got == ref['signs'], f'{metric} signs differ from the independent value')
            if (r['patience'], r['network'], r['target'], r['bn_state']) == ('4', 'LivenessNet', '5pct', 'ordinary'):
                expect(got == HAND_SIGNS[metric], f'{metric} signs differ from the hand count')
                expect(ref['mean_difference'] != 0 and ref['frame_weighted'] != ref['mean_difference'],
                       'the hand-built combination cannot tell the direction or the weighting')
                expect(ref['min_difference'] < 0 < ref['max_difference'] and got[1] > 0,
                       'the hand-built combination lacks a negative, a positive or a zero difference')
    # Every cell of the table.
    text = (out/'session_table.md').read_text(encoding='utf-8')
    body = [line for line in text.splitlines() if line.startswith('| ')]
    expected_lines = [TABLE_HEAD]
    for p in PATIENCES:
        for net in NETWORKS:
            for col in COLS:
                for state in STATES:
                    cells = [str(p), net, col[:-3], STATE_NAME[state]]
                    for metric in ('apcer', 'bpcer'):
                        ref = independent_exact(rows, p, net, col, state, metric)
                        cells += [pp_ref(ref['sessions_1_2_mean']), pp_ref(ref['session_3_mean']),
                                  pp_ref(ref['mean_difference'], signed=True), '/'.join(map(str, ref['signs']))]
                    if (p, col) == (4, '5pct'):
                        cells = [f'**{c}**' for c in cells]
                    expected_lines.append('| ' + ' | '.join(cells) + ' |')
    expect(len(body) == 17, 'session_table.md does not have a header and 16 rows')
    for i, (got, want) in enumerate(zip(body, expected_lines)):
        expect(got == want, f'session_table.md line {i} differs from the independent table')
    note = 'Test frames per fold, sessions 1–2 and session 3: ' + '; '.join(
        f'fold {f}: photograph {SIZES[f][0]} and {SIZES[f][1]}, bona fide {SIZES[f][2]} and {SIZES[f][3]}'
        for f in FOLDS) + '. The two groups contain different sets of manifest identifiers (PROTOCOL.md, limits).'
    expect(note in text.splitlines(), 'denominator note of session_table.md')
    pdf = (out/'session_differences.pdf').read_bytes()
    expect(pdf.startswith(b'%PDF') and len(pdf) > 1000, 'session_differences.pdf')


def check_figure_and_format(package, base, counts, rows):
    """The values of the figure (figure_data), the rounding of pp, and the rule for undefined quantities in the
    summary, the table, the CSV and the figure."""
    module = load_module('session_summary_under_test', package/'session_summary.py')
    for value, signed, text in ((Fraction(29, 10000), False, '0.29'), (Fraction(1, 800), False, '0.12'),
                                (Fraction(3, 800), False, '0.38'), (Fraction(-1, 400), True, '-0.25'),
                                (Fraction(1, 400), True, '+0.25'), (Fraction(1, 400), False, '0.25'),
                                (Fraction(-1, 10**7), True, '0.00'), (Fraction(1234567, 1000000), False, '123.46')):
        expect(module.pp(value, signed=signed) == text == pp_ref(value, signed=signed), f'pp of {value}')
    cfg = json.loads((base/'config'/'session_config.json').read_text(encoding='utf-8'))
    parsed = module.parse_counts(counts)
    panels = module.figure_data(module.summarise(parsed, cfg))
    expect(len(panels) == 8, 'figure panels')
    i = 0
    for row, metric in enumerate(('apcer', 'bpcer')):
        for column, (p, net) in enumerate(FIGURE_PANELS):
            panel = panels[i]
            i += 1
            expect((panel['row'], panel['column'], panel['metric'], panel['patience'], panel['network']) ==
                   (row, column, metric, p, net), f'figure panel {i} position')
            expect(panel['title'] == f'{net}, patience {p}' and panel['xticklabels'] == FIGURE_TICKS and
                   panel['ylabel'] == f'{metric.upper()} difference (pp)\nsession 3 − sessions 1–2',
                   f'figure panel {i} labels')
            points, means = [], []
            for x, (col, state) in enumerate(FIGURE_POSITIONS):
                ref = independent_exact(rows, p, net, col, state, metric)
                points += [(x + FIGURE_OFFSET[f], float(100 * d), FIGURE_COLOR[state])
                           for (f, _), d in zip(ref['folds_seeds'], ref['d'])]
                means.append((x, float(100 * ref['mean_difference']), FIGURE_COLOR[state]))
            expect(panel['points'] == points, f'figure panel {i} points')
            expect(panel['means'] == means, f'figure panel {i} means')
    # Undefined quantities: one unit of the hand-built combination with no session 3 photograph frame.
    expect(module.sc.rate(0, 0) is None and module.sc.rate(1, 4) == .25, 'rate with denominator 0')
    target = (4, 1, 42, 'LivenessNet', 'ordinary', '5pct')
    changed = [dict(r, session_3_n_photograph=0, session_3_photograph_errors=0)
               if (r['patience'], r['fold'], r['seed'], r['network'], r['bn_state'], r['target']) == target else r
               for r in parsed]
    units = sorted((r for r in changed if (r['patience'], r['network'], r['target'], r['bn_state']) ==
                    (4, 'LivenessNet', '5pct', 'ordinary')), key=lambda r: (r['fold'], r['seed']))
    apcer = module.summarise_metric(units, 'apcer')
    bpcer = module.summarise_metric(units, 'bpcer')
    expect(apcer['apcer_defined_units'] == 11 and apcer['apcer_mean_difference'] is None and
           apcer['apcer_sessions_1_2_mean'] is None and apcer['_apcer'] is None, 'undefined APCER not withheld')
    expect(bpcer['bpcer_defined_units'] == 12 and bpcer['bpcer_mean_difference'] is not None, 'BPCER affected')
    results = module.summarise(changed, cfg)
    text = module.table(results, module.sc.load_expected(base/'config'/'session_config.json', cfg))
    expect('| **4** | **LivenessNet** | **5** | **stored** | **not reported (11 of 12 units defined)** |  |  |  |'
           in text, 'undefined cell of the table')
    with TemporaryDirectory() as temp:
        path = Path(temp)/'summary.csv'
        module.sc.write(path, [{k: r[k] for k in module.SUMMARY_COLUMNS} for r in results])
        row = read(path)[0]
        expect(row['apcer_defined_units'] == '11' and all(row['apcer_'+name] == '' for name in (
            'sessions_1_2_mean', 'session_3_mean', 'mean_difference', 'min_difference', 'max_difference',
            'units_up', 'units_zero', 'units_down')) and row['bpcer_mean_difference'] != '',
            'undefined values of session_summary.csv')
    panels = module.figure_data(results)
    expect([x for x, _, _ in panels[0]['means']] == [1, 2, 3] and len(panels[0]['points']) == 36,
           'undefined position of the figure')
    expect(module.draw(panels).startswith(b'%PDF'), 'figure with an undefined position')


# --- Stop cases. ---

def edit_csv(path, change):
    rows = read(path)
    change(rows)
    write(path, rows)


def edit_config(base, change):
    path = base/'config'/'session_config.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    change(cfg)
    path.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')


def flip(hexdigest):
    return hexdigest[:-1] + ('0' if hexdigest[-1] != '0' else '1')


def first(rows, **match):
    return next(r for r in rows if all(r[k] == v for k, v in match.items()))


def move_session(row, session):
    row['sample_id'] = re.sub(r'_[0-9]{2}_([0-9]+)\.jpg$', rf'_{session}_\1.jpg', row['sample_id'])
    row['session_id'] = session


def nudge(path, row_match, column, direction=2.0):
    def change(rows):
        row = first(rows, **row_match)
        row[column] = repr(float(np.nextafter(float(row[column]), direction)))
    edit_csv(path, change)


def edit_fold_tests(b, f, sample, change):
    """The same change to one frame in all 24 test files of a fold, both BN states (so that C3 cannot see it)."""
    for p in PATIENCES:
        for seed in SEEDS:
            for net in NETWORKS:
                edit_csv(b/'test'/f'p{p}_{net}_fold{f}_seed{seed}_test_scores.csv',
                         lambda rows: [change(r) for r in rows if r['sample_id'] == sample])


def edit_fold_devs(b, f, sample, change):
    """The same change to one frame in all 24 development files of a fold."""
    for devname in ('dev4', 'dev12'):
        for seed in SEEDS:
            for net in NETWORKS:
                for state in STATES:
                    edit_csv(b/devname/f'{net}_fold{f}_seed{seed}_dev_{state}.csv',
                             lambda rows: [change(r) for r in rows if r['sample_id'] == sample])


def compare_cases(sm):
    cases = []

    def case(check, reason, prepare):
        cases.append((check, reason, prepare))

    def refreshed(change):
        def prepare(b):
            change(b)
            refresh(b)
        return prepare

    metrics = lambda b: b/'public'/'score_metrics.csv'
    expected = lambda b: b/'config'/'expected_counts.csv'
    # C1
    case('C1', 'SHA-256 of score_metrics.csv', lambda b: edit_config(
        b, lambda c: c.update(metrics_sha256=flip(c['metrics_sha256']))))
    case('C1', 'lacks expected columns', refreshed(lambda b: edit_csv(
        metrics(b), lambda rows: [r.pop('10pct_test_bpcer') for r in rows])))
    case('C1', 'does not have the 96 expected', refreshed(lambda b: edit_csv(
        metrics(b), lambda rows: rows.__setitem__(1, dict(rows[0])))))
    # C2
    case('C2', 'test row metadata', lambda b: edit_csv(
        b/'test'/'p12_ResNet-18_fold2_seed43_test_scores.csv',
        lambda rows: first(rows, bn_state='re-estimated').update(train_seed='99')))
    case('C2', 'unexpected BN state', lambda b: edit_csv(
        b/'test'/'p4_ResNet-18_fold2_seed42_test_scores.csv', lambda rows: rows.append(dict(rows[0], bn_state='other'))))
    case('C2', 'stored decision at 0.5', lambda b: edit_csv(
        b/'test'/'p12_LivenessNet_fold4_seed44_test_scores.csv',
        lambda rows: first(rows, bn_state='re-estimated').update(
            prediction_05='1' if first(rows, bn_state='re-estimated')['prediction_05'] == '0' else '0')))
    case('C2', 'test score not finite or outside', lambda b: edit_csv(
        b/'test'/'p4_LivenessNet_fold1_seed42_test_scores.csv',
        lambda rows: next(r for r in rows if r['bn_state'] == 'ordinary' and float(r['score_attack']) > .7).update(
            score_attack='1.5', prediction_05='1')))
    case('C2', 'test patience differs', lambda b: edit_csv(
        b/'test'/'p12_ResNet-18_fold1_seed42_test_scores.csv', lambda rows: rows[0].update(patience='7')))
    # C3
    case('C3', 'test frames, identifiers, labels or sessions differ', lambda b: edit_csv(
        b/'test'/'p12_ResNet-18_fold3_seed44_test_scores.csv',
        lambda rows: move_session(first(rows, bn_state='re-estimated', session_id='02'), '03')))
    case('C3', 'development frames, identifiers or labels differ', lambda b: edit_csv(
        b/'dev12'/'LivenessNet_fold2_seed44_dev_re-estimated.csv', lambda rows: rows[0].update(subject_id='8299')))
    dev_sample = dev_frames(1)[0]['sample_id']
    case('C3', 'unexpected fit of the development and test', lambda b: edit_fold_devs(
        b, 1, dev_sample, lambda r: r.update(subject_id=test_frames(1)[0]['subject_id'])))

    def extra_dev_frame(b):
        """One more development row with a new sample_id, the same in all 24 development files of fold 1, so that only
        the count of 400 development frames can see it."""
        extra = dev_sample.replace('_5000.jpg', '_5999.jpg')
        for devname in ('dev4', 'dev12'):
            for seed in SEEDS:
                for net in NETWORKS:
                    for state in STATES:
                        edit_csv(b/devname/f'{net}_fold1_seed{seed}_dev_{state}.csv',
                                 lambda rows: rows.append(dict(first(rows, sample_id=dev_sample), sample_id=extra)))
    case('C3', 'unexpected fit of the development and test', extra_dev_frame)
    # C4
    def other_checkpoint(b):
        for path in (b/'test'/'p4_LivenessNet_fold2_seed43_test_scores.csv',
                     b/'dev4'/'LivenessNet_fold2_seed43_dev_re-estimated.csv'):
            edit_csv(path, lambda rows: [r.update(checkpoint_sha256='e' * 64) for r in rows
                                         if r['bn_state'] == 're-estimated'])
    case('C4', 'different parent checkpoints', other_checkpoint)
    case('C4', 'test and development checkpoint hashes differ', lambda b: edit_csv(
        b/'dev4'/'LivenessNet_fold2_seed43_dev_ordinary.csv',
        lambda rows: [r.update(checkpoint_sha256='d' * 64) for r in rows]))

    def not_hex(b):
        for path in [b/'test'/'p12_ResNet-18_fold3_seed42_test_scores.csv'] + [
                b/'dev12'/f'ResNet-18_fold3_seed42_dev_{state}.csv' for state in STATES]:
            edit_csv(path, lambda rows: [r.update(checkpoint_sha256='g' * 64) for r in rows])
    case('C4', 'invalid checkpoint SHA-256', not_hex)
    # C5: the first four leave groups and counts unchanged and are applied to the whole fold, so that only C5 can see
    # them (they run first: without C5 the first failure is one of them); the last three change the group of a frame.
    frame = test_frames(1)[0]  # photograph, sessions 1-2, session 01, identifier X of fold 1
    other = test_frames(1)[1]['subject_id']  # identifier Y, in the same cell
    case('C5', 'differs from the session field', lambda b: edit_fold_tests(
        b, 1, frame['sample_id'], lambda r: r.update(session_id='02')))
    case('C5', 'not of the expected form', lambda b: edit_fold_tests(
        b, 1, frame['sample_id'], lambda r: r.update(sample_id=r['sample_id'].replace('.jpg', '.png'))))
    case('C5', 'class prefix', lambda b: edit_fold_tests(
        b, 1, frame['sample_id'], lambda r: r.update(sample_id=r['sample_id'].replace('attack/', 'real/'))))
    case('C5', 'subject_id differs from the identifier fields', lambda b: edit_fold_tests(
        b, 1, frame['sample_id'], lambda r: r.update(subject_id=other)))
    name = 'p4_ResNet-18_fold1_seed43_test_scores.csv'
    case('C5', 'not among the configured values', lambda b: edit_csv(
        b/'test'/name, lambda rows: move_session(first(rows, bn_state='re-estimated', session_id='03'), '04')))
    case('C5', 'not among the configured values', lambda b: edit_csv(
        b/'test'/name, lambda rows: first(rows, bn_state='re-estimated', session_id='03').update(session_id='3')))
    case('C5', 'not among the configured values', lambda b: edit_csv(
        b/'test'/name, lambda rows: first(rows, bn_state='re-estimated', session_id='03').update(session_id=' 03')))
    # C6: the first two leave the fold totals unchanged, so that only C6 can see them (they run first).
    def expected_change(field, deltas):
        return lambda b: edit_csv(expected(b), lambda rows: [
            rows[i].update({field: str(int(rows[i][field]) + d)}) for i, d in deltas])
    case('C6', 'frame or identifier count differs', refreshed(expected_change('identifiers', [(0, 1)])))
    case('C6', 'frame or identifier count differs', refreshed(expected_change('frames', [(0, -1), (1, 1)])))
    case('C6', 'frame or identifier count differs', refreshed(expected_change('frames', [(0, 1)])))
    case('C6', 'SHA-256 of expected_counts.csv', expected_change('frames', [(0, 1)]))
    # C7
    first_key = {'patience': '4', 'fold': '1', 'seed': '42', 'network': 'LivenessNet', 'bn_state': 'ordinary'}
    case('C7', 'recomputed development threshold', refreshed(lambda b: nudge(metrics(b), first_key, '5pct_threshold')))
    case('C7', 'recomputed development dev_apcer', refreshed(lambda b: nudge(metrics(b), first_key, '5pct_dev_apcer')))
    where = next((k, c) for k, r in sm.items() for c in COLS if float(r[c+'_dev_bpcer']) !=
                 (200 - round((1 - float(r[c+'_dev_bpcer'])) * 200)) / 200)

    def bpcer_integer_form(b):
        def change(rows):
            row = first(rows, patience=str(where[0][0]), fold=str(where[0][1]), seed=str(where[0][2]),
                        network=where[0][3], bn_state=where[0][4])
            k = round((1 - float(row[where[1]+'_dev_bpcer'])) * 200)
            row[where[1]+'_dev_bpcer'] = repr((200 - k) / 200)
        edit_csv(metrics(b), change)
    case('C7', 'recomputed development dev_bpcer', refreshed(bpcer_integer_form))
    # C9
    def rate_one_frame(b):
        def change(rows):
            row = first(rows, patience='12', fold='2', seed='43', network='ResNet-18', bn_state='ordinary')
            n = sum(SIZES[2][:2])
            k = round(float(row['10pct_test_apcer']) * n)
            row['10pct_test_apcer'] = repr((k + 1 if k < n else k - 1) / n)
        edit_csv(metrics(b), change)
    case('C9', 'pooled apcer differs from score_metrics.csv', refreshed(rate_one_frame))
    # C10, C11
    case('C10', 'input score file missing', lambda b: (b/'dev12'/'ResNet-18_fold4_seed44_dev_ordinary.csv').unlink())
    case('C11', 'new or empty output directory', None)
    # C12
    ps_row = {'patience': '4', 'network': 'LivenessNet', 'metric': '5pct_test_apcer'}
    case('C12', 'twelve-unit mean of the pooled apcer', refreshed(
        lambda b: nudge(b/'public'/'paired_summary.csv', ps_row, 'stored_mean')))
    case('C12', 'SHA-256 of paired_summary.csv', lambda b: nudge(b/'public'/'paired_summary.csv', ps_row,
                                                                 'stored_mean'))
    return cases


def summary_cases(sm):
    cases = []

    def other_state_threshold(b, counts):
        def change(rows):
            row = rows[0]
            other = sm[int(row['patience']), int(row['fold']), int(row['seed']), row['network'],
                       STATES[1 - STATES.index(row['bn_state'])]]
            row['threshold_session_3'] = other[row['target']+'_threshold']
        edit_csv(counts, change)
    cases.append(('C8', 'thresholds of the two groups', other_state_threshold))
    cases.append(('C1', 'SHA-256 of score_metrics.csv', lambda b, counts: edit_config(
        b, lambda c: c.update(metrics_sha256=flip(c['metrics_sha256'])))))

    def expected_plus_one(b, counts):
        edit_csv(b/'config'/'expected_counts.csv', lambda rows: rows[0].update(frames=str(int(rows[0]['frames']) + 1)))
        refresh(b)
    cases.append(('C6', 'differs from expected_counts.csv', expected_plus_one))

    def one_more_error(b, counts):
        def change(rows):
            row = rows[3]
            e, n = int(row['session_3_photograph_errors']), int(row['session_3_n_photograph'])
            row['session_3_photograph_errors'] = str(e + 1 if e < n else e - 1)
        edit_csv(counts, change)
    cases.append(('C9', 'apcer', one_more_error))
    cases.append(('C10', 'operating points missing or repeated', lambda b, counts: edit_csv(counts, lambda rows: rows.pop())))

    def ps_mean(b, counts):
        nudge(b/'public'/'paired_summary.csv', {'patience': '12', 'network': 'ResNet-18',
                                                 'metric': '10pct_test_bpcer'}, 'reestimated_mean')
        refresh(b)
    cases.append(('C12', 'twelve-unit mean of the pooled bpcer', ps_mean))
    cases.append(('output', 'an output file already exists', None))
    return cases


def stopped(proc, check, reason, work):
    text = proc.stdout + proc.stderr
    lines = [line for line in text.splitlines() if line.startswith('CHECK FAILED')]
    return (proc.returncode == EXIT_CHECK and len(lines) == 1 and lines[0].startswith(f'CHECK FAILED {check}:')
            and reason in lines[0] and 'Traceback' not in text and str(work) not in text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent,
                        help='folder with session_compare.py, session_summary.py and session_config.json')
    package = parser.parse_args().package.resolve()
    compare, summarise = package/'session_compare.py', package/'session_summary.py'
    with TemporaryDirectory() as temp:
        work = Path(temp)
        base = work/'base'
        truth = build(base, package/'session_config.json')
        sm = {(int(r['patience']), int(r['fold']), int(r['seed']), r['network'], r['bn_state']): r
              for r in read(base/'public'/'score_metrics.csv')}
        out = work/'out'
        proc = run(compare, compare_args(base, out))
        expect(proc.returncode == 0, 'session_compare.py failed on the invented inputs: ' + proc.stdout[-300:])
        expect(sorted(p.name for p in out.iterdir()) == ['session_counts.csv'], 'files written by session_compare.py')
        counts = out/'session_counts.csv'
        rows = read(counts)
        check_counts(rows, truth, sm)
        proc = run(summarise, summary_args(base, counts, out))
        expect(proc.returncode == 0, 'session_summary.py failed on the invented inputs: ' + proc.stdout[-300:])
        expect(sorted(p.name for p in out.iterdir()) == ['session_counts.csv', 'session_differences.pdf',
                                                         'session_summary.csv', 'session_table.md'],
               'files written by session_summary.py')
        check_summary(out, rows)
        check_figure_and_format(package, base, counts, rows)
        n_cases = 0
        for i, (check, reason, prepare) in enumerate(compare_cases(sm)):
            case = work/f'compare-{i:02d}'
            shutil.copytree(base, case/'in')
            if prepare is not None:
                prepare(case/'in')
            else:
                (case/'out').mkdir()
                (case/'out'/'existing.txt').write_text('x', encoding='utf-8')
            proc = run(compare, compare_args(case/'in', case/'out'))
            expect(stopped(proc, check, reason, work) and not (case/'out'/'session_counts.csv').exists(),
                   f'stop case {i} {check} ({reason}) of session_compare.py: ' + proc.stdout[-300:])
            n_cases += 1
        for i, (check, reason, prepare) in enumerate(summary_cases(sm)):
            case = work/f'summary-{i:02d}'
            shutil.copytree(base, case/'in')
            shutil.copy(counts, case/'session_counts.csv')
            before = []
            if prepare is not None:
                prepare(case/'in', case/'session_counts.csv')
            else:
                (case/'out').mkdir()
                (case/'out'/'session_summary.csv').write_text('x', encoding='utf-8')
                before = ['session_summary.csv']
            proc = run(summarise, summary_args(case/'in', case/'session_counts.csv', case/'out'))
            after = sorted(p.name for p in (case/'out').iterdir()) if (case/'out').exists() else []
            expect(stopped(proc, check, reason, work) and after == before,
                   f'stop case {i} {check} ({reason}) of session_summary.py: ' + proc.stdout[-300:])
            n_cases += 1
    print(f'Synthetic end-to-end check passed: 192 operating points, 16 combinations, table and figure, '
          f'{n_cases} stop cases.')


if __name__ == '__main__':
    main()
