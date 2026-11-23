"""Session comparison at the registered development thresholds of pull request #36. Export aggregate counts only.

Reads the restricted test and development score files, the published score_metrics.csv and paired_summary.csv of
pull request #36 and the registered session_config.json; runs checks C1-C12 of PROTOCOL.md; writes only
session_counts.csv. The public inputs are checked before any restricted file is opened. Any failed check stops the
script before anything is written.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import tempfile
import traceback
import warnings
from pathlib import Path

import numpy as np

NETWORKS = ('LivenessNet','ResNet-18')
STATES = ('ordinary','re-estimated')
SEEDS = (42,43,44)
PATIENCES = (4, 12)
FOLDS = (1, 2, 3, 4)
TARGETS = (.05, .10)
TARGET_COLS = tuple(str(int(100*target))+'pct' for target in TARGETS)
GROUPS = ('sessions 1-2', 'session 3')
PREFIX = {'sessions 1-2': 'sessions_1_2', 'session 3': 'session_3'}
CLASSES = (('photograph', 1), ('bona fide', 0))
COUNT_FIELDS = ('n_photograph', 'n_bona_fide', 'photograph_errors', 'bona_fide_errors',
                'photograph_ties', 'bona_fide_ties')
KEY_COLUMNS = ('patience', 'fold', 'seed', 'network', 'bn_state', 'target')
COUNTS_COLUMNS = (KEY_COLUMNS + tuple('threshold_'+PREFIX[g] for g in GROUPS)
                  + tuple(PREFIX[g]+'_'+name for g in GROUPS for name in COUNT_FIELDS + ('apcer', 'bpcer'))
                  + ('pooled_apcer', 'pooled_bpcer', 'published_test_apcer', 'published_test_bpcer'))
EXPECTED_KEYS = frozenset(itertools.product(PATIENCES, FOLDS, SEEDS, NETWORKS, STATES, TARGET_COLS))
PS_METRICS = tuple(f'{col}_test_{metric}' for col in TARGET_COLS for metric in ('apcer', 'bpcer'))
SAMPLE_ID = re.compile(r'(real|attack)/([0-9]{4})_([0-9]{4})_([0-9]{2})_([0-9]{2})_([0-9]{2})_([0-9]+)\.jpg')
HEX64 = re.compile(r'[0-9a-f]{64}')
EXIT_CHECK, EXIT_UNEXPECTED = 4, 5  # argparse uses 2 for argument errors


class CheckFailed(Exception):
    """A failed check. The message names the check and a key of fixed values only, never an identifier."""

    def __init__(self, check, reason, **key):
        super().__init__(check)
        self.check, self.reason, self.key = check, reason, key

    def line(self):
        where = ', '.join(f'{name}={value}' for name, value in self.key.items())
        return f'CHECK FAILED {self.check}: {self.reason}' + (f' [{where}]' if where else '')


# --- Copied from score_analysis.py (commit fe87b3d), lines 19-28 and 31-42; code unchanged, one comment added. ---

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
    # Development BPCER is 1.0 - k/n, as in score_analysis.py: do not recompute as an integer ratio.
    b=1.0-np.searchsorted(genuine,thr,side='right')/len(genuine)
    return thr,a,b


# --- Copied from score_analysis.py (commit fe87b3d), lines 78-93, unchanged. ---

def prepare(rows,net,fold,seed,state,role):
    if len(rows)==0:
        raise ValueError('Missing score rows')
    order=sorted(rows,key=lambda r:r['sample_id'])
    if any((r['network'],r['fold'],r['train_seed'],r['bn_state'],r['role']) !=
           (net,str(fold),str(seed),state,role) for r in order):
        raise ValueError('Inconsistent checkpoint metadata')
    ids=tuple(r['sample_id'] for r in order)
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate sample within a checkpoint state')
    persons=tuple(r['subject_id'] for r in order)
    labels=np.array([int(r['label']) for r in order])
    scores=np.array([float(r['score_attack']) for r in order])
    if not np.array_equal(np.array([int(r['prediction_05']) for r in order]),scores>0.5):
        raise ValueError('Stored fixed-threshold decisions differ')
    return ids,persons,labels,scores


# --- Counting at one threshold. ---

def count_errors(scores, labels, t):
    """The one function used for both session groups. Decision rule of score_analysis.py (errors): attack if
    score > t, so a photograph is an error if score <= t and a bona fide frame is an error if score > t. Ties are
    frames whose score equals t (counts, not errors). The threshold written for the group is repr of the float used."""
    s, y = scores, labels
    return {'threshold': repr(t),
            'n_photograph': int(np.count_nonzero(y==1)),
            'n_bona_fide': int(np.count_nonzero(y==0)),
            'photograph_errors': int(np.count_nonzero(s[y==1]<=t)),
            'bona_fide_errors': int(np.count_nonzero(s[y==0]>t)),
            'photograph_ties': int(np.count_nonzero(s[y==1]==t)),
            'bona_fide_ties': int(np.count_nonzero(s[y==0]==t))}


def rate(errors, frames):
    """A rate with denominator 0 is undefined (None, written empty), never 0 or 1."""
    return errors / frames if frames else None


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(data):
    try:
        cfg = json.loads(data.decode('utf-8'))
        groups = cfg['groups']
        values = [v for g in GROUPS for v in groups[g]]
        highlight = cfg['highlight']
        valid = (tuple(groups) == GROUPS and all(groups[g] for g in GROUPS)
                 and all(isinstance(v, str) and re.fullmatch(r'[0-9]{2}', v) for v in values)
                 and len(values) == len(set(values))
                 and tuple(cfg['targets']) == TARGET_COLS
                 and highlight['patience'] in PATIENCES and highlight['target'] in TARGET_COLS
                 and tuple(highlight['networks']) == NETWORKS and tuple(highlight['bn_states']) == STATES
                 and tuple(highlight['metrics']) == ('apcer', 'bpcer')
                 and all(HEX64.fullmatch(cfg[k]) for k in ('metrics_sha256', 'ps_sha256'))
                 and HEX64.fullmatch(cfg['expected_counts']['sha256'])
                 and re.fullmatch(r'[A-Za-z0-9_.-]+', cfg['expected_counts']['file']))
    except (ValueError, KeyError, TypeError, AttributeError):
        valid = False
    if not valid:
        raise CheckFailed('configuration', 'session_config.json does not have the registered structure')
    return cfg


def metrics_key(r):
    return (int(r['patience']), int(r['fold']), int(r['seed']), r['network'], r['bn_state'])


def expected_path(cfg_path, cfg):
    return cfg_path.with_name(cfg['expected_counts']['file'])


def load_expected(cfg_path, cfg):
    rows = read(expected_path(cfg_path, cfg))
    return {(int(r['fold']), r['class'], r['group']): (int(r['frames']), int(r['identifiers'])) for r in rows}


def fold_totals(expected):
    return {(f, name): sum(expected[f, name, g][0] for g in GROUPS) for f in FOLDS for name, _ in CLASSES}


# --- Checks C1-C12 (PROTOCOL.md). Each one only raises CheckFailed. ---

def check_c1(metrics_path, cfg):
    """C1: score_metrics.csv is the published file: SHA-256, 96 rows, unique and expected keys."""
    try:
        digest = sha256_file(metrics_path)
        sm_rows = read(metrics_path)
    except (OSError, UnicodeDecodeError, csv.Error):
        raise CheckFailed('C1', 'score_metrics.csv cannot be read') from None
    if digest != cfg['metrics_sha256']:
        raise CheckFailed('C1', 'SHA-256 of score_metrics.csv differs from the configuration')
    needed = {'patience', 'fold', 'seed', 'network', 'bn_state'} | {
        col+'_'+name for col in TARGET_COLS
        for name in ('threshold', 'dev_apcer', 'dev_bpcer', 'test_apcer', 'test_bpcer')}
    if not sm_rows or not needed <= set(sm_rows[0]):
        raise CheckFailed('C1', 'score_metrics.csv lacks expected columns')
    try:
        keys = [metrics_key(r) for r in sm_rows]
    except ValueError:
        raise CheckFailed('C1', 'score_metrics.csv has an unreadable key') from None
    if len(keys) != 96 or len(set(keys)) != 96 or set(keys) != set(itertools.product(
            PATIENCES, FOLDS, SEEDS, NETWORKS, STATES)):
        raise CheckFailed('C1', 'score_metrics.csv does not have the 96 expected checkpoint-states')


def check_c2(file_rows, net, fold, seed, state, role, key, patience=None):
    """C2: row metadata as score_analysis.py prepare; no duplicate frame; stored decisions at 0.5; scores finite in
    [0, 1]; labels 0 or 1; on a test file, only the two expected BN states and the patience of the file name."""
    rows = [r for r in file_rows if r.get('bn_state') == state] if role == 'test' else file_rows
    if role == 'test' and not {r.get('bn_state') for r in file_rows} <= set(STATES):
        raise CheckFailed('C2', 'unexpected BN state in a test file', **key)
    if role == 'test' and any(r.get('patience') != str(patience) for r in rows):
        raise CheckFailed('C2', 'test patience differs from the file name', **key)
    try:
        prepare(rows, net, fold, seed, state, role)
        scores = np.array([float(r['score_attack']) for r in rows])
    except (ValueError, KeyError):
        raise CheckFailed('C2', f'{role} row metadata, duplicate frame or stored decision at 0.5', **key) from None
    if not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
        raise CheckFailed('C2', f'{role} score not finite or outside [0, 1]', **key)
    if any(r['label'] not in ('0', '1') for r in rows):
        raise CheckFailed('C2', f'{role} label other than 0 or 1', **key)


def check_c3(fold, test_pair, dev_pair, refs, key):
    """C3: within a fold, the ordered (sample_id, subject_id, label, session_id) tuple of the test and the ordered
    (sample_id, subject_id, label) tuple of the development partition are identical across detectors, seeds,
    patience settings and BN states; development and test share no frame and no identifier, and the development
    partition has 400 frames (as score_analysis.py line 211)."""
    if refs.setdefault(('test', fold), test_pair) != test_pair:
        raise CheckFailed('C3', 'test frames, identifiers, labels or sessions differ across conditions', **key)
    if refs.setdefault(('dev', fold), dev_pair) != dev_pair:
        raise CheckFailed('C3', 'development frames, identifiers or labels differ across conditions', **key)
    if set(test_pair[0]) & set(dev_pair[0]) or set(test_pair[1]) & set(dev_pair[1]) or len(dev_pair[2]) != 400:
        raise CheckFailed('C3', 'unexpected fit of the development and test partitions', **key)


def check_c4(test, dev, model, checkpoint_hashes, key):
    """C4: one checkpoint SHA-256 per model, the same in the two states and in the dev and test files (as
    score_analysis.py lines 203-210)."""
    hashes = {r['checkpoint_sha256'] for r in test + dev}
    if len(hashes) != 1:
        raise CheckFailed('C4', 'test and development checkpoint hashes differ', **key)
    checksum = hashes.pop()
    if not HEX64.fullmatch(checksum):
        raise CheckFailed('C4', 'invalid checkpoint SHA-256', **key)
    if checkpoint_hashes.setdefault(model, checksum) != checksum:
        raise CheckFailed('C4', 'the two BN states point to different parent checkpoints', **key)


def check_c5(ids, subjects, sessions, labels, allowed, key):
    """C5: session_id is one of the configured strings, exactly; it equals the session field of the original file
    name in sample_id; the class prefix agrees with label; subject_id equals both identifier fields."""
    for sample, subject, session, label in zip(ids, subjects, sessions, labels):
        match = SAMPLE_ID.fullmatch(sample)
        if match is None:
            raise CheckFailed('C5', 'sample_id not of the expected form', **key)
        if session not in allowed:
            raise CheckFailed('C5', 'session_id not among the configured values', **key)
        if session != match.group(6):
            raise CheckFailed('C5', 'session_id differs from the session field of the file name', **key)
        if (match.group(1) == 'attack') != (int(label) == 1):
            raise CheckFailed('C5', 'class prefix of sample_id differs from label', **key)
        if not subject == match.group(2) == match.group(3):
            raise CheckFailed('C5', 'subject_id differs from the identifier fields of sample_id', **key)


def check_c6_file(cfg_path, cfg):
    """C6, before any restricted file is read: expected_counts.csv exists, has the SHA-256 of the configuration and
    16 readable cells."""
    path = expected_path(cfg_path, cfg)
    try:
        digest = sha256_file(path)
    except OSError:
        raise CheckFailed('C6', 'expected_counts.csv cannot be read') from None
    if digest != cfg['expected_counts']['sha256']:
        raise CheckFailed('C6', 'SHA-256 of expected_counts.csv differs from the configuration')
    try:
        expected = load_expected(cfg_path, cfg)
    except (ValueError, KeyError, UnicodeDecodeError, csv.Error):
        raise CheckFailed('C6', 'expected_counts.csv has an unreadable value') from None
    if set(expected) != set(itertools.product(FOLDS, (name for name, _ in CLASSES), GROUPS)):
        raise CheckFailed('C6', 'cells of expected_counts.csv differ from the 16 expected')


def check_c6(expected, observed):
    """C6: frames and manifest identifiers per fold x class x group equal expected_counts.csv (16 cells, exact);
    every denominator > 0. observed maps (fold, class, group) to (frames, identifiers or None)."""
    cells = set(itertools.product(FOLDS, (name for name, _ in CLASSES), GROUPS))
    if set(expected) != cells or set(observed) != cells:
        raise CheckFailed('C6', 'cells of the counts differ from the 16 expected')
    for cell in sorted(cells):
        frames, identifiers = observed[cell]
        if frames <= 0 or frames != expected[cell][0] or (identifiers is not None and identifiers != expected[cell][1]):
            raise CheckFailed('C6', 'frame or identifier count differs from expected_counts.csv',
                              fold=cell[0], cls=cell[1], group=cell[2])


def check_c7(dy, ds, sm_row, key):
    """C7: thresholds re-selected from the development scores with profile and the selection of score_analysis.py
    (lines 31-42 and 224-233, commit fe87b3d) equal score_metrics.csv, as do development APCER and BPCER."""
    try:
        cuts,da,db=profile(dy,ds)
    except ValueError:
        raise CheckFailed('C7', 'development scores cannot be profiled', **key) from None
    for target in TARGETS:
        idx=np.where(da<=target)[0][-1]
        t=float(cuts[idx])
        col=str(int(100*target))
        for name,val in [('threshold',t),('dev_apcer',da[idx]),('dev_bpcer',db[idx])]:
            if float(val) != float(sm_row[col+'pct_'+name]):
                raise CheckFailed('C7', f'recomputed development {name} differs from score_metrics.csv',
                                  target=col+'pct', **key)


def check_c8(rows, sm):
    """C8: the threshold written for each group (repr of the float passed to count_errors) is the same for the two
    groups and equal to the string of score_metrics.csv."""
    for row in rows:
        published = sm[row['patience'], row['fold'], row['seed'], row['network'], row['bn_state']][
            row['target']+'_threshold']
        if not row['threshold_sessions_1_2'] == row['threshold_session_3'] == published:
            raise CheckFailed('C8', 'thresholds of the two groups differ from each other or from score_metrics.csv',
                              **{k: row[k] for k in KEY_COLUMNS})


def check_c9(rows, sm, totals):
    """C9: pooling the two groups gives the fold totals and the test APCER and BPCER of score_metrics.csv exactly;
    the rates written agree with the integer counts."""
    for row in rows:
        key = {k: row[k] for k in KEY_COLUMNS}
        sm_row = sm[row['patience'], row['fold'], row['seed'], row['network'], row['bn_state']]
        for (name, _), metric in zip(CLASSES, ('apcer', 'bpcer')):
            n_field, e_field = ('n_photograph', 'photograph_errors') if metric == 'apcer' else (
                'n_bona_fide', 'bona_fide_errors')
            for g in GROUPS:
                if row[PREFIX[g]+'_'+metric] != rate(row[PREFIX[g]+'_'+e_field], row[PREFIX[g]+'_'+n_field]):
                    raise CheckFailed('C9', f'group {metric} differs from its counts', **key)
            n = sum(row[PREFIX[g]+'_'+n_field] for g in GROUPS)
            e = sum(row[PREFIX[g]+'_'+e_field] for g in GROUPS)
            v = float(sm_row[row['target']+'_test_'+metric])
            if n != totals[row['fold'], name]:
                raise CheckFailed('C9', f'pooled {name} denominator differs from the fold total', **key)
            if not (e == round(v*n) and abs(v*n - e) < 1e-6 and e / n == v):
                raise CheckFailed('C9', f'pooled {metric} differs from score_metrics.csv', **key)
            if row['pooled_'+metric] != e / n or float(row['published_test_'+metric]) != v:
                raise CheckFailed('C9', f'pooled or published {metric} column differs', **key)


def check_c10(test_dir=None, dev_dirs=None, rows=None):
    """C10: completeness. Inputs: the 48 test and 96 development files exist. Rows: the 16 combinations x 12
    fold-seed units, each once, with both groups."""
    if test_dir is not None:
        for p, devroot in zip(PATIENCES, dev_dirs):
            for f, seed, net in itertools.product(FOLDS, SEEDS, NETWORKS):
                name = f'{net}_fold{f}_seed{seed}'
                files = [test_dir/f'p{p}_{name}_test_scores.csv'] + [
                    devroot/f'{name}_dev_{state}.csv' for state in STATES]
                if not all(path.is_file() for path in files):
                    raise CheckFailed('C10', 'input score file missing', patience=p, fold=f, seed=seed, network=net)
    if rows is not None:
        keys = [tuple(row[k] for k in KEY_COLUMNS) for row in rows]
        if len(keys) != len(EXPECTED_KEYS) or len(set(keys)) != len(keys) or set(keys) != EXPECTED_KEYS:
            raise CheckFailed('C10', 'operating points missing or repeated')
        for row in rows:
            if not all(isinstance(row['threshold_'+PREFIX[g]], str) and row['threshold_'+PREFIX[g]] for g in GROUPS):
                raise CheckFailed('C10', 'a group is missing', **{k: row[k] for k in KEY_COLUMNS})


def check_c11(out):
    """C11: the output directory is new or empty."""
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise CheckFailed('C11', 'choose a new or empty output directory')


def check_c12_file(ps_path, cfg):
    """C12, before any restricted file is read: paired_summary.csv exists, has the SHA-256 ps_sha256 of the
    configuration and one readable row for each of the 16 (patience, detector, metric) needed."""
    try:
        digest = sha256_file(ps_path)
        ps_rows = read(ps_path)
    except (OSError, UnicodeDecodeError, csv.Error):
        raise CheckFailed('C12', 'paired_summary.csv cannot be read') from None
    if digest != cfg['ps_sha256']:
        raise CheckFailed('C12', 'SHA-256 of paired_summary.csv differs from the configuration')
    try:
        for p, net, metric in itertools.product(PATIENCES, NETWORKS, PS_METRICS):
            entries = [r for r in ps_rows if (r['patience'], r['network'], r['metric']) == (str(p), net, metric)]
            if len(entries) != 1:
                raise CheckFailed('C12', 'paired_summary.csv lacks a row or repeats it', patience=p, network=net)
            float(entries[0]['stored_mean']), float(entries[0]['reestimated_mean'])
    except (KeyError, ValueError):
        raise CheckFailed('C12', 'paired_summary.csv has an unreadable value') from None


def check_c12(ps_path, cfg, rows):
    """C12: for the 16 combinations the 12-unit mean of the pooled rates, computed as in score_analysis.py
    (np.array in (fold, seed) order, .mean()), equals stored_mean or reestimated_mean of paired_summary.csv (32
    values)."""
    ps = {(r['patience'], r['network'], r['metric']): r for r in read(ps_path)}
    for p, net, col, metric in itertools.product(PATIENCES, NETWORKS, TARGET_COLS, ('apcer', 'bpcer')):
        entry = ps[str(p), net, f'{col}_test_{metric}']
        for state, field in zip(STATES, ('stored_mean', 'reestimated_mean')):
            units = sorted((r for r in rows if (r['patience'], r['network'], r['bn_state'], r['target']) ==
                            (p, net, state, col)), key=lambda r: (r['fold'], r['seed']))
            mean = float(np.array([r['pooled_'+metric] for r in units]).mean())
            if len(units) != 12 or mean != float(entry[field]):
                raise CheckFailed('C12', f'twelve-unit mean of the pooled {metric} differs from paired_summary.csv',
                                  patience=p, network=net, bn_state=state, target=col)


# --- Computation. ---

def extract(rows):
    order = sorted(rows, key=lambda r: r['sample_id'])
    return (tuple(r['sample_id'] for r in order), tuple(r['subject_id'] for r in order),
            tuple(r['session_id'] for r in order), np.array([int(r['label']) for r in order]),
            np.array([float(r['score_attack']) for r in order]))


def build_row(key6, counted, sm_row):
    p, f, seed, net, state, col = key6
    row = {'patience': p, 'fold': f, 'seed': seed, 'network': net, 'bn_state': state, 'target': col}
    for g in GROUPS:
        row['threshold_'+PREFIX[g]] = counted[g]['threshold']
    for g in GROUPS:
        c = counted[g]
        for name in COUNT_FIELDS:
            row[PREFIX[g]+'_'+name] = c[name]
        row[PREFIX[g]+'_apcer'] = rate(c['photograph_errors'], c['n_photograph'])
        row[PREFIX[g]+'_bpcer'] = rate(c['bona_fide_errors'], c['n_bona_fide'])
    for metric, e_field, n_field in (('apcer', 'photograph_errors', 'n_photograph'),
                                     ('bpcer', 'bona_fide_errors', 'n_bona_fide')):
        row['pooled_'+metric] = rate(sum(counted[g][e_field] for g in GROUPS),
                                     sum(counted[g][n_field] for g in GROUPS))
    row['published_test_apcer'] = sm_row[col+'_test_apcer']
    row['published_test_bpcer'] = sm_row[col+'_test_bpcer']
    assert tuple(row) == COUNTS_COLUMNS
    return row


def compare(args, cfg):
    # Public inputs and the output folder first: a failure here stops the script before any score file is opened.
    check_c11(args.out)
    check_c1(args.metrics, cfg)
    check_c12_file(args.paired, cfg)
    check_c6_file(args.config, cfg)
    expected = load_expected(args.config, cfg)
    sm = {metrics_key(r): r for r in read(args.metrics)}
    check_c10(test_dir=args.test, dev_dirs=(args.dev4, args.dev12))
    allowed = {v for values in cfg['groups'].values() for v in values}
    group_of = {v: g for g, values in cfg['groups'].items() for v in values}
    refs, fold_frames, checkpoint_hashes, counted = {}, {}, {}, {}
    for p, devroot in zip(PATIENCES, (args.dev4, args.dev12)):
        for f in FOLDS:
            for seed in SEEDS:
                for net in NETWORKS:
                    name = f'{net}_fold{f}_seed{seed}'
                    test_rows = read(args.test/f'p{p}_{name}_test_scores.csv')
                    cut = {(state, col): float(sm[p, f, seed, net, state][col+'_threshold'])
                           for state in STATES for col in TARGET_COLS}
                    for si, state in enumerate(STATES):
                        key = dict(patience=p, fold=f, seed=seed, network=net, bn_state=state)
                        dev_rows = read(devroot/f'{name}_dev_{state}.csv')
                        check_c2(test_rows, net, f, seed, state, 'test', key, patience=p)
                        check_c2(dev_rows, net, f, seed, state, 'dev', key)
                        test = [r for r in test_rows if r['bn_state'] == state]
                        ids, sub, ses, y, s = extract(test)
                        did, dsub, _, dy, ds = extract(dev_rows)
                        check_c4(test, dev_rows, (p, f, seed, net), checkpoint_hashes, key)
                        check_c5(ids, sub, ses, y, allowed, key)
                        check_c3(f, (ids, sub, tuple(int(v) for v in y), ses),
                                 (did, dsub, tuple(int(v) for v in dy)), refs, key)
                        check_c7(dy, ds, sm[p, f, seed, net, state], key)
                        fold_frames.setdefault(f, (sub, ses, y))
                        mask = {g: np.array([group_of.get(v) == g for v in ses], dtype=bool) for g in GROUPS}
                        for col in TARGET_COLS:
                            thresholds = {g: cut[state, col] for g in GROUPS}
                            counted[p, f, seed, net, state, col] = {
                                g: count_errors(s[mask[g]], y[mask[g]], thresholds[g]) for g in GROUPS}
    rows = [build_row(k, counted[k], sm[k[:5]]) for k in counted]
    check_c10(rows=rows)
    observed = {}
    for f, (sub, ses, y) in fold_frames.items():
        for name, label in CLASSES:
            for g in GROUPS:
                cell = [person for person, session, value in zip(sub, ses, y)
                        if value == label and group_of.get(session) == g]
                observed[f, name, g] = (len(cell), len(set(cell)))
    check_c6(expected, observed)
    check_c9(rows, sm, fold_totals(expected))
    check_c8(rows, sm)
    check_c12(args.paired, cfg, rows)
    check_c11(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out) as temp:
        write(Path(temp)/'session_counts.csv', rows)
        os.replace(Path(temp)/'session_counts.csv', args.out/'session_counts.csv')
    return len(rows)


def save_traceback(path):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(traceback.format_exc(), encoding='utf-8')


def main():
    warnings.simplefilter('ignore')  # a library warning would print source paths into the public log
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('test', 'dev4', 'dev12', 'metrics', 'paired', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('session_config.json'),
                        help='used by check_synthetic.py only; the registered run reads the file next to this script')
    parser.add_argument('--traceback', type=Path, default=None,
                        help='restricted file that receives the full traceback of a failure')
    args = parser.parse_args()
    try:
        data = args.config.read_bytes()
        print('configuration sha256:', hashlib.sha256(data).hexdigest(), flush=True)
        n = compare(args, load_config(data))
    except CheckFailed as error:
        print(error.line() + '. No output written.', flush=True)
        save_traceback(args.traceback)
        sys.exit(EXIT_CHECK)
    except Exception as error:
        print(f'FAILED: unexpected error ({type(error).__name__}). No output written.', flush=True)
        save_traceback(args.traceback)
        sys.exit(EXIT_UNEXPECTED)
    print(f'{n} operating points written to session_counts.csv; checks C1-C12 passed. Aggregate counts only.')


if __name__=='__main__':
    main()
