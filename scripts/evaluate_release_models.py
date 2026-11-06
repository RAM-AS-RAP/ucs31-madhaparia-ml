"""Matched V1/V3 synthetic-source evaluation and research-only selection policy.

Criteria below are frozen engineering acceptance criteria, NOT clinical safety
standards. No duration labels, flu subtypes or COVID cases are invented.
"""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

ROOT = Path(__file__).resolve().parents[1]


def backend_root(ml_root):
    for workspace in (ml_root.parent, ml_root.parents[1]):
        for name in ('ucs31-madhaparia-backend', 'ucs31-madhaparia-backend-main'):
            repository = workspace / name
            if not repository.is_dir():
                continue
            backend = repository if (repository / 'app').is_dir() else repository / name
            if not (backend / 'app/ml/v3_predictor.py').is_file():
                raise FileNotFoundError(f'The sibling backend at {repository} lacks the merged V3 predictor.')
            return backend
    raise FileNotFoundError('Place the merged ucs31-madhaparia-backend repository beside this ML project.')


BACKEND = backend_root(ROOT)
sys.path[:0] = [str(ROOT), str(BACKEND)]
from scripts.prepare_data import encode_evidences
from scripts.restore_research_sources import fetch
from app.ml.v3_predictor import load_v3_bundle, predict_v3

OUTPUT = ROOT / 'reports/release_model_evaluation'
SEED = 20261104
VIEWS = ['up_to_1', 'up_to_3', 'up_to_3_plus_3_negative', 'full_supported']
POLICY_VIEWS = VIEWS[1:]
WARNING_FEATURES = {'E_45', 'E_66', 'E_194', 'E_190', 'E_14'}
MODEL_HASH = 'ae709ef5ddf6dd7a101fe426b610c3e83ef58aa93cb895ca13e7500220dc889c'


def restore_split(name):
    specifications = {
        'validate': ('40278022', 'e4db88e9082f18410de6fd0cb7e74d18', 18706053),
        'test': ('40278016', '6fd53d331cf19ed4439bf4f46b92ee94', 18986243),
    }
    file_id, md5, size = specifications[name]
    directory = ROOT / 'data/external/ddxplus-official'
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f'release_{name}_patients.zip'
    fetch(archive, f'https://ndownloader.figshare.com/files/{file_id}', md5, size)
    target = ROOT / f'data/raw/release_{name}_patients'
    with zipfile.ZipFile(archive) as source:
        entries = [i for i in source.infolist() if not i.is_dir()]
        if len(entries) != 1 or Path(entries[0].filename).name != target.name:
            raise ValueError('Unexpected official split archive layout')
        if not target.exists():
            part = target.with_suffix('.part')
            with source.open(entries[0]) as src, part.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            if part.stat().st_size != entries[0].file_size:
                raise ValueError('Incomplete source split')
            part.replace(target)
        if target.stat().st_size != entries[0].file_size:
            raise ValueError('Existing split differs; not overwritten')
    return target


def profile(row, metadata, features):
    evidence = ast.literal_eval(row['EVIDENCES'])
    values = np.asarray([float(row['AGE']), float(row['SEX'] == 'M')] + encode_evidences(evidence, metadata, features), dtype='<f4')
    return evidence, values, hashlib.blake2b(values.tobytes(), digest_size=16).digest()


def read_split(path, classes, metadata, features, excluded):
    rows, profiles, counts = [], set(), {'scope_rows': 0, 'excluded_exact_profile_overlap': 0, 'excluded_no_supported_positive': 0}
    supported_binary = set(load_v3_bundle()['features']) - {'E_55_@_V_89', 'E_55_@_V_101'}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for number, row in enumerate(csv.DictReader(stream)):
            if row['PATHOLOGY'] not in classes:
                continue
            counts['scope_rows'] += 1
            evidence, values, signature = profile(row, metadata, features)
            profiles.add(signature)
            if signature in excluded:
                counts['excluded_exact_profile_overlap'] += 1
                continue
            if not set(evidence) & supported_binary:
                counts['excluded_no_supported_positive'] += 1
                continue
            rows.append((number, row, set(evidence), values, signature))
    return rows, profiles, counts


def v3_batch(present, absent, ages, sexes, bundle):
    names = bundle['features']
    groups = np.array([sum(age >= edge for edge in bundle['age_edges']) for age in ages])
    if bundle['demographic_mode'] == 'none':
        groups[:] = 0
    elif bundle['demographic_mode'] == 'age_sex':
        groups = groups * 2 + np.asarray(sexes)
    result = np.empty((len(present), len(bundle['classes'])))
    pain = names.index('E_53')
    locations = [i for i, f in enumerate(names) if f.startswith('E_55_@_')]
    for group in np.unique(groups):
        ix = np.flatnonzero(groups == group)
        likelihood = np.asarray(bundle['group_likelihoods'][group])
        logs = np.log(bundle['group_priors'][group]) + present[ix] @ np.log(likelihood).T + absent[ix] @ np.log1p(-likelihood).T
        with_pain = np.flatnonzero(present[ix, pain])
        if len(with_pain):
            conditional = np.asarray(bundle['pain_location_given_pain'][group])
            logs[with_pain] += present[ix[with_pain]][:, locations] @ (np.log(conditional[:, locations]) - np.log(likelihood[:, locations])).T
            logs[with_pain] += absent[ix[with_pain]][:, locations] @ (np.log1p(-conditional[:, locations]) - np.log1p(-likelihood[:, locations])).T
        logs /= bundle['temperature']
        exp = np.exp(logs - logs.max(axis=1, keepdims=True))
        result[ix] = exp / exp.sum(axis=1, keepdims=True)
    assert np.isfinite(result).all() and np.allclose(result.sum(axis=1), 1)
    return result


def evaluate(rows, model, features, bundle, split):
    labels = bundle['classes']
    names = bundle['features']
    native = [f for f in names if '_@_' not in f]
    positions = {f: i for i, f in enumerate(names)}
    observations = {v: {'present': [], 'absent': []} for v in VIEWS}
    actual, signatures, ages, sexes = [], [], [], []
    for number, row, evidence, _, signature in rows:
        positives = sorted(evidence & set(native))
        rng = random.Random(f'{SEED}:{split}:{number}')
        chosen = rng.sample(positives, min(3, len(positives)))
        negatives = rng.sample(sorted(set(native) - evidence), min(3, len(set(native) - evidence)))
        full = sorted(evidence & set(names))
        for view, pos, neg in [
            ('up_to_1', chosen[:1], []), ('up_to_3', chosen, []),
            ('up_to_3_plus_3_negative', chosen, negatives),
            ('full_supported', full, sorted(set(names) - evidence)),
        ]:
            observations[view]['present'].append([float(f in pos) for f in names])
            observations[view]['absent'].append([float(f in neg) for f in names])
        actual.append(labels.index(row['PATHOLOGY']))
        signatures.append(signature.hex())
        ages.append(float(row['AGE']))
        sexes.append(row['SEX'] == 'M')
    actual = np.asarray(actual)
    _, groups = np.unique(signatures, return_inverse=True)
    predictions, records, metrics = {}, {}, {}
    for view in VIEWS:
        p = np.asarray(observations[view]['present'])
        a = np.asarray(observations[view]['absent'])
        v1 = np.zeros((len(rows), len(features) + 2), dtype='float32')
        v1[:, 0], v1[:, 1] = ages, sexes
        for feature, index in features.items():
            if feature in positions:
                v1[:, index + 2] = p[:, positions[feature]]
        v1_scores = model.predict_proba(pd.DataFrame(v1, columns=['AGE', 'SEX_M'] + list(features)))
        v3_scores = v3_batch(p, a, ages, sexes, bundle)
        # Actual serving predictor parity on a fixed spread of rows in every view.
        for i in range(0, len(rows), max(1, len(rows) // 20)):
            serving = predict_v3([f for j, f in enumerate(names) if p[i, j]], [f for j, f in enumerate(names) if a[i, j]], ages[i], 'M' if sexes[i] else 'F', top_k=len(labels))
            lookup = {r['condition']: r['raw_relative_score'] for r in serving['predictions']}
            assert np.allclose(v3_scores[i], [lookup[c] for c in labels], atol=1e-12)
        predictions[view] = v3_scores
        rank = np.argsort(-v3_scores, axis=1, kind='stable')
        records[view] = {'actual': actual, 'groups': groups, 'score': v3_scores[np.arange(len(rows)), rank[:, 0]],
            'margin': v3_scores[np.arange(len(rows)), rank[:, 0]] - v3_scores[np.arange(len(rows)), rank[:, 1]],
            'correct': rank[:, 0] == actual, 'positive_count': p[:, [positions[f] for f in native]].sum(axis=1),
            'warning': np.any(p[:, [positions[f] for f in WARNING_FEATURES]], axis=1)}
        metrics[view] = {}
        for version, scores in [('V1', v1_scores), ('V3', v3_scores)]:
            ranking = np.argsort(-scores, axis=1, kind='stable')
            top = ranking[:, 0]
            report = classification_report(actual, top, labels=np.arange(len(labels)), target_names=labels, output_dict=True, zero_division=0)
            metrics[view][version] = {'rows': len(actual), 'top1_accuracy': float(accuracy_score(actual, top)),
                'top3_recall': float(np.mean(np.any(ranking[:, :3] == actual[:, None], axis=1))),
                'macro_f1': report['macro avg']['f1-score'], 'log_loss': float(log_loss(actual, scores, labels=np.arange(len(labels)))),
                'multiclass_brier': float(np.mean(np.sum((scores - np.eye(len(labels))[actual]) ** 2, axis=1))),
                'per_class': report, 'confusion_matrix': confusion_matrix(actual, top, labels=np.arange(len(labels))).tolist()}
    return metrics, records


def wilson_lower(success, total):
    if not total:
        return 0.0
    z, p = 1.96, success / total
    return (p + z*z/(2*total) - z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1 + z*z/total)


def selection(record, score, margin):
    accepted = (record['score'] >= score) & (record['margin'] >= margin) & (record['positive_count'] >= 2) & ~record['warning']
    groups = record['groups']
    present = np.bincount(groups[accepted], minlength=int(groups.max()) + 1) > 0
    failed = np.bincount(groups[accepted & ~record['correct']], minlength=len(present)) > 0
    total, success = int(present.sum()), int((present & ~failed).sum())
    return {'accepted_rows': int(accepted.sum()), 'row_coverage': float(accepted.mean()),
            'accepted_row_accuracy': float(record['correct'][accepted].mean()) if accepted.any() else None,
            'accepted_profile_groups': total, 'all_accepted_group_cases_correct': success,
            'group_accuracy_wilson_lower_95': wilson_lower(success, total)}


def main():
    if (OUTPUT / 'evaluation.json').exists():
        raise SystemExit('Evaluation already exists; preserve it rather than retuning on the test set')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bundle = load_v3_bundle()
    assert hashlib.sha256((BACKEND / 'app/ml/artifacts/v3_partial_evidence.json').read_bytes()).hexdigest() == MODEL_HASH
    model = joblib.load(ROOT / 'models/respiratory_random_forest_v1.joblib')
    names = json.loads((ROOT / 'models/model_features.json').read_text())
    assert names[:2] == ['AGE', 'SEX_M'] and list(model.classes_) == bundle['classes']
    features = {f: i for i, f in enumerate(names[2:])}
    metadata = json.loads((ROOT / 'data/raw/release_evidences.json').read_bytes())
    csv.field_size_limit(2_147_483_647)
    train_groups = set()
    train_count = 0
    print('Collecting complete 198-feature training signatures; no source labels used in signatures', flush=True)
    with (ROOT / 'data/raw/release_train_patients').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['PATHOLOGY'] in bundle['classes']:
                _, _, signature = profile(row, metadata, features)
                train_groups.add(signature)
                train_count += 1
    validation, validation_groups, validation_counts = read_split(restore_split('validate'), bundle['classes'], metadata, features, train_groups)
    val_metrics, val_records = evaluate(validation, model, features, bundle, 'validation')
    candidates = []
    for score in np.arange(0.50, 1.0, 0.05):
        for margin in [0.0, 0.1, 0.2, 0.3]:
            results = {view: selection(val_records[view], float(score), margin) for view in POLICY_VIEWS}
            if all(r['accepted_profile_groups'] >= 100 and r['group_accuracy_wilson_lower_95'] >= 0.90 for r in results.values()):
                candidates.append({'min_score': round(float(score), 2), 'min_margin': margin, 'validation': results,
                    'coverage': float(np.mean([r['row_coverage'] for r in results.values()]))})
    chosen = max(candidates, key=lambda c: (c['coverage'], c['min_score'], c['min_margin'])) if candidates else None
    # Freeze policy selection before reading/scoring the official test partition.
    (OUTPUT / 'frozen-validation-selection.json').write_text(json.dumps(chosen, indent=2), encoding='utf-8')
    print(f'Validation complete; qualifying policies: {len(candidates)}; selection frozen', flush=True)
    test, _, test_counts = read_split(restore_split('test'), bundle['classes'], metadata, features, train_groups | validation_groups)
    test_metrics, test_records = evaluate(test, model, features, bundle, 'test')
    accepted_test = {view: selection(test_records[view], chosen['min_score'], chosen['min_margin']) for view in POLICY_VIEWS} if chosen else {}
    approved = bool(chosen) and all(r['accepted_profile_groups'] >= 100 and r['group_accuracy_wilson_lower_95'] >= 0.90 for r in accepted_test.values())
    policy = {'policy_id': 'ddxplus_selective_research_v1', 'dataset': 'DDXPlus-synthetic', 'model_sha256': MODEL_HASH,
        'mode': 'selective_research' if approved else 'withhold_all', 'clinical_validation': False,
        'min_score': chosen['min_score'] if chosen else None, 'min_margin': chosen['min_margin'] if chosen else None,
        'minimum_positive_features': 2, 'exclude_reported_warning_features': True,
        'criterion': 'At least 100 accepted complete-profile groups and Wilson 95% lower bound >=0.90 in each declared view, validation then test. Engineering research criterion, not clinical approval.',
        'validation': chosen['validation'] if chosen else {}, 'test': accepted_test}
    report = {'model_sha256': MODEL_HASH, 'training_signature_rows': train_count, 'unique_train_signatures': len(train_groups),
        'validation_source': validation_counts, 'test_source': test_counts, 'classes': bundle['classes'],
        'validation': val_metrics, 'test': test_metrics, 'policy': policy, 'script_sha256': script_hash,
        'limitations': ['Synthetic source labels, not clinical diagnostic performance or local prevalence.',
            'Same cohort and feature disclosures across models; V1 cannot learn explicit absence and encodes unmentioned as zero.',
            'Complete 198-feature overlap exclusion is label-independent; near duplicates/generator dependence remain.',
            'Recorded model parameters were already selected upstream; this is a reproduction, not a newly untouched external clinical test.',
            'No duration/taste/COVID features fabricated; none are used in this 21-class benchmark.']}
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == script_hash
    (OUTPUT / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (OUTPUT / 'research_selection_policy.json').write_text(json.dumps(policy, indent=2), encoding='utf-8')
    print(json.dumps({'policy_mode': policy['mode'], 'selected': chosen and {k: chosen[k] for k in ['min_score', 'min_margin', 'coverage']},
        'test': {view: {v: {k: row[k] for k in ['top1_accuracy', 'top3_recall', 'log_loss']} for v, row in result.items()} for view, result in test_metrics.items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
