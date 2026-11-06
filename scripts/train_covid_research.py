"""Offline GCCR002 experiment, never a COVID/flu/TB clinical diagnosis service.

Uses reported laboratory-test groups and real recorded survey measurements.
No invented duration values, flu labels, or replacement of the working model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
from urllib.request import urlopen

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'd717674e3b86c3c7c314af05ea44497abc88a622'
BASE_URL = f'https://raw.githubusercontent.com/GCCR/GCCR002/{COMMIT}/'
SEED = 20261104
BASE_FEATURES = ['Age', 'Symptoms_fever', 'Symptoms_dry_cough', 'Symptoms_cough_with_mucus',
    'Symptoms_difficulty_breathing_/_shortness_of_breath', 'Symptoms_chest_tightness',
    'Symptoms_runny_nose', 'Symptoms_sore_throat', 'Symptoms_loss_of_appetite',
    'Symptoms_headache', 'Symptoms_muscle_aches', 'Symptoms_fatigue']
SENSORY = ['Symptoms_changes_in_food_flavor', 'Symptoms_changes_in_smell',
           'Taste_before_illness', 'Taste_during_illness', 'Smell_before_illness', 'Smell_during_illness']
DURATION = ['Days_since_onset']
ALL_FEATURES = BASE_FEATURES + SENSORY + DURATION
FEATURE_SETS = {'symptoms': BASE_FEATURES, 'symptoms_duration': BASE_FEATURES + DURATION,
                'symptoms_sensory': BASE_FEATURES + SENSORY, 'all': ALL_FEATURES}


def download(directory):
    files = ['LICENSE', 'README.md', 'data/processed/data-clean.csv', 'data/processed/targets.json', 'data/processed/data-types.json']
    hashes = {}
    for name in files:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urlopen(BASE_URL + name, timeout=60) as response:
                content = response.read(20_000_001)
            if len(content) > 20_000_000:
                raise ValueError('Unexpectedly large source file')
            path.write_bytes(content)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'repository': 'https://github.com/GCCR/GCCR002', 'commit': COMMIT, 'sha256': hashes}
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Existing source identity differs; source was not overwritten')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def cohort(directory):
    data = pd.read_csv(directory / 'data/processed/data-clean.csv', dtype={'UniqueID': str}).set_index('UniqueID')
    if not data.index.is_unique:
        raise ValueError('Repeated participant IDs must be resolved before splitting')
    targets = json.loads((directory / 'data/processed/targets.json').read_text(encoding='utf-8'))
    positive, negative = set(map(str, targets['lab-covid'])), set(map(str, targets['non-covid']))
    if positive & negative or not positive or not negative:
        raise ValueError('Invalid/overlapping reported laboratory outcome groups')
    ids = positive | negative
    if not ids <= set(data.index):
        raise ValueError('Outcome identifiers do not match source rows')
    frame = data.loc[sorted(ids), ALL_FEATURES].apply(pd.to_numeric, errors='raise')
    y = pd.Series([int(i in positive) for i in frame.index], index=frame.index)
    complete = frame.notna().all(axis=1) & np.isfinite(frame).all(axis=1)
    valid = complete & frame['Age'].between(0, 120) & frame['Days_since_onset'].between(0, 3650)
    for column in ALL_FEATURES:
        if column.startswith('Symptoms_'):
            valid &= frame[column].isin([0, 1])
        elif column.startswith(('Taste_', 'Smell_')):
            valid &= frame[column].between(0, 100)
    info = {'source_reported_lab_positive': len(positive), 'source_reported_lab_negative': len(negative),
            'complete_valid_cohort': int(valid.sum()), 'excluded_incomplete_or_invalid': int((~valid).sum()),
            'excluded_missing_values': int((~complete).sum())}
    if not valid.any():
        raise ValueError('No complete compatible source records; no values will be invented')
    return frame.loc[valid], y.loc[valid], info


def split_groups(frame, y):
    # Group identical complete predictor profiles without consulting disease labels.
    groups = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    indices = np.arange(len(frame))
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    develop, test = next(first.split(indices, y, groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    train_local, validation_local = next(second.split(develop, y.iloc[develop], groups[develop]))
    partitions = {'train': develop[train_local], 'validation': develop[validation_local], 'test': test}
    keys = list(partitions)
    for i, a in enumerate(keys):
        assert set(y.iloc[partitions[a]]) == {0, 1}
        for b in keys[i + 1:]:
            assert not set(groups[partitions[a]]) & set(groups[partitions[b]])
    assert sorted(np.concatenate(list(partitions.values()))) == list(range(len(frame)))
    return partitions, groups


def metrics(y, probabilities):
    p = np.asarray(probabilities)
    assert len(p) == len(y) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return {'rows': len(y), 'reported_positive': int(np.sum(y)),
            'roc_auc': float(roc_auc_score(y, p)), 'average_precision': float(average_precision_score(y, p)),
            'log_loss': float(log_loss(y, p)), 'brier': float(brier_score_loss(y, p)),
            'confusion_at_0_5_descriptive_only': confusion_matrix(y, p >= 0.5, labels=[0, 1]).tolist()}


def uncertainty(y, predictions, groups):
    """Paired profile-cluster bootstrap; intervals describe this source cohort only."""
    rng = np.random.default_rng(SEED + 2)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    samples = {name: {'roc_auc': [], 'log_loss': []} for name in predictions}
    for _ in range(300):
        ix = np.concatenate([members[g] for g in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(np.asarray(y)[ix])) < 2:
            continue
        for name, p in predictions.items():
            samples[name]['roc_auc'].append(float(roc_auc_score(np.asarray(y)[ix], p[ix])))
            samples[name]['log_loss'].append(float(log_loss(np.asarray(y)[ix], p[ix])))
    intervals = {name: {metric: np.quantile(values, [0.025, 0.975]).tolist() for metric, values in result.items()}
                 for name, result in samples.items()}
    paired = {}
    for name in ['symptoms_duration', 'symptoms_sensory', 'all']:
        paired[name + '_minus_symptoms'] = {metric: np.quantile(
            np.asarray(samples[name][metric]) - np.asarray(samples['symptoms'][metric]), [0.025, 0.975]).tolist()
            for metric in ['roc_auc', 'log_loss']}
    return {'method': '300 paired complete-predictor-profile cluster resamples; percentile 95% intervals',
            'source_cohort_only': True, 'intervals': intervals, 'ablation_difference_intervals': paired}


def portable_score(bundle, observation):
    if set(observation) != set(bundle['features']):
        raise ValueError('Complete structured study-compatible observations required; unknown values are not zero')
    values = [observation[f] for f in bundle['features']]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('Study-compatible finite numeric observations required')
    for feature, value in zip(bundle['features'], values):
        if feature.startswith('Symptoms_') and value not in (0, 1):
            raise ValueError('Binary symptom fields require explicit 0 or 1')
        if feature.startswith(('Taste_', 'Smell_')) and not 0 <= value <= 100:
            raise ValueError('Sensory ratings require the original 0–100 study scale')
        if feature == 'Age' and not 0 <= value <= 120:
            raise ValueError('Invalid age')
        if feature == 'Days_since_onset' and not 0 <= value <= 3650:
            raise ValueError('Invalid illness onset interval')
    z = bundle['intercept'] + sum(c * (v - m) / s for v, m, s, c in zip(values, bundle['mean'], bundle['scale'], bundle['coefficients']))
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def main():
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--source', type=Path, default=ROOT / 'data/external/gccr002')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/covid_research')
    args = parser.parse_args()
    if (args.output / 'evaluation.json').exists():
        parser.error('Results already exist; use a different output directory to preserve evidence')
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = download(args.source) if args.download else json.loads((args.source / 'manifest.json').read_text(encoding='utf-8'))
    for name, digest in provenance['sha256'].items():
        assert hashlib.sha256((args.source / name).read_bytes()).hexdigest() == digest
    frame, y, counts = cohort(args.source)
    partitions, groups = split_groups(frame, y)
    report = {'training_script_sha256': script_hash,
        'scope': 'Offline reported-lab COVID-positive vs COVID-negative among recent respiratory illness respondents; not COVID vs confirmed influenza/TB',
        'source': provenance, 'cohort': counts, 'seed': SEED,
        'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__},
        'partitions': {}, 'validation': {}, 'test': {}, 'clinical_validation': False,
        'activated_in_application': False,
        'limitations': ['2020 crowdsourced study; laboratory outcomes are respondent reports, not independently linked assays.',
            'Chemosensory selection/ascertainment bias, class imbalance and temporal/geographic transportability limits.',
            'COVID-negative does not mean influenza, URTI, healthy, or TB-negative. No 22-class differential or co-infection inference.',
            'Days_since_onset is overall time since respiratory illness onset, not per-symptom cough duration.',
            'Only complete structured numeric observations supported. Vague text, sparse inputs and missing ratings are not imputed.',
            'Source processed data follows the upstream study cleaning; not independently raw-clinical adjudicated data.',
            'No patient use, score threshold, clinical sign-off or diagnostic claim is approved by this experiment.']}
    for name, ix in partitions.items():
        report['partitions'][name] = {'rows': len(ix), 'reported_positive': int(y.iloc[ix].sum()),
            'unique_predictor_groups': len(set(groups[ix])),
            'participant_set_sha256': hashlib.sha256('\n'.join(sorted(frame.index[ix])).encode()).hexdigest()}
    models = {}
    train, validation, test = [partitions[k] for k in ['train', 'validation', 'test']]
    for name, features in FEATURE_SETS.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
        model.fit(frame.iloc[train][features], y.iloc[train])
        models[name] = model
        report['validation'][name] = metrics(y.iloc[validation], model.predict_proba(frame.iloc[validation][features])[:, 1])
    selected = min(report['validation'], key=lambda name: report['validation'][name]['log_loss'])
    report['selected_by_validation_log_loss'] = selected
    # Freeze the choice before accessing final test scores; test never selects parameters.
    predictions = {}
    for name, model in models.items():
        features = FEATURE_SETS[name]
        predictions[name] = model.predict_proba(frame.iloc[test][features])[:, 1]
        report['test'][name] = metrics(y.iloc[test], predictions[name])
    prevalence = float(y.iloc[train].mean())
    report['test']['constant_training_prevalence_baseline'] = metrics(y.iloc[test], np.full(len(test), prevalence))
    report['test_uncertainty'] = uncertainty(y.iloc[test], predictions, groups[test])
    scaler, predictor = models[selected].steps[0][1], models[selected].steps[1][1]
    artifact = {'model_version': 'covid_survey_research_v1', 'scope': report['scope'], 'features': FEATURE_SETS[selected],
        'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'coefficients': predictor.coef_[0].tolist(),
        'intercept': float(predictor.intercept_[0]), 'clinical_validation': False, 'complete_observations_required': True,
        'source_commit': COMMIT, 'selected_feature_set': selected, 'score_meaning': 'Exploratory within-study binary score, not personal disease risk'}
    expected = models[selected].predict_proba(frame.iloc[test][artifact['features']])[:, 1]
    actual = [portable_score(artifact, row.to_dict()) for _, row in frame.iloc[test][artifact['features']].iterrows()]
    assert np.allclose(actual, expected, atol=1e-12)
    report['portable_prediction_parity_rows'] = len(actual)
    artifact_path = args.output / 'covid_survey_research_v1.json'
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    report['artifact_sha256'] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == script_hash, 'Training code changed during execution'
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'cohort': counts, 'selected': selected, 'test': report['test'], 'patient_app_changed': False}, indent=2))


if __name__ == '__main__':
    main()
