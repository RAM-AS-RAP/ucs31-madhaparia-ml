import numpy as np
import pytest

from scripts.evaluate_release_models import backend_root, load_v3_bundle, predict_v3, selection, v3_batch, wilson_lower


@pytest.mark.parametrize('ml_name,backend_name,nested', [
    ('ucs31-madhaparia-ml', 'ucs31-madhaparia-backend', False),
    ('ucs31-madhaparia-ml-main', 'ucs31-madhaparia-backend-main', False),
    ('ucs31-madhaparia-ml-main/ucs31-madhaparia-ml-main', 'ucs31-madhaparia-backend-main', True),
])
def test_backend_lookup_supports_repository_and_archive_layouts(tmp_path, ml_name, backend_name, nested):
    ml = tmp_path / ml_name
    ml.mkdir(parents=True)
    backend = tmp_path / backend_name
    if nested:
        backend /= backend_name
    predictor = backend / 'app/ml/v3_predictor.py'
    predictor.parent.mkdir(parents=True)
    predictor.write_text('# test fixture, not imported', encoding='utf-8')
    assert backend_root(ml) == backend


def test_missing_backend_has_actionable_error(tmp_path):
    ml = tmp_path / 'ucs31-madhaparia-ml'
    ml.mkdir()
    with pytest.raises(FileNotFoundError, match='Place the merged'):
        backend_root(ml)


def test_incomplete_canonical_backend_does_not_silently_use_another_copy(tmp_path):
    ml = tmp_path / 'ucs31-madhaparia-ml'
    ml.mkdir()
    (tmp_path / 'ucs31-madhaparia-backend/app').mkdir(parents=True)
    other = tmp_path / 'ucs31-madhaparia-backend-main/app/ml/v3_predictor.py'
    other.parent.mkdir(parents=True)
    other.write_text('# wrong sibling must not be used', encoding='utf-8')
    with pytest.raises(FileNotFoundError, match='lacks the merged V3 predictor'):
        backend_root(ml)


def test_vectorized_evaluation_matches_actual_serving_predictor():
    bundle = load_v3_bundle()
    names = bundle['features']
    cases = [(['E_201', 'E_91'], []), (['E_53', 'E_55_@_V_89'], ['E_65', 'E_55_@_V_101'])]
    p = np.asarray([[float(f in pos) for f in names] for pos, _ in cases])
    a = np.asarray([[float(f in neg) for f in names] for _, neg in cases])
    scores = v3_batch(p, a, [22, 50], [False, True], bundle)
    for i, (pos, neg) in enumerate(cases):
        result = predict_v3(pos, neg, [22, 50][i], ['F', 'M'][i], top_k=21)
        lookup = {r['condition']: r['raw_relative_score'] for r in result['predictions']}
        assert np.allclose(scores[i], [lookup[c] for c in bundle['classes']], atol=1e-12)


def test_selection_uses_independent_profile_failures_and_warning_priority():
    record = {'score': np.array([0.95, 0.95, 0.95, 0.95]), 'margin': np.ones(4),
              'positive_count': np.array([3, 3, 3, 1]), 'warning': np.array([False, False, True, False]),
              'groups': np.array([0, 0, 1, 2]), 'correct': np.array([True, False, True, True])}
    result = selection(record, 0.8, 0.1)
    assert result['accepted_rows'] == 2
    assert result['accepted_profile_groups'] == 1
    assert result['all_accepted_group_cases_correct'] == 0


def test_wilson_gate_has_no_empty_sample_approval():
    assert wilson_lower(0, 0) == 0
    assert wilson_lower(0, 100) < 0.01
    assert 0.95 < wilson_lower(100, 100) < 1
