import math

import numpy as np
import pandas as pd
import pytest

from scripts.train_covid_research import portable_score, split_groups, uncertainty


def test_portable_model_requires_explicit_complete_observations():
    bundle = {'features': ['Symptoms_changes_in_food_flavor'], 'mean': [0.5], 'scale': [0.5], 'coefficients': [1], 'intercept': 0}
    assert portable_score(bundle, {'Symptoms_changes_in_food_flavor': 1.0}) == pytest.approx(1 / (1 + math.exp(-1)))
    for invalid in [{}, {'Symptoms_changes_in_food_flavor': None}, {'Symptoms_changes_in_food_flavor': 2}, {'Symptoms_changes_in_food_flavor': True}]:
        with pytest.raises(ValueError):
            portable_score(bundle, invalid)


def test_rating_and_time_are_not_invented_or_unbounded():
    bundle = {'features': ['Taste_during_illness', 'Days_since_onset'], 'mean': [50, 10], 'scale': [10, 5], 'coefficients': [1, 1], 'intercept': 0}
    assert portable_score(bundle, {'Taste_during_illness': 50, 'Days_since_onset': 10}) == 0.5
    for invalid in [{'Taste_during_illness': 101, 'Days_since_onset': 10}, {'Taste_during_illness': 50, 'Days_since_onset': -1}]:
        with pytest.raises(ValueError):
            portable_score(bundle, invalid)


def test_identical_profiles_cannot_cross_train_validation_test():
    frame = pd.DataFrame({'value': np.repeat(np.arange(100), 2)})
    y = pd.Series(np.tile([0, 1], 100))
    partitions, groups = split_groups(frame, y)
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not set(groups[partitions[a]]) & set(groups[partitions[b]])
    again, _ = split_groups(frame, y)
    assert all(np.array_equal(partitions[k], again[k]) for k in partitions)


def test_paired_bootstrap_gives_zero_difference_for_identical_models():
    y = np.tile([0, 1], 20)
    p = np.where(y == 1, 0.7, 0.3)
    result = uncertainty(y, {name: p for name in ['symptoms', 'symptoms_duration', 'symptoms_sensory', 'all']}, np.arange(40))
    for pair in result['ablation_difference_intervals'].values():
        assert pair['roc_auc'] == [0.0, 0.0]
        assert pair['log_loss'] == [0.0, 0.0]
