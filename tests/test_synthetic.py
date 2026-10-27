import pandas as pd
import pytest

from kasc_v2.synthetic import congestion_label, generate_events
from kasc_v2.training import FEATURES, train_and_evaluate


def test_reproducible_generated_events():
    first = generate_events(n_events=6, bins=12, seed=100)
    second = generate_events(n_events=6, bins=12, seed=100)
    pd.testing.assert_frame_equal(first, second)


def test_shape_and_unique_records():
    frame = generate_events(n_events=6, bins=12)
    assert len(frame) == 6 * 6 * (12 - 3)
    assert frame[['event_id', 'gate', 'minute_index']].duplicated().sum() == 0
    assert frame.event_id.nunique() == 6


def test_label_thresholds():
    assert congestion_label(3.99) == 'Low'
    assert congestion_label(4) == 'Medium'
    assert congestion_label(9) == 'High'


def test_future_target_is_a_later_observation():
    frame = generate_events(n_events=6, bins=14)
    group = frame[(frame.event_id == 'SIM-E001') & (frame.gate == 'G1')].sort_values('minute_index')
    assert group.iloc[0].future_wait_min == group.iloc[3].current_wait_min


def test_future_targets_are_not_in_features():
    assert not set(FEATURES) & {'future_wait_min', 'future_level', 'event_id'}


def test_event_isolated_split_and_baseline():
    frame = generate_events(n_events=10, bins=14)
    result = train_and_evaluate(frame, persist=False)
    assert result['train_event_ids'].isdisjoint(result['test_event_ids'])
    metrics = result['metrics']
    assert metrics['train_test_event_overlap'] == 0
    assert metrics['n_events'] == 10
    assert 0 <= metrics['classifier']['macro_f1'] <= 1


def test_generator_rejects_too_few_events():
    with pytest.raises(ValueError):
        generate_events(n_events=3)


def test_generator_rejects_too_short_series():
    with pytest.raises(ValueError):
        generate_events(bins=5)
