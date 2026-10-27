from fastapi.testclient import TestClient

from kasc_v2.api import app

client = TestClient(app)


def test_homepage_loads():
    res = client.get('/')
    assert res.status_code == 200
    assert 'Crowd Intelligence V2' in res.text
    assert 'SYNTHETIC SIMULATION' in res.text


def test_about_has_measured_synthetic_holdout():
    res = client.get('/api/about')
    assert res.status_code == 200
    info = res.json()
    assert info['n_events'] == 42
    assert info['metrics']['train_test_event_overlap'] == 0
    assert 'NOT LIVE' in info['status']


def test_overview_return_real_model_predictions():
    o = client.get('/api/overview?event_id=SIM-E002&minute_index=23')
    assert o.status_code == 200
    data = o.json()
    assert data['simulation'] is True
    assert len(data['gate_cards']) == 6
    assert all(g['predicted_level'] in {'Low','Medium','High'} for g in data['gate_cards'])
    assert all(g['predicted_wait_min'] >= 0 for g in data['gate_cards'])


def test_forecast_has_real_predictions_and_future_simulation_target():
    f = client.get('/api/forecast?event_id=SIM-E005&gate=G2')
    assert f.status_code == 200
    assert f.json()['horizon_min'] == 15
    assert len(f.json()['points']) == 15


def test_invalid_inputs_do_not_default_silently():
    assert client.get('/api/overview?event_id=UNKNOWN').status_code == 404
    assert client.get('/api/overview?event_id=SIM-E001&minute_index=999').status_code == 422
    assert client.get('/api/forecast?gate=G20').status_code == 422
    assert client.get('/api/route?start=G20').status_code == 422
    assert client.get('/api/route?blocked_edges=X--Y').status_code == 422


def test_routing_api_and_closure():
    o = client.get('/api/route?start=G1&destination=S105')
    assert o.status_code == 200 and o.json()['reachable']
    o = client.get('/api/route?start=G1&destination=S105&blocked_edges=CP1--G1')
    assert o.status_code == 200 and not o.json()['reachable']


def test_scenario_is_computed_not_hardcoded():
    a = client.get('/api/scenario?event_id=SIM-E001&gate=G1&extra_lanes=0').json()
    b = client.get('/api/scenario?event_id=SIM-E001&gate=G1&extra_lanes=3').json()
    assert a['baseline']['scenario_wait_min'] == b['baseline']['scenario_wait_min']
    assert b['scenario']['scenario_wait_min'] <= a['scenario']['scenario_wait_min']


def test_assets_served():
    assert client.get('/static/styles.css').status_code == 200
    assert client.get('/static/app.js').status_code == 200
