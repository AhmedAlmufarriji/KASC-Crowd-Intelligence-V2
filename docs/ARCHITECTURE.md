# Architecture and API

## Scope

A locally runnable independent synthetic stadium research prototype. No real KASC plans, tickets, sensor feeds or team-authored datasets are included.

**Data layer:** `synthetic.py` produces 42 simulated events × 6 gates × 45 labeled bins = 11,340 rows. Each sample represents observations at time `t`, a future wait target at `t+15m`, and a future High/Medium/Low class. Event IDs are group keys only, **not model inputs**.

**Learning layer:** `training.py` applies a seeded 24% event-level held-out split via scikit-learn `GroupShuffleSplit`. Classifier: `RandomForestClassifier`, 90 trees with balancing, trained inside a `ColumnTransformer` preprocessing pipeline. Regressor: `HistGradientBoostingRegressor` with the same feature schema and an independent preprocessing pipeline. Dummy baselines are trained on the same split. All metrics reported only for held-out *synthetic* events.

**Serving layer:** FastAPI supports event/time/gate filters, raw model predictions, synthetic ground-truth comparison, a single illustrated routing topology, walkway closures, and a simple deterministic staffing model.

**Frontend:** No third-party JS framework or CDN, responsive HTML/CSS and SVG, asynchronous calls to real endpoints. The UI is bilingual EN/AR and uses direction switching.

## Trust and safety boundaries

- A result from `/api/forecast` is a computed **t+15-minute model prediction** on synthetic inputs; a displayed synthetic outcome is a separately generated comparison label, not a measurement.
- Routing outputs are graph-derived path costs on a constructed schematic; `travel_min` and edge weights **are not a verified physical distance or transit time**.
- `/api/scenario` applies `max(0, Q_now + arrivals × h - lanes × throughput × h)` and derives future wait by dividing by simulated service rate. It is **not** validated causal inference about staffing outcomes.
- User choices are validated. Invalid event, gate, section or blocked-edge identifiers cause 4xx errors; disconnected routes are explicit, never fabricated.
- Joblib model loading occurs only from a trusted locally generated training artifact. Do not load untrusted `.joblib` files because deserialization can execute code.
- No user data, authentication, personal record, camera access, connection to a venue, evacuation guidance or claims of real-time conditions exist.

## API endpoints

| Endpoint | Purpose | Source |
| --- | --- | --- |
| `GET /api/about` | List 42 synthetic events, topology options, evaluation summary | Seeded generated CSV / local metrics |
| `GET /api/overview?event_id=SIM-E001&minute_index=22` | Six gate conditions, actual model predictions at t+15 minutes | Loaded estimator predictions |
| `GET /api/forecast?event_id=SIM-E001&gate=G1` | Sampled forecast series, including generated future outcomes for comparison | Model + held-out generated labels |
| `GET /api/topology` | Illustrative nodes and edges | Immutable Python routing graph |
| `GET /api/route?event_id=SIM-E001&minute_index=22&start=G1&destination=S105&mode=balanced` | Congestion-weighted or base-weight Dijkstra route | Graph + current gate predictions |
| `GET /api/scenario?event_id=SIM-E001&minute_index=22&gate=G1&extra_lanes=2` | Deterministic queue impact estimate | Fluid queue scenario equation |
| `GET /` | Interactive Arabic/English dashboard | Static HTML/JS served locally |

For routes, `blocked_edges` is a comma-separated list of canonical edge IDs such as `CP1--G1` or `E--N`. All optional user selections are restricted to enumerated values.

## Future production prerequisites

Obtain appropriate authorization, physical wayfinding topology, permissioned timestamped data, a secure ingestion service, feature availability SLAs, calibrated uncertainty, operator review, incident override mechanisms, assistive-accessibility testing, ongoing drift monitoring, and a safe fallback when data is stale or unavailable. The present code does not implement these requirements.
