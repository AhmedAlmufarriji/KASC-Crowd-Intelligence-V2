from kasc_v2.synthetic import export_csv
from kasc_v2.config import DATA_DIR

if __name__ == '__main__':
    frame = export_csv(DATA_DIR / 'synthetic_gate_flow.csv')
    print(f'Generated {len(frame):,} rows across {frame.event_id.nunique()} synthetic events')
