"""Parse benchmark_model logs written by phone_benchmark.sh into phone_benchmark.csv / .json (E15).
Usage: python3 parse_phone_results.py results_phone/<timestamp>
benchmark_model reports, per run set: 'Inference timings in us: Init: ..., First inference: ..., Warmup (avg): ..., Inference (avg): ...'
and, with --report_peak_memory_footprint, 'Overall peak memory footprint ... = ... MB'. Per-run latencies are not exposed by the binary,
so the table carries the mean and standard deviation (and min/max when printed); this is recorded as a deviation from the registered
'median and 90th percentile' in the results JSON."""
import sys, re, glob, os, json, statistics
import pandas as pd, numpy as np
out = sys.argv[1]
rows = []
for log in sorted(glob.glob(f'{out}/*.log')):
    name = os.path.basename(log)[:-4]; model, t, s = name.split('__'); txt = open(log, errors='replace').read()
    r = {'model': model, 'threads': int(t[1:]), 'session': int(s[1:]), 'ok': 'Inference timings' in txt}
    m = re.search(r'Inference timings in us: Init: ([\d.]+), First inference: ([\d.]+), Warmup \(avg\): ([\d.]+), Inference \(avg\): ([\d.]+)', txt)
    if m: r.update(init_ms=float(m.group(1)) / 1000, first_ms=float(m.group(2)) / 1000, warmup_avg_ms=float(m.group(3)) / 1000, inference_avg_ms=float(m.group(4)) / 1000)
    m = re.search(r'count=(\d+) first=(\d+) curr=(\d+) min=(\d+) max=(\d+) avg=([\d.]+) std=(\d+)', txt)
    if m: r.update(runs=int(m.group(1)), min_ms=int(m.group(4)) / 1000, max_ms=int(m.group(5)) / 1000, avg_ms=float(m.group(6)) / 1000, std_ms=int(m.group(7)) / 1000)
    m = re.search(r'peak memory footprint[^\n]*?[:=]\s*([\d.]+)', txt, re.I)
    if m: r['peak_memory_MB'] = float(m.group(1))
    m = re.search(r'The input model file size \(MB\): ([\d.]+)', txt)
    if m: r['file_MB'] = float(m.group(1))
    for k in ['before', 'after']:
        m = re.search(rf'battery_temp_decideg_{k}=(\d+)', txt)
        if m: r[f'battery_temp_C_{k}'] = int(m.group(1)) / 10
    r['xnnpack'] = 'XNNPACK' in txt
    rows.append(r)
df = pd.DataFrame(rows)
for c in ['inference_avg_ms', 'min_ms', 'max_ms', 'peak_memory_MB', 'file_MB']:
    if c not in df: df[c] = np.nan
df.to_csv(f'{out}/phone_benchmark.csv', index=False)
summary = df[df.ok].groupby(['model', 'threads']).agg(sessions=('session', 'count'), inference_avg_ms_mean=('inference_avg_ms', 'mean'), inference_avg_ms_sd=('inference_avg_ms', 'std'),
                                                       min_ms=('min_ms', 'min'), max_ms=('max_ms', 'max'), peak_memory_MB=('peak_memory_MB', 'max'), file_MB=('file_MB', 'first')).reset_index()
summary.to_csv(f'{out}/phone_benchmark_summary.csv', index=False)
dev = dict(l.strip().split('=', 1) for l in open(f'{out}/device.txt') if '=' in l)
json.dump({'device': dev, 'binary_sha256': open(f'{out}/benchmark_binary.sha256').read().split()[0], 'runs': rows,
           'deviation': 'benchmark_model reports mean, standard deviation, min and max over the timed runs; per-run latencies are not exposed, '
                        'so the registered median and 90th percentile could not be computed on the phone. The Mac benchmark reports them.'},
          open(f'{out}/phone_benchmark.json', 'w'), indent=1)
pd.set_option('display.width', 200); print(summary.round(2).to_string(index=False)); print('\nwritten:', f'{out}/phone_benchmark_summary.csv')
