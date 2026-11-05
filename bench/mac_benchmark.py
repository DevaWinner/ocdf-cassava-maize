"""Offline laptop benchmark (E15): latency (median, 90th percentile) and peak memory of each LiteRT file,
1 and 4 threads, 50 warm-up + 200 timed runs, three sessions, CPU only, network off (the script never opens a socket).
Usage: python3 mac_benchmark.py models/            (needs: pip install ai-edge-litert numpy pandas psutil)
Output: results_mac/<timestamp>/mac_benchmark.csv, mac_benchmark_summary.csv, mac_benchmark.json"""
import sys, os, time, glob, json, platform, hashlib, statistics, datetime, subprocess
import numpy as np, pandas as pd
try:
    from ai_edge_litert.interpreter import Interpreter; backend = 'ai_edge_litert'
except Exception:
    import tensorflow as tf; Interpreter = tf.lite.Interpreter; backend = 'tf.lite'
try: import psutil
except Exception: psutil = None
MODELS = sys.argv[1] if len(sys.argv) > 1 else 'models'
THREADS = [1, 4]; SESSIONS = 3; WARMUP = 50; RUNS = 200; IMG = 224
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'); out = f'results_mac/{stamp}'; os.makedirs(out, exist_ok=True)
def sysctl(k):
    try: return subprocess.run(['sysctl', '-n', k], capture_output=True, text=True).stdout.strip()
    except Exception: return ''
machine = {'utc': stamp, 'platform': platform.platform(), 'machine': platform.machine(), 'python': sys.version.split()[0], 'backend': backend,
           'cpu_brand': sysctl('machdep.cpu.brand_string'), 'cpu_cores': os.cpu_count(), 'mem_total_GB': round(int(sysctl('hw.memsize') or 0) / 2**30, 1)}
rng = np.random.default_rng(2026); x = rng.integers(0, 256, (1, IMG, IMG, 3)).astype(np.float32)   # one fixed synthetic image (latency does not depend on content)
rows = []
for s in range(1, SESSIONS + 1):
    for f in sorted(glob.glob(f'{MODELS}/*.tflite')):
        name = os.path.basename(f)[:-7]; b = open(f, 'rb').read()
        for t in THREADS:
            rss0 = psutil.Process().memory_info().rss if psutil else None
            it = Interpreter(model_content=b, num_threads=t); it.allocate_tensors(); i = it.get_input_details()[0]['index']
            it.set_tensor(i, x); t0 = time.perf_counter(); it.invoke(); first = (time.perf_counter() - t0) * 1000
            for _ in range(WARMUP): it.set_tensor(i, x); it.invoke()
            lat = []
            for _ in range(RUNS):
                it.set_tensor(i, x); t0 = time.perf_counter(); it.invoke(); lat.append((time.perf_counter() - t0) * 1000)
            rss1 = psutil.Process().memory_info().rss if psutil else None
            r = {'model': name, 'threads': t, 'session': s, 'file_MiB': round(len(b) / 2**20, 3), 'sha256': hashlib.sha256(b).hexdigest(), 'first_ms': round(first, 3),
                 'median_ms': round(statistics.median(lat), 3), 'p90_ms': round(float(np.percentile(lat, 90)), 3), 'mean_ms': round(float(np.mean(lat)), 3), 'sd_ms': round(float(np.std(lat, ddof=1)), 3),
                 'min_ms': round(min(lat), 3), 'max_ms': round(max(lat), 3), 'rss_increase_MB': None if rss0 is None else round((rss1 - rss0) / 2**20, 1)}
            rows.append(r); print(f"s{s} {name} t{t}: median {r['median_ms']} ms, p90 {r['p90_ms']} ms, first {r['first_ms']} ms"); del it
df = pd.DataFrame(rows); df.to_csv(f'{out}/mac_benchmark.csv', index=False)
summary = df.groupby(['model', 'threads']).agg(file_MiB=('file_MiB', 'first'), median_ms=('median_ms', 'median'), p90_ms=('p90_ms', 'median'), first_ms=('first_ms', 'median'),
                                               rss_increase_MB=('rss_increase_MB', 'max'), sessions=('session', 'count')).reset_index()
summary.to_csv(f'{out}/mac_benchmark_summary.csv', index=False)
json.dump({'machine': machine, 'settings': {'threads': THREADS, 'sessions': SESSIONS, 'warmup': WARMUP, 'runs': RUNS}, 'runs': rows}, open(f'{out}/mac_benchmark.json', 'w'), indent=1)
pd.set_option('display.width', 200); print(summary.to_string(index=False)); print('written:', out)
