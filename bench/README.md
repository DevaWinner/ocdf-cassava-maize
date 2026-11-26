# Benchmarks

Latency and memory of the released LiteRT files, CPU only, measured three ways.

| Script / source | Where it runs | Results |
|---|---|---|
| `notebooks/12_device_benchmark_aihub.ipynb` | real Android phones hosted by Qualcomm AI Hub (Galaxy S24, Snapdragon 8 Gen 3; Pixel 3, Snapdragon 845) | `records/nb12/device_benchmark.csv`, raw profiles in `records/nb12/` |
| `mac_benchmark.py` | any laptop or desktop, network off | `results_mac/20261105T221240Z/` (Apple M5 Pro) |
| `phone_benchmark.sh` + `parse_phone_results.py` | an Android phone over USB with the official LiteRT `benchmark_model` binary | (not used for the released results) |

## Laptop benchmark

```bash
pip install ai-edge-litert numpy pandas psutil
# turn Wi-Fi off and close other applications
python bench/mac_benchmark.py models/phase1
```

Runs every `.tflite` file with 1 and 4 threads: 50 warm-up and 200 timed runs, three sessions.
Writes median, 90th percentile, first-run time and memory increase to `bench/results_mac/<timestamp>/`.
The script never opens a network connection.

## Phone over USB

1. Enable developer options and USB debugging on the phone; install `adb`.
2. `bash bench/phone_benchmark.sh models/phase1` downloads the prebuilt `benchmark_model`
   binary (its SHA-256 is recorded), records the phone's model, chipset, Android version and
   battery temperature, and runs every file with 1 and 4 threads.
3. `python bench/parse_phone_results.py bench/results_phone/<folder>` summarises the logs.

`benchmark_model` reports mean, standard deviation, minimum and maximum, not per-run times, so
medians come from the laptop benchmark and the hosted-phone profiles.
