#!/usr/bin/env bash
# E15: on-device benchmark of the frozen LiteRT files on a physical Android phone, using the official LiteRT
# benchmark_model binary over USB (adb). No application is built or installed. Run from the Mac:
#   bash phone_benchmark.sh models/      (models/ = the folder of .tflite files copied from Drive; see README.md)
# Output: results_phone/<timestamp>/ with one log per (file, threads, session) plus device.txt. Then run parse_phone_results.py.
set -euo pipefail
MODELS_DIR="${1:-models}"
THREADS="1 4"; SESSIONS=3; WARMUP=50; RUNS=200
BIN_URL="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model"
STAMP=$(date -u +%Y%m%dT%H%M%SZ); OUT="results_phone/$STAMP"; mkdir -p "$OUT"
command -v adb >/dev/null || { echo "adb not found: brew install android-platform-tools"; exit 1; }
adb get-state >/dev/null 2>&1 || { echo "No phone found. Enable USB debugging, connect the cable, accept the prompt on the phone."; exit 1; }

echo "== device record =="
{
  echo "utc=$STAMP"
  for p in ro.product.manufacturer ro.product.model ro.product.device ro.board.platform ro.hardware ro.soc.model ro.soc.manufacturer ro.build.version.release ro.build.version.sdk ro.product.cpu.abi; do
    echo "$p=$(adb shell getprop $p | tr -d '\r')"
  done
  echo "mem_total_kB=$(adb shell cat /proc/meminfo | grep MemTotal | awk '{print $2}' | tr -d '\r')"
  echo "cpu_cores=$(adb shell cat /proc/cpuinfo | grep -c ^processor | tr -d '\r')"
  echo "battery_temp_decideg=$(adb shell dumpsys battery | grep temperature | awk '{print $2}' | tr -d '\r')"
  echo "battery_level=$(adb shell dumpsys battery | grep level | awk '{print $2}' | tr -d '\r')"
  echo "thermal_zones_millideg=$(adb shell 'for f in /sys/class/thermal/thermal_zone*/temp; do cat $f 2>/dev/null; done' | tr '\r\n' ' ')"
} | tee "$OUT/device.txt"

echo "== benchmark binary =="
if [ ! -f android_aarch64_benchmark_model ]; then
  curl -L -o android_aarch64_benchmark_model "$BIN_URL"
fi
shasum -a 256 android_aarch64_benchmark_model | tee "$OUT/benchmark_binary.sha256"
adb push android_aarch64_benchmark_model /data/local/tmp/benchmark_model >/dev/null
adb shell chmod +x /data/local/tmp/benchmark_model

echo "== models =="
for f in "$MODELS_DIR"/*.tflite; do
  adb push "$f" /data/local/tmp/ >/dev/null; echo "pushed $(basename "$f") $(stat -f %z "$f") bytes"
done
shasum -a 256 "$MODELS_DIR"/*.tflite | tee "$OUT/models.sha256"

echo "== runs: $SESSIONS sessions x threads {$THREADS} x $(ls "$MODELS_DIR"/*.tflite | wc -l | tr -d ' ') files; $WARMUP warm-up + $RUNS timed each =="
for s in $(seq 1 $SESSIONS); do
  for f in "$MODELS_DIR"/*.tflite; do
    name=$(basename "$f" .tflite)
    for t in $THREADS; do
      log="$OUT/${name}__t${t}__s${s}.log"
      echo "session $s  $name  threads=$t"
      echo "battery_temp_decideg_before=$(adb shell dumpsys battery | grep temperature | awk '{print $2}' | tr -d '\r')" > "$log"
      adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/"$name".tflite --num_threads=$t --warmup_runs=$WARMUP --num_runs=$RUNS \
        --report_peak_memory_footprint=true --use_xnnpack=true 2>&1 | tr -d '\r' >> "$log" || echo "FAILED (see log)"
      echo "battery_temp_decideg_after=$(adb shell dumpsys battery | grep temperature | awk '{print $2}' | tr -d '\r')" >> "$log"
      sleep 5
    done
  done
  echo "cooling 60 s between sessions"; sleep 60
done
echo "done: $OUT"
echo "Now: python3 parse_phone_results.py $OUT"
