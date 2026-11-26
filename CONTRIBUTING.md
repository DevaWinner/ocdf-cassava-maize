# Contributing

Issues and pull requests are welcome. The most useful contributions are:

- **New test sources.** Results on images from a source not used here, especially Nigerian or
  other West African field photographs with expert labels. Please report per-class recall and
  macro-F1 with group-bootstrap intervals (`src/ocdf/evaluation.py`), and state how images
  were grouped.
- **Runtime ports.** Running the released `.tflite` files on other devices or runtimes, with
  latency measured as in `bench/`.
- **Fixes** to the inference package, scripts or documentation.

Ground rules:

- Do not commit dataset images. Link to the source and describe how to obtain it.
- Do not edit anything in `protocol/` or `records/`: those files are hashed and form the audit
  trail. New experiments belong in new, separately versioned records.
- Run `python -m pytest -q` before opening a pull request.
