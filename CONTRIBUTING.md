# Contributing

Issues and pull requests are welcome on GitHub.

For a bug report, include the smallest input example that reproduces the
issue, the full prediction output, and the Python, PyTorch, and NumPy
versions involved.

For a code change:

1. Add or update tests for the behavior you changed.
2. Run `python -m pytest -q`.
3. If you touch preprocessing, weights, or metrics, re-run
   `python examples/verify_metrics.py` and include its output — the model
   card's numbers must always reproduce from the shipped artifacts.
4. Do not add benchmark claims without the script, data description, and
   measured output needed to reproduce them.
