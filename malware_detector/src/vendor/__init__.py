"""Third-party code vendored into this project.

ember_features.py is ember/features.py from https://github.com/elastic/ember
at commit d97a0b523de02f3fe5ea6089d080abacab6ee931, copied unmodified (see
EMBER_LICENSE.txt). Vendored so the Docker build only needs PyPI packages and
so importing it does not pull in ember/__init__.py's training-only imports
(pandas, tqdm, lightgbm).
"""
