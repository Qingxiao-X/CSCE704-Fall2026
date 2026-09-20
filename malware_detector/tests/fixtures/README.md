# Test fixtures

`valid_sample.exe` is the benign console launcher stub shipped inside the
standard `setuptools` Python package (`setuptools/cli.exe`), copied here
unmodified from a local pip-installed virtualenv. It is a small, legitimate,
non-malicious PE32 executable used only to exercise the static feature
extractor and the CLI against a real, well-formed PE file. It is **never
executed** by this project or its tests -- only statically parsed.

No malware samples are included in or downloaded by this repository.
