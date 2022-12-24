# CSSFinder

## Building

To build wheel file You have to use [poetry](https://pypi.org/project/poetry/)
(version 1.3.1 is recommended):

```
pip install poetry==1.3.1
```

Then run:

```
poetry build --format=wheel
```

Wheel file will be created: `dist/cssfinder-x.y.z-py3-none-any.whl`. To install
it, use pip:

```
pip install dist/cssfinder-x.y.z-py3-none-any.whl
```

**Important**: Replace `x.y.z` with current version of CSSFinder. You can find
it in `pyproject.toml` file in root of repository.

```toml
[tool.poetry]
# ...
version = "x.y.z"
# ...
```

## Profiling

Performance profiling is done using dummy dataset from profiling directory.
Mentioned directory also contains `fsqn.py` which calls cssfinder machinery
with fixed set of params.

To run simple profiling, You can use following command:

```
python -mcProfile -o cssf_64_64.prof profiling/fsqn_64_64.py
```

Then You can view output using [snakeviz](https://pypi.org/project/snakeviz/):

```
snakeviz cssf_64_64.prof
```
