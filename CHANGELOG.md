# Changelog

NOTE: CSSFinder follows the [semver](https://semver.org/) versioning standard.

### 0.7.0 - May 19, 2023

- Added G3PaE3qD mode support.
- Changed CLI structure to be flatter.
- Added command allowing for inspecting projects and outputs.
- Added command for batch calculation of summaries for task outputs.
- Bumped version of cssfinder-backend-rust to 0.1.1
- Bumped version of cssfinder-backend-numpy to 0.5.0

### 0.6.0 - April 27, 2023

- Added 32x32 matrix FSnQd example `benchmark_32x32`.
- Added 64x64 matrix FSnQd example `benchmark_64x64`.
- Added automatic flush to json file for perf measurements.
- Added automated task parallel queue.
- Added parallel control parameters to `task run` cli.
- Added extras groups `backend-numpy` and `backend-rust` containing optional
  backend dependencies.
- Added CLI/TUI interface for adding new task.
- Added command for creating new static projects.
- Changed `cssfinder project ./path/to/project` to
  `cssfinder project -p ./path/to/project` (replaced argument with an option).
- Changed logger to output log files with `.log` extension.
- Fixed `jinja2` dependency missing error.
- Bump rich from 13.3.2 to 13.3.3 (#33)
- Bump pydantic from 1.10.6 to 1.10.7 (#32)

### 0.5.0 - Mar 20, 2023

- Add dynamically loaded backends.
- Remove bundled numpy backend. Now it has to be installed separately from
  `cssfinder_backend_numpy`.
- Add automatic priority elevation, may require administrator privileges.

### 0.4.0 - Mar 17, 2023

- Add interface for accessing bundled examples.
- Add support for projections and symmetries.

### 0.3.0 - Mar 15, 2023

- Add HTML and PDF reports.
- Fix matrix shape deduction in SBiPa mode.
- Fix system tests when run in parallel.
- Add examples to `CSSFinder` package (wheel/sdist).

### 0.2.0 - Mar 10, 2023

- Fix SBiPa mode when given fixed system size. System size detection is still
  not working.

### 0.1.0 - Mar 8, 2023

- Add project based execution using `cssfproject.json` file.
- Add report generation for projects with
  `cssfinder project <project-path> task-report <task-name>` command.
- Add fully separable state for n quDits (`FSnQd`) mode.
- Add backend based on numpy with single (32 bit float) and double (64 bit
  float) precision.
- Add CI system with tests, quality checks and automated deployments.
