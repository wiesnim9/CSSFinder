# Changelog

NOTE: CSSFinder follows the [semver](https://semver.org/) versioning standard.

### 0.2.0 - Mar 10, 2023

- Fix SBiPi mode when given fixed system size. System size detection is still
  not working.

### 0.1.0 - Mar 8, 2023

- Add project based execution using `cssfproject.json` file.

- Add report generation for projects with
  `cssfinder project <project-path> task-report <task-name>` command.

- Add fully separable state for n quDits (`FSnQd`) mode.

- Add backend based on numpy with single (32 bit float) and double (64 bit
  float) precision.

- Add CI system with tests, quality checks and automated deployments.
