# Repository Guidelines

## Project Structure & Module Organization
The repository centers on Python scheduling utilities stored in the project root. Core optimization lives in , while FIFO benchmarking resides in . Reference technology data is kept in ; keep it versioned because every script reads it during startup. Generated reports (, , charts) are emitted beside the scripts—treat them as disposable outputs when reviewing changes.

## Build, Test, and Development Commands
Work in Python 3.10+ and isolate dependencies in a virtual environment. Create one with  and activate it before installing packages. Install required tooling via . Run the optimization entry point with  to produce the optimized production schedule. Validate the baseline comparison with . For visual checks of the latest outputs, run , which reads the CSV exports and renders the bundled Gantt charts.

## Coding Style & Naming Conventions
Follow PEP 8 defaults: four-space indentation, snake_case for functions and variables, UpperCamelCase for classes, and uppercase constants at module scope. Keep log/print statements bilingual only when necessary for operators; prefer concise English for new additions. When adding helpers, include short docstrings that describe inputs and side effects. Place new scripts at the repository root unless they belong squarely in assets or data.

## Testing Guidelines
 doubles as the primary regression harness—run it after any change that touches scheduling logic or stage definitions. Confirm that generated CSV/TXT outputs open without tracebacks and that printed warnings remain empty. When altering data contracts (, batch sizing), add a minimal scenario in-code or as a fixture file so others can reproduce your edge case locally.

## Commit & Pull Request Guidelines
Commits should mirror the existing short imperative style (e.g., , ). Group related edits and avoid mixing data refreshes with code changes. Pull requests need a concise problem statement, bullet-pointed solution summary, links to tracking issues, and screenshots of new charts if visuals changed. Call out required follow-up work or manual steps in the PR body so reviewers can validate without guesswork.
