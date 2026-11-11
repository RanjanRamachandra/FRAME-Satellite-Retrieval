# Contributing to FRAME Satellite Image Retrieval

Thank you for your interest in contributing. This document explains how
to contribute code, documentation, annotations, or experiments to this
project.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Ways to contribute](#2-ways-to-contribute)
3. [Setting up your environment](#3-setting-up-your-environment)
4. [Branch and commit conventions](#4-branch-and-commit-conventions)
5. [Code style guidelines](#5-code-style-guidelines)
6. [Adding a new script](#6-adding-a-new-script)
7. [Contributing annotations](#7-contributing-annotations)
8. [Running tests](#8-running-tests)
9. [Submitting a pull request](#9-submitting-a-pull-request)
10. [Research integrity requirements](#10-research-integrity-requirements)

---

## 1. Project overview

FRAME is an exploratory research project investigating region-specific
satellite image retrieval using masked embedding prediction on
Himawari-8 AHI data.

The current baseline covers January 2022, three infrared bands, and
224 observations. All results are exploratory. No method-superiority
claim has been established.

Before contributing, read:
- `README.md` — project overview and current results
- `CHANGELOG.md` — history of changes
- `reports/statistical_validation_report.md` — current statistical findings

---

## 2. Ways to contribute

| Type | Examples |
|---|---|
| **Annotations** | Rate candidate images using `create_improved_visual_review.py` |
| **Bug fixes** | Fix incorrect logic, broken paths, wrong column names |
| **New experiments** | Ablation scripts, new model variants, new baselines |
| **Documentation** | Improve docstrings, README sections, report clarity |
| **Statistical review** | Identify flaws in statistical methodology |
| **Code quality** | Refactoring, adding type hints, improving error messages |

---

## 3. Setting up your environment

```bash
# Clone the repository
git clone https://github.com/RanjanRamachandra/FRAME-Satellite-Retrieval.git
cd FRAME-Satellite-Retrieval

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data and checkpoints

Large files are excluded from Git. To reproduce:

1. Run `scripts/download_expanded_28days.sh` to fetch satellite data.
2. Follow the reproduction steps in `README.md` to generate NPZ files.
3. Run training scripts to produce model checkpoints.

---

## 4. Branch and commit conventions

### Branch naming

```
feature/short-description       # new feature or experiment
fix/short-description           # bug fix
docs/short-description          # documentation only
stats/short-description         # statistical methodology change
annotation/short-description    # annotation-related changes
```

Examples:
```
feature/circle-mask-ablation
fix/temporal-distance-calculation
docs/improve-rubric-instructions
stats/add-effect-sizes
```

### Commit message format

```
<type>: <short summary (under 72 chars)>

<optional body — what changed and why>
<optional bullet list of specific changes>
```

Types:
- `feat` — new feature or experiment
- `fix` — bug fix
- `docs` — documentation only
- `refactor` — code restructure without behaviour change
- `stats` — statistical methodology change
- `data` — data pipeline change
- `chore` — maintenance (gitignore, requirements, etc.)

Examples:
```
feat: add circular mask variant to train_ablation_masks.py

fix: correct temporal distance calculation in evaluate_ablation_results.py
- parse_hours() now handles HHMM format correctly for midnight (0000)

docs: add rubric anchor descriptions to create_improved_visual_review.py

stats: replace Wilson intervals with cluster-bootstrap CIs in v3
- corrects within-day query correlation assumption violation
```

---

## 5. Code style guidelines

- Follow **PEP 8** for all Python code.
- Use **4 spaces** for indentation (no tabs).
- Maximum line length: **100 characters**.
- All scripts must have a **module-level docstring** explaining:
  - What the script does
  - How to run it
  - Input files
  - Output files
  - Any important interpretation notes
- Use **descriptive variable names** — avoid single letters except loop counters.
- Add **inline comments** on non-obvious logic.
- All file paths must use `pathlib.Path`, not string concatenation.
- Wrap all file I/O in try/except with informative error messages.

### Example docstring structure

```python
"""
Short one-line description.

Longer explanation if needed.

USAGE
-----
    python scripts/my_script.py

INPUT FILES
-----------
    path/to/input.csv   — description of format

OUTPUT FILES
------------
    path/to/output.csv  — description of contents

INTERPRETATION NOTE
-------------------
    Any caveats about how results should be read.
"""
```

---

## 6. Adding a new script

1. Place it in `scripts/`.
2. Add a module-level docstring (see Section 5).
3. Use `pathlib.Path` for all paths.
4. Print progress and key results to stdout.
5. Save all outputs to the appropriate subfolder under `results/`.
6. Add an entry to `CHANGELOG.md` under a new version or under
   `[Unreleased]`.
7. Update `README.md` if the script changes the main workflow.

---

## 7. Contributing annotations

Annotations are the highest-priority contribution to this project.
The current visual pilot produced no discriminative data (all ratings = 2).

### How to annotate

1. Run `create_improved_visual_review.py` to generate review sheets.
2. Read `results/expanded_2022/improved_review/instructions.txt` carefully.
3. Rate each candidate independently using the 0–3 rubric.
4. Save your completed `rating_template.csv` with your name in the filename:
   `annotator_yourname.csv`
5. Send it to the project maintainer or submit via pull request to:
   `results/expanded_2022/annotations/annotator_yourname.csv`

### Annotation rules

- Rate **independently** — do not discuss ratings with other annotators
  until all ratings are complete.
- Focus **only on the highlighted region**, not the whole image.
- Use the **full range** of scores (0, 1, 2, 3) — avoid giving the
  same score to everything.
- Do **not** modify the query, retrieved_image, or blind_code columns.
- Complete **all rows** — do not leave blanks.

### After annotations are collected

The project maintainer will run:
```bash
python results/expanded_2022/improved_review/merge_ratings.py
python scripts/inter_annotator_agreement.py
```

If Krippendorff's Alpha >= 0.667, annotations will be merged and
retrieval metrics computed with:
```bash
python scripts/compute_retrieval_metrics.py
```

---

## 8. Running tests

There is currently no automated test suite. Before submitting a pull
request, manually verify:

```bash
# Check the dataset loader works
python scripts/expanded_satellite_dataset.py

# Check the statistical validation runs end to end
python scripts/statistical_validation_v3.py

# Check the retrieval metrics script runs (needs annotations)
python scripts/compute_retrieval_metrics.py
```

If you add a new script, include a brief smoke-test at the bottom:

```python
if __name__ == "__main__":
    # Quick sanity check
    print("Smoke test passed.")
```

---

## 9. Submitting a pull request

1. Fork the repository on GitHub.
2. Create a branch following the naming convention in Section 4.
3. Make your changes.
4. Update `CHANGELOG.md` with a description of your changes.
5. Push your branch and open a pull request against `main`.
6. In the pull request description, explain:
   - What you changed and why.
   - What you tested.
   - Any limitations or known issues with your contribution.

Pull requests are reviewed for:
- Correctness of logic and file paths.
- Consistency with the project's research integrity standards.
- Code style (Section 5).
- Documentation completeness (docstrings, CHANGELOG).

---

## 10. Research integrity requirements

This project follows strict research integrity standards. All
contributors must observe the following:

### Do not overclaim results

- Do not add language claiming FRAME outperforms baselines.
- Do not describe temporal separation as evidence of better retrieval.
- Do not describe low FRAME training loss as evidence of retrieval quality.
- Do not use the visual pilot (all ratings = 2) as evidence for
  or against any method.

### Do not fabricate or modify data

- Do not modify relevance ratings to introduce artificial variation.
- Do not alter timestamps or observation IDs.
- Do not change normalization statistics.

### Label exploratory findings correctly

Any findings from the current 7-cluster statistical analysis must be
labelled as **exploratory**. Use language such as:

> "These results are exploratory and should not be interpreted as..."
> "With only 7 daily clusters, this finding has limited statistical power."
> "No method superiority has been established."

### When in doubt

If you are unsure whether a claim is justified by the current evidence,
err on the side of caution and label it exploratory, or open an issue
to discuss before submitting.

---

## Questions?

Open an issue on GitHub or contact the project maintainer.
