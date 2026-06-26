# CSC8646 — Generative AI: Interactive Storytelling App
### Prompt Engineering, Multi-Model Comparison & NLP Bias Detection

Coursework submission for **CSC8646 (Generative AI)**, MSc Data Science & AI, Newcastle University.
This repository contains the deliverables for a brief that asks for **prompting-based solutions** built around a hypothetical interactive children's storytelling app: story generation, a marketing campaign, and a bias-detection pipeline — each compared across multiple Generative AI models.

> **Scope note:** This is an academic, prompt-engineering and analysis project. It is **not** a deployed full-stack application. The only executable code is the Task 3 bias-detection notebook; Tasks 1 and 2 are documented model outputs and a written report.

---

## Table of Contents
- [Aim & Abstract](#aim--abstract)
- [What This Project Does](#what-this-project-does)
- [Outcomes](#outcomes)
- [Directory Structure](#directory-structure)
- [Dataset](#dataset)
- [Image / Output Location Structure](#image--output-location-structure)
- [How to Run (Task 3 Notebook)](#how-to-run-task-3-notebook)
- [Crucial Steps to Keep in Mind](#crucial-steps-to-keep-in-mind)
- [Common Errors & Fixes](#common-errors--fixes)
- [Known Limitations](#known-limitations)
- [Tech Stack](#tech-stack)

---

## Aim & Abstract
The brief frames a software company launching an interactive children's storytelling app and requires three prompting-driven solutions, each critically comparing the output of different Generative AI models, prompting techniques, and parameter settings:

- **Task 1 — Story Generation:** Generate story content using three prompting strategies (basic, structured, creative) and compare outputs across **GPT, Claude, and Google AI Studio**.
- **Task 2 — Marketing Campaign:** Produce multi-modal campaign media and a marketing plan, comparing **GPT** and **Gemini** at basic and advanced prompting levels.
- **Task 3 — Bias Detection:** Programmatically analyse a corpus of ~100 story premises for potential bias (gender roles, cultural/geographic representation, recurring themes) using a Python NLP pipeline runnable in Google Colab.

The accompanying report (`CSC8646_Report.pdf`) documents the iterative prompting process, model comparisons, findings, and conclusions.

## What This Project Does
The reproducible engineering surface lives entirely in **`CSC_8646_Coursework/TASK_3/GENAI.ipynb`**, which:
1. Loads ~100 story premises from `premises.csv`.
2. Counts masculine / feminine / neutral pronoun frequency (token matching) → bar chart.
3. Extracts adjectives associated with gendered subjects via spaCy dependency parsing → bar chart.
4. Runs spaCy Named Entity Recognition, filtering `GPE` and `NORP` entities to map cultural/geographic representation → bar chart.
5. Builds a noun-frequency word cloud of recurring narrative themes.
6. Prints an automated text summary (pronoun distribution percentages, top cultural references, top themes) with interpretation guidance.

Tasks 1 and 2 are delivered as PDF exports of model outputs (no code).

## Outcomes
- A side-by-side comparison of how different LLMs handle the same storytelling and marketing prompts under varying prompting techniques.
- A quantitative bias profile of the premises dataset: gender-pronoun skew, dominant cultural/geographic references, and recurring themes — surfaced through charts and a printed summary.
- A written report tying the three tasks together with findings and conclusions.

---

## Directory Structure
```
GENERATIVE-AI/
├── README.md
├── .gitattributes
├── .gitignore
├── RAW/
│   ├── Coursework.pdf            # Assignment brief
│   └── premises.csv              # Dataset: ~100 story premises (provided)
└── CSC_8646_Coursework/
    ├── CSC8646_Report.pdf        # Final written report
    ├── TASK_1/                   # Story generation — model outputs (PDF)
    │   ├── GPT/        GPT_BASIC.pdf, GPT_STRUCTURED.pdf, GPT_CREATIVE.pdf
    │   ├── CLAUDE/     CLAUDE_BASIC.pdf, CLAUDE_STRUCTURED.pdf, CLAUDE_CREATIVE.pdf
    │   └── AI_STUDIO/  AI_STUDIO_T02.pdf, AI_STUDIO_T07.pdf, AI_STUDIO_T10.pdf
    ├── TASK_2/                   # Marketing campaign — model outputs (PDF)
    │   ├── GPT/        GPT_BASIC.pdf, GPT_ADVANCED.pdf
    │   └── GEMINI/     GEMINI_BASIC.pdf, GEMINI_ADVANCED.pdf
    └── TASK_3/                   # Bias detection — code + generated charts
        ├── GENAI.ipynb           # The runnable analysis notebook
        ├── Gender Pronoun Frequency in Story Premises.png
        ├── Top Gender-Associated Adjectives.png
        ├── Top Cultural_Geographic References.png   # see note below
        ├── Common Narrative Themes Word Cloud.png
        └── final bias.png
```

## Dataset
- **File:** `RAW/premises.csv` — ~100 single-line story premises, one per row, no header.
- **Availability:** This dataset is **not publicly available**. It was **provided by the module professor at Newcastle University** as part of the CSC8646 coursework brief and is included here solely for reproducing the analysis.
- **Format:** Plain CSV, single text column. The notebook reads it with `header=None` and assigns the column name `premise`.

## Image / Output Location Structure
All charts generated by the Task 3 notebook are stored alongside it in **`CSC_8646_Coursework/TASK_3/`**:

| File | Produced by |
|------|-------------|
| `Gender Pronoun Frequency in Story Premises.png` | Pronoun frequency analysis (Step 3) |
| `Top Gender-Associated Adjectives.png` | Gendered adjective extraction (Step 4) |
| `Top Cultural_Geographic References.png` | NER on GPE/NORP entities (Step 5) |
| `Common Narrative Themes Word Cloud.png` | Theme word cloud (Step 6) |
| `final bias.png` | Summary visual |

> ⚠️ **Filename warning:** The committed file is currently named `Top Cultural:Geographic References.png`. The **colon (`:`)** is illegal in Windows filenames and breaks `git checkout`/clone on Windows and several CI tools. **Rename it to `Top Cultural_Geographic References.png`** (underscore) before relying on cross-platform clones. The table above already uses the corrected name.

---

## How to Run (Task 3 Notebook)
The notebook was written for **Google Colab** and reads the dataset from Google Drive as-is. Two paths:

### Option A — Google Colab (as written)
1. Open `CSC_8646_Coursework/TASK_3/GENAI.ipynb` in Colab.
2. Upload `premises.csv` to your Drive at `MyDrive/premises.csv` (the notebook expects this exact path).
3. Run all cells. The first cell installs dependencies and downloads the spaCy model; the second mounts Drive.

### Option B — Local / clone (recommended for reproducibility)
1. Clone the repo:
   ```bash
   git clone https://github.com/Krishnanand-hub/GENERATIVE-AI.git
   cd GENERATIVE-AI
   ```
2. Create an environment and install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install pandas numpy matplotlib seaborn spacy nltk wordcloud jupyter
   python -m spacy download en_core_web_sm
   ```
3. **Edit the data-loading cell.** Replace the Colab/Drive block:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   file_path = '/content/drive/MyDrive/premises.csv'
   ```
   with a relative read of the dataset already in the repo:
   ```python
   file_path = 'RAW/premises.csv'
   ```
4. Launch and run:
   ```bash
   jupyter notebook CSC_8646_Coursework/TASK_3/GENAI.ipynb
   ```

## Crucial Steps to Keep in Mind
- **Download the spaCy model.** `en_core_web_sm` is required for NER and dependency parsing. Without it, the notebook fails on `spacy.load("en_core_web_sm")`.
- **Fix the data path for local runs.** The hardcoded Drive path (`/content/drive/MyDrive/premises.csv`) only works in Colab. Locally, point it at `RAW/premises.csv`.
- **The CSV has no header.** It must be read with `header=None`; the notebook then assigns the `premise` column name. Don't add a header row.
- **Run cells in order.** Later cells (the summary) depend on variables (`male_count`, `entity_counts`, `theme_counts`) defined in earlier cells.

## Common Errors & Fixes
| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: premises.csv` | Dataset not at the expected path | Use `RAW/premises.csv` locally, or place the file at `MyDrive/premises.csv` in Colab |
| `OSError: [E050] Can't find model 'en_core_web_sm'` | spaCy model not downloaded | Run `python -m spacy download en_core_web_sm` |
| `ModuleNotFoundError: wordcloud` (or spacy/nltk) | Dependencies not installed | Install per the steps above; on some systems `wordcloud` needs build tools (`pip install wheel` first) |
| `ModuleNotFoundError: google.colab` | Running locally | Remove the Colab/Drive cell and read the CSV directly (Option B, step 3) |
| Clone fails / file missing on Windows | Colon in `Top Cultural:Geographic References.png` | Rename the file to use an underscore |

## Known Limitations
- **Gendered-adjective extraction is a weak heuristic.** It collects adjectives among the children of each pronoun's head token. Pronouns don't take adjectival modifiers, so results are noisy and don't reliably capture character traits. Treat that chart as indicative, not conclusive; resolving pronouns to their referent subject (coreference) would be the correct fix.
- **Pronoun counting conflates senses** (e.g. "her" as possessive vs. object) — acceptable for a coarse distribution, but not a precise measure.
- **No pinned dependencies.** There is no `requirements.txt`; versions are not locked, so results may vary across spaCy versions.

## Tech Stack
- **Language:** Python 3
- **NLP:** spaCy (`en_core_web_sm`), NLTK
- **Data / viz:** pandas, NumPy, matplotlib, seaborn, wordcloud
- **Environment:** Google Colab (notebook as written); runs locally with the data-path change above
- **Models compared (Tasks 1 & 2):** GPT, Claude, Google Gemini, Google AI Studio
