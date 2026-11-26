# AI-Powered Code Complexity & Technical Debt Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React_18-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React" />
  <img src="https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/XGBoost-EB6424?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License" />
</p>

An enterprise-grade, predictive software engineering platform that analyzes Git commit histories, extracts multi-language AST/graph metrics (Python, TypeScript, TSX, JavaScript), and uses machine learning to **forecast which source files will become bug-prone or expensive maintenance bottlenecks over the next 6 months**.

Crucially, the system pairs each prediction with **Game-Theoretic SHAP Explainability** and **Automated Martin Fowler Refactoring Prescriptions**, prioritizing code remediation before defects reach production.

---

## 📌 Table of Contents
1. [The Problem: Silent Technical Debt](#-the-problem-silent-technical-debt)
2. [Industry Significance & Commercial Parallels](#-industry-significance--commercial-parallels)
3. [System Architecture & Dataflow](#-system-architecture--dataflow)
4. [Subsystem Deep-Dive](#-subsystem-deep-dive)
   - [A. Polyglot Static Analysis & Graph Theory Engine](#a-polyglot-static-analysis--graph-theory-engine)
   - [B. Git History & Churn Dynamics Engine](#b-git-history--churn-dynamics-engine)
   - [C. Leak-Free Temporal Snapshotting](#c-leak-free-temporal-snapshotting)
   - [D. Developer Frustration & NLP Sentiment Mining](#d-developer-frustration--nlp-sentiment-mining)
   - [E. Machine Learning Classifier & Probability Calibration](#e-machine-learning-classifier--probability-calibration)
   - [F. Effort-Aware Defect Prediction (P_opt & Recall@20%LOC)](#f-effort-aware-defect-prediction-p_opt--recall20loc)
   - [G. Game-Theoretic SHAP & Fowler Refactoring Engine](#g-game-theoretic-shap--fowler-refactoring-engine)
   - [H. Full-Stack Web Application & GitHub CI/CD](#h-full-stack-web-application--github-cicd)
5. [Repository Structure](#-repository-structure)
6. [Getting Started (Local Installation)](#-getting-started-local-installation)
7. [API Reference](#-api-reference)
8. [Testing & Verification](#-testing--verification)
9. [Academic Defense & Viva Voce Guide](#-academic-defense--viva-voce-guide)
10. [Future Engineering Roadmap](#-future-engineering-roadmap)
11. [Research Foundations & Citations](#-research-foundations--citations)

---

## 🚨 The Problem: Silent Technical Debt

Software technical debt is estimated to cost organizations **$1.52 Trillion annually** (CISQ Report). 

In modern agile teams, debt accumulates silently:
* Modules do not fail overnight; rather, they experience continuous, chaotic changes, developer turnover, increasing architectural coupling, and creeping cognitive complexity.
* Traditional static analysis linters (SonarQube, ESLint, Pylint) merely count **static syntax violations** at a single instant in time. They cannot answer the forward-looking question: *"Which file has an 80% probability of causing a production outage in the next quarter?"*
* Engineering managers lack quantitative, predictive prioritization to justify refactoring sprints before deadlines are derailed.

---

## 🏢 Industry Significance & Commercial Parallels

This platform is modeled after proprietary internal quality systems built by major tech companies and specialized software intelligence startups:
* **Google's Critique & Tricorder:** Analyzes change histories and static analysis warnings to block risky code during code review.
* **CodeScene (Empathic Code):** Commercial Swedish platform combining behavioral Git metrics (hotspots, author churn) with code complexity to score architectural debt.
* **Meta's Infer & Predictive Test Selection:** Uses historical change patterns to predict fault probability and optimize testing budgets.

This project delivers an open, transparent, and rigorous implementation bridging **Data Structures & Graph Algorithms (DSA)**, **Machine Learning (ML)**, and **Full-Stack Systems Engineering**.

---

## 🏗 System Architecture & Dataflow

The system follows a modular 4-tier architecture:

```mermaid
flowchart TD
    subgraph Layer1["1. Data Ingestion & Extraction Layer"]
        GH["GitHub URL / Local Repo"] --> A1["Git Mining Engine<br/>(GitPython / PyDriller)"]
        GH --> A2["Polyglot AST Analyzer<br/>(Radon + Lizard)"]
        A1 -->|Commit log, Churn, Authors, Timezones| STORE[("Commit & Static Cache<br/>(data/cache/)")]
        A2 -->|Cyclomatic, Nesting, Cycles, PageRank| STORE
    end

    subgraph Layer2["2. Temporal Feature Engineering Layer"]
        STORE --> A3["Temporal Snapshot Engine<br/>(pipeline/dataset.py)"]
        A3 -->|History before T| F_PAST["Past Process & Product Features"]
        A3 -->|Window [T, T + 6mo)| F_FUTURE["Future Defect Labeling<br/>(is_bug_fix &ge; 1)"]
        F_PAST --> B1["Developer NLP Analyzer<br/>(Frustration & Urgency)"]
        B1 --> MATRIX["Leak-Free Training Matrix<br/>(data/processed/dataset.csv)"]
        F_FUTURE --> MATRIX
    end

    subgraph Layer3["3. Machine Learning & XAI Core"]
        MATRIX --> B2["Expanding-Window CV<br/>(2019..2022 Folds)"]
        B2 --> B3["Model Training & Tuning<br/>(XGBoost / Logistic / RF)"]
        B3 --> CALIB["Platt Scaling Sigmoid Calibration<br/>P(y=1|x)"]
        CALIB --> EFFORT["Effort-Aware Evaluation<br/>(Recall@20%LOC, P_opt)"]
        CALIB --> B4["Game-Theoretic SHAP<br/>(TreeSHAP / LinearSHAP)"]
        B4 --> ADVICE["Martin Fowler Refactoring Engine<br/>(Actionable Code Advice)"]
    end

    subgraph Layer4["4. Full-Stack Application & CI/CD Layer"]
        CALIB --> C1["FastAPI REST Backend (:8001)<br/>(Async Background Scans)"]
        ADVICE --> C1
        C1 --> DB[("SQLite / PostgreSQL<br/>(risk_predictor.db)")]
        C1 <--> C2["GitHub App Webhook Engine<br/>(Automated PR Risk Comments)"]
        C1 <--> D1["React 18 / TypeScript Dashboard (:5173)<br/>Heatmap, FileTable, Risk Trends"]
    end
```

### Detailed Sequence Diagram: The Scan Lifecycle

```mermaid
sequenceDiagram
    autonumber
    actor Manager as Engineering Manager
    participant UI as React Dashboard (:5173)
    participant API as FastAPI Backend (:8001)
    participant Pipe as Scanner (pipeline/scanner.py)
    participant AST as Static & Graph Engine
    participant ML as ML Inference (ml/predictor.py)
    participant DB as SQLite / PostgreSQL

    Manager->>UI: Enter repo URL (e.g. psf/requests)
    UI->>API: POST /api/repositories/{id}/scans
    API-->>UI: 202 Accepted (Scan queued)
    
    rect rgb(240, 248, 255)
    Note over API,Pipe: Async Worker Execution
    API->>Pipe: scan_repository(repo_url, history_points=6)
    Pipe->>Pipe: Clone or update repo in data/repos/
    Pipe->>Pipe: Parse Git commit history (PyDriller incremental)
    Pipe->>AST: Analyze source files (.py, .ts, .tsx, .js, .jsx)
    AST->>AST: Compute Cyclomatic & Cognitive Complexity
    AST->>AST: Detect Circular Cycles (Tarjan SCC) & PageRank
    Pipe->>Pipe: Merge features at snapshot date T (today)
    Pipe-->>API: ScanResult (features matrix)
    end

    rect rgb(245, 255, 245)
    Note over API,ML: Model Inference & Calibration
    API->>ML: RiskPredictor.predict(features)
    ML->>ML: Apply log1p scaling & Platt calibration
    ML->>ML: Compute SHAP attributions & factor grouping
    ML->>ML: Match Fowler refactoring recommendations
    ML-->>API: Predictions (Risk score, rank, SHAP, advice)
    end

    API->>DB: Store FileAnalysis rows & commit history
    API->>DB: Mark scan status = 'completed'
    
    loop Every 3s Polling
        UI->>API: GET /api/scans/{id}
        API-->>UI: Status: completed (63 files scored)
    end
    UI->>Manager: Render Interactive Treemap, Risk Donut & FileTable
```

---

## 🔬 Subsystem Deep-Dive

### A. Polyglot Static Analysis & Graph Theory Engine
*Located in [`pipeline/static_analysis.py`](pipeline/static_analysis.py)*

The static analysis engine evaluates code across multiple programming languages: **Python (`.py`)**, **TypeScript (`.ts`, `.tsx`)**, and **JavaScript (`.js`, `.jsx`)** using AST parsers and **Lizard**.

```
                           Module Import Graph
                  ┌─────────────────────────────────┐
                  │                                 ▼
           ┌──────────────┐     imports      ┌──────────────┐
           │   module_A   │ ───────────────► │   module_B   │
           └──────────────┘                  └──────────────┘
                  ▲                                 │
                  │             imports             │
                  └─────────────────────────────────┘
             Tarjan's SCC identifies circular cycle: {A, B}
```

Key Algorithmic Implementations:
1. **Tarjan's Strongly Connected Components (SCC) — $O(V + E)$:**
   Detects direct and transitive circular dependency cycles. Cycles violate the *Acyclic Dependencies Principle (ADP)*, cause memory retention, and amplify change propagation. Outputs `in_cycle (1/0)` and `cycle_size`.
2. **PageRank Centrality:**
   Executes power-iteration over the repository's module dependency graph. Files with high PageRank are architectural hubs; a defect in a high-PageRank file triggers massive cascading system failures.
3. **Cognitive Complexity (SonarQube Specification):**
   Unlike cyclomatic complexity (which simply counts branches $M = E - N + 2P$), cognitive complexity adds nesting penalties (loops inside conditionals inside functions) to measure true human mental effort required to understand control flow.

---

### B. Git History & Churn Dynamics Engine
*Located in [`pipeline/git_history.py`](pipeline/git_history.py)*

Extracts empirical process metrics from Git commit logs with robust handling of renames, broken timezones, and merge commits:
* **Shannon Churn Entropy ($H(P)$):**
  $$H(P) = -\sum_{i=1}^n p_i \log_2(p_i) \quad \text{where } p_i = \frac{\text{churn}_i}{\text{total churn}}$$
  *High entropy* indicates modifications were scattered chaotically across dozens of small commits; *low entropy* indicates focused overhauls. Hassan (ICSE 2009) demonstrated that entropy is far more predictive of bugs than raw line count.
* **Code Ownership Fragmentation (Minor Authors Count):**
  Counts developers who contributed $< 5\%$ of a file's lifetime churn. Bird et al. (Microsoft Research, FSE 2011) proved that files touched by multiple minor contributors have significantly elevated post-release failure rates due to lack of domain stewardship.
* **Bug-Fix Keyword Heuristics:**
  Uses strict regex patterns (`fix`, `bug`, `defect`, `crash`, `resolves #123`) while filtering out non-behavioral changes (typos, docs, linting, formatting) and bulk changes touching $> 10$ files.

---

### C. Leak-Free Temporal Snapshotting
*Located in [`pipeline/dataset.py`](pipeline/dataset.py)*

In software engineering ML, **random train/test splits cause catastrophic data leakage** due to temporal autocorrelation (a file's code today is 99% correlated with its code yesterday).

Our pipeline constructs time-windowed historical snapshots:
```
Time Timeline:
───|─────────────────────────────[ T ]─────────────────────────────|──────────►
   Past Commits & Code (t < T)          Forward Horizon [T, T + 6mo)
   -> Used to compute features          -> Used ONLY to observe future bug-fix commits
   (Churn, Complexity, Authors)            label_bug_prone = (future_bug_fixes >= 1)
```
* **Strict Embargo:** Features never peek forward; labels never look backward.
* **Window Dropping:** Snapshots whose 6-month forward horizon exceeds the repository's latest commit are discarded to eliminate partial-window labeling bias.

---

### D. Developer Frustration & NLP Sentiment Mining
*Located in [`ml/nlp.py`](ml/nlp.py)*

Mining affective signals from developer communication:
* **Frustration Lexicon:** Detects emotional stress (`stupid`, `again`, `wtf`, `ugh`, `broken`, `failing`, `!!`).
* **Rushed-Work Lexicon:** Flags temporary hacks (`wip`, `quick fix`, `hack`, `workaround`, `for now`, `oops`, `todo`).
* **HuggingFace Emotion Transformer:** Employs DistilRoBERTa (`j-hartmann/emotion-english-distilroberta-base`) to score $P(\text{anger}) + P(\text{disgust})$ per commit message, with graceful offline fallback (`RISK_USE_HF=0`).

---

### E. Machine Learning Classifier & Probability Calibration
*Located in [`ml/train.py`](ml/train.py), [`ml/models.py`](ml/models.py), and [`ml/predictor.py`](ml/predictor.py)*

1. **Feature Transformation:**
   Software metrics follow heavy-tailed power-law distributions. A clipped `log1p_nonneg(x) = \log(1 + \max(0, x))` transformation normalizes counts before gradient descent.
2. **Expanding-Window Cross-Validation:**
   Evaluates validation folds across historical years (2019..2022). Training snapshots must have their label window completed before January 1 of the validation year.
3. **Class Imbalance Handling:**
   Bug-prone files represent $10\% - 15\%$ of a codebase. We employ cost-sensitive loss weighting:
   * XGBoost: `scale_pos_weight = n_negative / n_positive`
   * Random Forest: `class_weight="balanced_subsample"`
   * Logistic Regression: `class_weight="balanced"`
   * *Why not SMOTE?* SMOTE interpolates between points in continuous space, generating physically impossible software code metrics (e.g. 2.4 classes or broken AST trees).
4. **Platt Sigmoid Calibration:**
   $$P(y = 1 \mid x) = \frac{1}{1 + e^{-(a \cdot f(x) + b)}}$$
   Calibrates uncalibrated tree margins on held-out validation predictions so output scores reflect true empirical probabilities (e.g. a score of 0.35 means an estimated 35% chance of a defect).

---

### F. Effort-Aware Defect Prediction (P_opt & Recall@20%LOC)
*Located in [`ml/effort.py`](ml/effort.py)*

Standard ROC-AUC treats a 5,000-line God class as having the same inspection cost as a 50-line utility. Following Menzies et al. and Kamei et al. (IEEE TSE), we benchmark models under finite engineering budgets:

$$\text{Recall@20\%LOC} = \frac{\sum_{i \in \text{top files within 20\% LOC}} \text{Defects}_i}{\sum_{\text{all files}} \text{Defects}_i}$$

$$P_{opt} = 1 - \frac{\text{AUC}_{\text{optimal}} - \text{AUC}_{\text{model}}}{\text{AUC}_{\text{optimal}} - \text{AUC}_{\text{worst}}}$$

* **Target:** Catching $\ge 65\%$ of future bug-prone files by auditing the top-risk files representing only $20\%$ of total codebase lines.
* **$P_{opt}$ Score:** Normalizes performance between 0.0 (worst-case ordering) and 1.0 (optimal oracle). Random guessing yields 0.50; our model achieves $\ge 0.70$.

---

### G. Game-Theoretic SHAP & Fowler Refactoring Engine
*Located in [`ml/explain.py`](ml/explain.py)*

Rather than providing black-box predictions, every score is explained using **TreeSHAP / LinearSHAP** grounded in cooperative game theory:
1. **The 4 Shapley Axioms:** Efficiency ($\sum \phi_i = f(x) - E[f(x)]$), Symmetry, Dummy Player, and Additivity.
2. **8-Factor Aggregation:** Collinear features (e.g. `loc` and `sloc`) are summed into 8 intuitive categories: *Size*, *Complexity*, *Change Activity*, *Bug History*, *Team / Ownership*, *Coupling*, *Tests*, and *Commit Messages*.
3. **Automated Fowler Refactoring Prescriptions:**

| Elevated SHAP Factor | Trigger Condition | Prescribed Martin Fowler Refactoring Pattern |
| :--- | :--- | :--- |
| **Complexity** | High Cognitive / Cyclomatic Complexity | *Extract Method*, *Decompose Conditional*, *Replace Nested Conditional with Guard Clauses* |
| **Coupling** | Circular Import Detected (`in_cycle = 1`) | *Dependency Inversion Principle*, *Introduce Mediator*, *Extract Interface* |
| **Size** | Monolithic File ($\ge 400$ LOC) | *Extract Class*, *Replace Method with Method Object* |
| **Team / Ownership** | Minor Contributors ($\ge 3$) | *Assign Code Steward / Code Owner approval required in GitHub CODEOWNERS* |
| **Change Activity** | High Churn Entropy ($H(P) > 2.5$) | *Split Shotgun Surgery commits into atomic, focused Pull Requests* |
| **Tests** | Complex file with no test importer | *Add Targeted Unit Tests covering boundary conditions* |

---

### H. Full-Stack Web Application & GitHub CI/CD
*Located in [`backend/`](backend/) and [`frontend/`](frontend/)*

* **FastAPI Backend (`http://127.0.0.1:8001`):** Asynchronous background scan worker, JWT bearer authentication, rate limiting, and SQLite/PostgreSQL support via SQLAlchemy 2.0.
* **React 18 Dashboard (`http://127.0.0.1:5173`):**
  * **Zoomable Codebase Heatmap (D3-hierarchy):** Treemap where box area = lines of code and color = predicted risk gradient (green $\to$ yellow $\to$ red).
  * **Risk Donut & Quality Averages:** Breakdown of files by Critical, High, Medium, Low risk.
  * **Prioritized Risk Table:** Ranked 1..N with interactive SHAP factor bars, cycle badges, search, and pagination.
  * **Technical Debt Forecast Chart (Recharts):** 6/12/18-month forward linear regression trend with 95% confidence intervals.
* **GitHub App Webhook Integration:** Listens for `pull_request` events to run automated risk reviews on PR diffs, updating a status comment directly on GitHub.

---

## 📂 Repository Structure

```
ai-code-risk-predictor/
├── backend/                  # FastAPI Application Layer
│   ├── app/
│   │   ├── api/routes/       # Endpoints: auth, repositories, scans, github webhooks
│   │   ├── models/           # SQLAlchemy ORM models (Repository, Scan, FileAnalysis)
│   │   ├── schemas/          # Pydantic v2 validation contracts
│   │   └── services/         # Background scan runner, GitHub App HMAC verifier
│   └── risk_predictor.db     # SQLite persistence database
│
├── frontend/                 # React 18 + TypeScript + Vite Dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── charts/       # Recharts: DebtChart, ShapChart, RiskHistoryChart
│   │   │   ├── heatmap/      # D3 Treemap Codebase Heatmap
│   │   │   └── report/       # FileTable, RiskDonut, QualityPanel
│   │   ├── pages/            # DashboardPage, RepoPage, FilePage, LoginPage
│   │   └── metrics.ts        # UI metric group definitions & formatters
│   └── vite.config.ts        # Dev server proxying /api to port 8001
│
├── pipeline/                 # Member 1 — Data Extraction & Graph Theory
│   ├── dataset.py            # Temporal snapshot builder & past/future joiner
│   ├── filters.py            # Polyglot extensions (.py, .ts, .tsx, .js, .jsx) & vendor filters
│   ├── git_history.py        # Churn, author entropy, bug-fix commits mining
│   ├── scanner.py            # Live repository scan orchestrator
│   ├── static_analysis.py    # Radon + Lizard AST, Tarjan's SCC cycles, PageRank
│   └── store.py              # Incremental commit & static pickle caching
│
├── ml/                       # Member 2 — Machine Learning & Explainability
│   ├── effort.py             # Effort-Aware Defect Prediction (Recall@20%LOC, P_opt)
│   ├── explain.py            # Game-Theoretic SHAP + Fowler Refactoring advice
│   ├── forecast.py           # Technical debt projection over time
│   ├── models.py             # Custom log1p transformation & RankAverageEnsemble
│   ├── nlp.py                # Frustration & urgency commit sentiment
│   ├── predictor.py          # Production RiskPredictor inference & calibration
│   └── train.py              # Expanding-window temporal CV, tuning & evaluation
│
├── tests/                    # Comprehensive Test Suite (37 tests)
│   ├── ml/                   # Model validation, SHAP, and effort-aware tests
│   └── pipeline/             # Synthetic git repo tests, AST tests, Tarjan SCC tests
│
├── generate_viva_pdf.py      # ReportLab automated PDF documentation generator
└── ML_Preparation_Code_and_Viva_Guide.pdf  # 8-Page Printable Defense Masterclass
```

---

## 🚀 Getting Started (Local Installation)

### Prerequisites
* **Python 3.11 or 3.12**
* **Node.js 18+ & npm**
* **Git** installed on PATH

### 1. Clone the Repository & Setup Python Environment
```powershell
# Navigate to the workspace
cd e:\clg-PROJJJ\ai-code-risk-predictor-main

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install pipeline, ml, and backend dependencies
pip install -r pipeline/requirements.txt -r backend/requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the `backend/` directory (or use dev defaults):
```ini
SECRET_KEY=dev-secret-key-for-local-testing-only-replace-in-prod
ENVIRONMENT=development
RISK_USE_HF=0
```
*(Setting `RISK_USE_HF=0` disables heavy HuggingFace downloads on local machines and runs fast lexicon sentiment).*

### 3. Start the FastAPI Backend
```powershell
cd backend
$env:PYTHONPATH=".;.."
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```
The backend API and Swagger docs will be live at:
* Swagger UI: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)
* Health Check: [http://127.0.0.1:8001/api/health](http://127.0.0.1:8001/api/health)

### 4. Start the React Frontend Dashboard
In a second terminal:
```powershell
cd e:\clg-PROJJJ\ai-code-risk-predictor-main\frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```
Open your browser at: **`http://127.0.0.1:5173`**

### 5. Login & Run a Repository Scan
* **Default Demo Credentials:**
  * Email: `demo@example.com`
  * Password: `demo-password`
* **Scan a Repository:**
  Enter any public GitHub URL (e.g. `https://github.com/psf/requests` or `https://github.com/pallets/flask` or your own multi-language TypeScript/React repo) and click **Add & Scan**. The dashboard will poll asynchronously and populate all heatmaps and ranking tables.

---

## 📡 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/auth/register` | Register a new user account |
| `POST` | `/api/auth/login` | Authenticate and obtain JWT bearer token |
| `GET` | `/api/repositories` | List all tracked repositories with latest scan status |
| `POST` | `/api/repositories` | Connect a new GitHub repository URL |
| `POST` | `/api/repositories/{id}/scans` | Trigger an asynchronous background scan |
| `GET` | `/api/scans/{id}` | Get scan summary, quality averages, and risk counts |
| `GET` | `/api/scans/{id}/files` | Paginated, sortable list of scored files with SHAP reasons |
| `GET` | `/api/scans/{id}/files/{file_id}` | Detailed file view with SHAP factors, Fowler advice, and history |
| `GET` | `/api/repositories/{id}/debt-forecast` | Historical and projected technical debt with 95% confidence bands |
| `POST` | `/api/github/webhook` | GitHub App webhook endpoint (processes PR reviews and push events) |

---

## 🧪 Testing & Verification

The test suite utilizes a **synthetic Git repository fixture** (generating 60 dated commits across 3 simulated authors with renames, bug fixes, and circular dependencies) to verify the entire pipeline offline without hitting GitHub rate limits:

```powershell
# Run full pipeline and ML test suite
$env:PYTHONPATH="."
$env:RISK_USE_HF="0"
python -m pytest tests/pipeline tests/ml -v
```

**Test Coverage Summary:**
* `tests/pipeline/test_dataset_scanner.py`: Verifies zero future data leakage, snapshot alignment, and scanner-to-model column contracts.
* `tests/pipeline/test_git_history.py`: Verifies churn calculations, author entropy, bot filtering, and timezone offsets.
* `tests/pipeline/test_static_analysis.py`: Verifies Lizard polyglot parsing, Cognitive Complexity spec, and Tarjan's SCC cycle detection.
* `tests/ml/test_effort.py`: Verifies $Recall@20\%LOC$ interpolation and $P_{opt}$ area normalization against oracle curves.
* `tests/ml/test_model.py`: Verifies SHAP efficiency axiom ($\sum \phi_i = \text{score} - \text{base}$), probability calibration, and priority ranking.

---

## 🎓 Academic Defense & Viva Voce Guide

An **8-page printable PDF technical guide** is pre-compiled in the root directory for capstone presentations, technical interviews, and university viva defenses:
* **PDF File:** [`ML_Preparation_Code_and_Viva_Guide.pdf`](ML_Preparation_Code_and_Viva_Guide.pdf)
* **Interactive Markdown:** [`ml_viva_preparation_guide.md`](C:/Users/Chitresh%20Bhatia/.gemini/antigravity/brain/7f14e8f2-20a2-4713-b351-9a96ccba70b5/ml_viva_preparation_guide.md)

### Top Viva Voce Quick-Fire Answers:
1. **Why not Deep Learning (LLMs/CodeBERT)?** Tree-based models dominate heterogeneous tabular data (Grinsztajn et al., NeurIPS 2022); they provide sub-second inference in CI/CD, require no GPU, avoid hallucinations, and permit exact polynomial-time TreeSHAP calculations.
2. **Why expanding temporal CV over K-Fold?** Code evolution has high temporal autocorrelation. Random splitting leaks future commits into past training folds, yielding fake 0.95+ ROC-AUCs that crash in production.
3. **Why avoid SMOTE?** Software metrics have discrete physical AST constraints (non-negative integers, SLOC $\le$ LOC). SMOTE creates physically impossible phantom files that distort decision trees.
4. **Why Tarjan's SCC?** Circular imports violate the Acyclic Dependencies Principle (ADP) and cause cascading change failures. Tarjan detects all cycle sets in $O(V + E)$ linear time.

---

## 🔮 Future Engineering Roadmap

1. **Autonomous Agentic Auto-Refactoring (Auto-PR):**
   When a file exceeds Critical risk ($> 0.60$), an autonomous LLM agent branches the repo, applies Martin Fowler refactorings (extracting methods, breaking circular imports), executes the unit test suite to verify zero regression, and opens a pre-validated PR on GitHub.
2. **Relational Graph Neural Networks (R-GCN):**
   Learn topological graph embeddings directly over AST nodes and inter-module call graphs to capture complex architectural code smells (Shotgun Surgery, Feature Envy).
3. **Just-In-Time (JIT) Commit-Level Prediction:**
   Shift prediction into pre-merge GitHub Actions checking PR diffs in real-time to block risky commits before they merge into `main`.
4. **Production APM Telemetry Integration:**
   Ingest live runtime exception stack traces from Sentry/Datadog to correlate static risk scores with actual production crashes.
5. **Reinforcement Learning for Sprint Technical Debt Paydown:**
   Formulate technical debt reduction as a Knapsack optimization problem: given file remediation costs (LOC/complexity) and risk-reduction payoffs, output the mathematically optimal portfolio of files to refactor in Sprint $N$.

---

## 📚 Research Foundations & Citations

* **Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022).** *"Why do tree-based models still outperform deep learning on tabular data?"* Advances in Neural Information Processing Systems (NeurIPS 2022).
* **Nagappan, N., & Ball, T. (2005).** *"Use of relative code churn measures to predict system defect density."* Proceedings of the 27th International Conference on Software Engineering (ICSE 2005).
* **Hassan, A. E. (2009).** *"Predicting faults using the complexity of code changes."* Proceedings of the 31st International Conference on Software Engineering (ICSE 2009, Most Influential Paper Award).
* **Bird, C., Nagappan, N., Devanbu, P., Gall, H., & Murphy, B. (2011).** *"Don't touch my code! Examining the effects of ownership on software quality."* Proceedings of the 19th ACM SIGSOFT Symposium on the Foundations of Software Engineering (FSE 2011).
* **Kamei, Y., et al. (2013).** *"A large-scale empirical study of just-in-time quality assurance."* IEEE Transactions on Software Engineering (IEEE TSE).
* **Menzies, T., et al. (2010).** *"Defect prediction from static code features: current results, limitations, new directions."* Automated Software Engineering.
* **Lundberg, S. M., et al. (2020).** *"From local explanations to global understanding with explainable AI for trees."* Nature Machine Intelligence.
* **Fowler, M. (2018).** *"Refactoring: Improving the Design of Existing Code."* Addison-Wesley Professional.

---

<p align="center">
  <b>AI-Powered Code Complexity & Technical Debt Predictor</b><br/>
  <i>Engineered for Predictive Quality Assurance, Architectural Health & Actionable Engineering Management.</i>
</p>
