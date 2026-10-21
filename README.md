# Neurofeedback BCI Application

A Python/Flask platform for cognitive games, adaptive difficulty, and EEG-driven interaction. It combines a browser application for participants and experimenters with a hybrid motor-imagery (MI) / steady-state visual evoked potential (SSVEP) classification pipeline.

The project includes user accounts, game sessions, trial-level performance records, experimenter dashboards, and REST endpoints for browser and Unity clients. The research scripts cover EEG acquisition, preprocessing, model training, and evaluation.

## Features

- **Cognitive tasks:** reaction speed, pattern memory, Stroop, and N-back games.
- **EEG game interaction:** procedural maze navigation and a shooter game with classifier-driven commands.
- **Adaptive difficulty:** threshold, moving-average, Elo-inspired, rule-based, performance-zone, and EEG-enhanced strategies.
- **Experiment management:** participant registration, experimenter assignment, administrator controls, session history, and analytics.
- **Hybrid EEG classification:** filter-bank common spatial patterns (FBCSP) with SVM for left/right motor imagery, plus spectral features and an LDA gate for MI versus SSVEP.
- **Client integration:** authenticated JSON APIs and an embedded Unity WebGL build.

## Technology

| Component | Implementation |
| --- | --- |
| Web application | Python, Flask, Jinja, JavaScript, HTML/CSS |
| Persistence | SQLAlchemy, SQLite for development; PostgreSQL configuration available |
| Accounts | Flask-Login, Werkzeug password hashing |
| Schema tooling | Flask-Migrate, Alembic |
| Signal processing | NumPy, SciPy, MNE |
| Classification | scikit-learn, CSP, SVM, LDA |
| Live EEG transport | Lab Streaming Layer (LSL), pylsl |
| Additional client | Unity WebGL and Unity-related REST endpoints |

## Application structure

```text
app/
  models/       Users, games, sessions, trial logs, EEG-derived metrics
  routes/       Authentication, dashboards, games, administration, JSON API
  services/     Difficulty strategies, maze generation, hybrid classifier
  templates/    Browser interface
  static/       Styles and Unity WebGL assets
config/         Development, production, and testing configuration
Emotive/        Acquisition integration, preprocessing, training, evaluation scripts
models/fusion/  Locations for separately supplied trained models
migrations/     Alembic migration files
tests/          Existing maze and manual DDA check scripts
main.py         Development entry point
seed*.py        Demo users and game definitions
```

## Run the web application locally

Python 3.10 or newer is recommended because some acquisition code uses modern type annotations. The commands below target Windows PowerShell and Python 3.11. They describe the source setup; a full application startup has not been verified in the documentation editing environment.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

$env:FLASK_ENV = 'development'
$env:SECRET_KEY = .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_hex(32))"

.\.venv\Scripts\python.exe seed.py
.\.venv\Scripts\python.exe seed_maze_game.py
.\.venv\Scripts\python.exe seed_shooter_game.py
.\.venv\Scripts\python.exe -m flask --app main run --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. The application factory creates the development SQLite tables automatically. The seed scripts add example accounts and games. Seeded credentials are local demonstration defaults: `admin` / `admin123`, `exp_ali` / `exp123`, and `testuser` / `test123`. Change them before making an instance accessible to others.

`main.py` also supports `python main.py`, which enables debug mode and binds to all interfaces. The explicit Flask command above binds to localhost for local development.

Hardware-independent cognitive tasks and manually supplied maze commands can be explored separately from live EEG. EEG command endpoints require the model files and live stream described below.

## Live EEG and model setup

The integrated classifier expects **14 channels at 256 Hz**, a 512-sample window, and updates after 64 additional samples. Its default trial duration is five seconds. It listens for an LSL stream named `EmotivEEG`, with fallback discovery for an EEG stream or a 14-channel stream.

The public source copy excludes trained models and recordings. Supply compatible model artifacts under:

```text
models/fusion/
  final_models/P201/
    model_info.json
    csp_models.pkl
    selector.pkl
    svm.pkl
  gate_model/
    model_info.json
    scaler.pkl
    lda.pkl
```

Use models you trust: the loader deserializes Python pickle files. Model compatibility depends on the training environment; the application contains a compatibility adapter for older MNE CSP objects.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `FLASK_ENV` | `development` | Configuration selection |
| `SECRET_KEY` | Development fallback | Flask session signing key |
| `DATABASE_URL` | `sqlite:///neurofeedback.db` in development | SQLAlchemy connection URI |
| `FUSION_MODEL_DIR` | `models/fusion` | Root of model artifacts |
| `MI_SUBJECT_MODEL` | `P201` | MI model subdirectory |
| `LSL_STREAM_NAME` | `EmotivEEG` | Preferred LSL outlet name |
| `HYBRID_CLASSIFIER_TRIAL_TIME` | `5.0` | Trial duration in seconds |
| `HYBRID_CLASSIFIER_LOGGING` | `1` | Enable prediction CSV output |
| `HYBRID_CLASSIFIER_JSON` | `1` | Enable latest-prediction JSON output |

`Emotive/emotive_lsl_stream.py` bridges a compatible Emotiv device to LSL using `emotiv_acquire.py`. Hardware acquisition additionally needs the original CyKIT-compatible `eeg.py`, its supporting USB/crypto packages, and device drivers. Those third-party device components and bundled binaries are not included in this public source copy; obtain and configure them separately with their applicable licenses, or supply a compatible LSL outlet through another acquisition tool.

The training scripts `Emotive/train_final_fbcsp.py` and `Emotive/train_mi_ssvep_gate.py` expect local datasets such as `dataset/` and `ssvep_dataset_raw/` relative to the working directory. They produce artifacts in `final_models/` and `gate_model/`. Prepare compatible datasets and review the script constants before training. Copy the resulting compatible artifacts into the application model layout above. Other research scripts may require additional packages and local path adjustments; `requirements.txt` is not a complete, pinned research environment.

## EEG command flow

1. A compatible acquisition tool publishes EEG samples to LSL.
2. The classifier applies signal processing and CSP/SVM motor-imagery classification, together with the spectral LDA gate.
3. MI predictions select `LEFT` or `RIGHT`; an SSVEP gate decision maps to `UP`.
4. Authenticated maze/shooter requests apply the command, record the trial, and update session state.
5. Difficulty strategies use recorded performance and, where configured, EEG-derived inputs.

The integrated classifier exposes `attention`, `relaxation`, and `cognitive_load` fields derived from model confidence values. These are application heuristics, not independently validated measurements of those cognitive states. No clinical effectiveness or classifier-accuracy benchmark is claimed by this README.

## Selected API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/games` | List active games |
| GET | `/api/user/stats` | Participant statistics |
| GET | `/api/user/history` | Participant session history |
| POST | `/api/maze/session/start` | Start a maze session |
| POST | `/api/maze/command` | Submit a manually supplied maze action |
| POST | `/api/maze/eeg/command` | Run an EEG trial and apply the maze command |
| POST | `/api/shooter/eeg/command` | Run an EEG trial for the shooter |
| POST | `/api/unity/session/start` | Start a Unity session |
| POST | `/api/unity/trial/submit` | Submit a Unity trial |
| POST | `/api/unity/eeg/metrics` | Record Unity EEG-derived metrics |

Most session and user endpoints require a logged-in session cookie. See `app/routes/api.py` for payloads, validation, and the remaining endpoints.

## Existing checks

The original project contains executable check scripts rather than a comprehensive automated test suite. A dependency-light standalone maze check is:

```powershell
python tests/test_maze_standalone.py
```

`tests/test_maze_generator.py` exercises the application maze service after installing the web dependencies. `tests/test_dda_manual.py` imports `db` from `app`, while the current extension is defined in `app.extensions`; that script needs its import corrected before use and also requires seeded users/games. It creates session data and should be run only against a local test database.

## Deployment and source provenance

The current entry point is for development. A production deployment needs a production WSGI server, a separately configured secret and database, disabled debug mode, and deployment-specific configuration. PostgreSQL requires a compatible driver such as `psycopg2-binary`, which is commented out in the supplied requirements.

This source snapshot was prepared from the local `D:\Application` checkout of `Neurofeeback-BCI/Application`. It omits the original Git history, participant recordings, generated datasets, prediction logs, trained model assets, project proposal document, and bundled third-party device components. The original project directory is unchanged. No new license has been assigned; retain existing attributions and resolve third-party licensing before redistributing additional components.
