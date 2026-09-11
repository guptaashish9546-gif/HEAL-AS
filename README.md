# HEAL-AS — Healthcare Evaluation and Analysis Autonomous System

A college-level full-stack AI/ML healthcare screening project built with Flask, SQLite, HTML/CSS/JavaScript and the supplied ML models.

## What's improved

- Modern responsive healthcare/AI interface
- Mobile navigation menu
- Clear active/coming-soon module states
- Better form layout, labels, units and validation
- Loading state while prediction/PDF generation runs
- Result badges and model-confidence progress bar
- Dashboard statistics and cleaner report history
- Friendlier backend errors instead of raw Python stack traces
- Secure session-cookie settings
- Health-check endpoint at `/api/health`
- Existing LaTeX report templates remain the report source
- PDF confidence value is now inserted dynamically

## Project structure

```text
HEAL-AS/
├── app.py
├── requirements.txt
├── README.md
├── heal_as.db                 # created automatically on first run
├── models/
│   ├── diabeties_model (2).pkl
│   ├── dengue_model.pkl
│   ├── typhoid_model (1).pkl
│   └── Typhoid_dataset(1).csv
├── templates/
├── static/
│   ├── css/style.css
│   └── js/script.js
├── latex/
└── reports/
    ├── diabetes/
    ├── dengue/
    └── typhoid/
```

## 1. Open the correct folder

Open the folder that directly contains `app.py` in VS Code.

If extraction produced `Downloads/HEAL-AS/HEAL-AS/`, open the **inner** `HEAL-AS` folder.

## 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

The terminal should begin with `(venv)`.

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## 4. Start the website

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Main pages

- `/` — Home
- `/register` — Registration
- `/login` — Login
- `/dashboard` — Dashboard
- `/diabetes` — Diabetes analysis
- `/dengue` — Dengue analysis
- `/typhoid` — Typhoid model status
- `/image-analysis` — Future X-ray analysis interface

## PDF reports

PDF generation uses the supplied LaTeX templates. Install MiKTeX or TeX Live and make sure `pdflatex` is available in PATH:

```powershell
pdflatex --version
```

Then run an active Diabetes or Dengue analysis. Generated PDFs are stored in the disease-specific `reports/` directory and are available through the authenticated dashboard.

## Supplied model notes

The Diabetes model is wired to exactly five features:

```text
age
bmi
blood_glucose_level
gender_Male
gender_Other
```

The Dengue model is wired to exactly seven features:

```text
Age
Haemoglobin
WBC Count
RBC PANEL
Platelet Count
Sex_Female
Sex_Male
```

The supplied Typhoid pickle is unfitted. HEAL-AS therefore refuses to guess its feature order or preprocessing and keeps Typhoid prediction disabled until a trained/fitted model is supplied.

## Medical safety

HEAL-AS is an educational/AI-assisted screening project. Model output is a preliminary prediction and is not a confirmed medical diagnosis. A positive result should be discussed with a qualified healthcare professional; a negative result does not guarantee absence of disease.

## Production note

Before public deployment, replace the development secret with a strong `HEAL_AS_SECRET_KEY` environment variable, enable HTTPS, add CSRF protection, and use a production WSGI server.

## Typhoid model replacement

The Typhoid page is wired to the 24 predictor columns supplied for this project. `Typhoid Status` is the target and is not collected from the user. Replace `models/typhoid_model (1).pkl` with your trained/fitted model when ready.
