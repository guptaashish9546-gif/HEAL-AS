import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from database.database import get_connection, init_db

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
LATEX_DIR = BASE_DIR / "latex"
REPORT_DIR = BASE_DIR / "reports"

app = Flask(__name__)
app.secret_key = os.environ.get("HEAL_AS_SECRET_KEY", "change-this-development-secret")
app.config.update(
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
logging.basicConfig(level=logging.INFO)
init_db()


def now():
    return datetime.now()


def latex_escape(value):
    value = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"
    }
    return "".join(replacements.get(c, c) for c in value)


def load_model(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {filename}")
    return joblib.load(path)


def predict_model(model, features, feature_names=None):
    x = np.asarray([features], dtype=object)
    if feature_names is not None:
        import pandas as pd
        x = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(x)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            confidence = float(np.max(model.predict_proba(x)[0])) * 100
        except Exception:
            confidence = None
    try:
        result = "YES" if int(prediction) == 1 else "NO"
    except (TypeError, ValueError):
        result = str(prediction).upper()
    return result, confidence


def compile_latex(template_name, values, disease):
    template_path = LATEX_DIR / template_name

    if not template_path.exists():
        raise FileNotFoundError(
            f"LaTeX template not found: {template_path}"
        )

    text = template_path.read_text(encoding="utf-8")

    for key, value in values.items():
        text = text.replace(
            "{{" + key + "}}",
            latex_escape(value)
        )

    report_id = values["REPORT_ID"]

    out_dir = REPORT_DIR / disease.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    work_dir = out_dir / f".work_{uuid.uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=True)

    tex_path = work_dir / f"{disease.lower()}_report.tex"
    tex_path.write_text(text, encoding="utf-8")

    pdf_name = f"{disease.lower()}_report_{report_id}.pdf"
    pdf_path = out_dir / pdf_name

    compiler = shutil.which("pdflatex")

    if not compiler:
        raise RuntimeError(
            "pdflatex was not found on PATH."
        )

    logging.info("LaTeX compiler: %s", compiler)
    logging.info("LaTeX source: %s", tex_path)

    cmd = [
        compiler,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        tex_path.name,
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=180,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(
            "LaTeX compilation timed out after 180 seconds."
        )

    logging.info("pdflatex return code: %s", proc.returncode)
    logging.info("pdflatex output:\n%s", proc.stdout)

    if proc.stderr:
        logging.error("pdflatex stderr:\n%s", proc.stderr)

    produced = work_dir / f"{disease.lower()}_report.pdf"

    if proc.returncode != 0 or not produced.exists():
        # Keep the work directory temporarily so we can inspect the log.
        log_file = work_dir / f"{disease.lower()}_report.log"

        error_details = proc.stdout[-5000:]

        logging.error(
            "PDF generation failed.\n"
            "Work directory: %s\n"
            "Expected PDF: %s\n"
            "LaTeX output:\n%s",
            work_dir,
            produced,
            error_details,
        )

        raise RuntimeError(
            "LaTeX compilation failed. "
            f"Check the PowerShell output. Work directory: {work_dir}"
        )

    shutil.move(str(produced), str(pdf_path))

    shutil.rmtree(work_dir, ignore_errors=True)

    logging.info("PDF successfully generated: %s", pdf_path)

    return pdf_path


def require_login():
    return session.get("user_id") is not None


def render_model_error(exc):
    logging.exception("HEAL-AS model/report error")
    message = str(exc)
    known = {
        "LaTeX compiler is not installed or is not available on PATH.": message,
        "The PDF report could not be generated. Check your LaTeX installation.": message,
    }
    return known.get(message, "Analysis could not be completed. Please verify the entered values and model setup.")


@app.route("/")
def index():
    return render_template("index.html", user=session.get("user_name"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    data = request.form
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    confirm = data.get("confirm_password", "")
    age = data.get("age", "").strip()
    gender = data.get("gender", "").strip()

    if not name or not email or not password or not confirm or not age or not gender:
        return render_template("register.html", error="Please fill all required fields.", form=data)
    if len(name) < 2:
        return render_template("register.html", error="Please enter a valid name.", form=data)
    if password != confirm:
        return render_template("register.html", error="Passwords do not match.", form=data)
    if len(password) < 8:
        return render_template("register.html", error="Password must contain at least 8 characters.", form=data)
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return render_template("register.html", error="Enter a valid email address.", form=data)
    try:
        age_i = int(age)
        if age_i < 0 or age_i > 120:
            raise ValueError
    except ValueError:
        return render_template("register.html", error="Enter an age between 0 and 120.", form=data)
    if gender not in ("Female", "Male", "Other"):
        return render_template("register.html", error="Select a valid gender.", form=data)

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users(name,email,password_hash,age,gender,created_at) VALUES(?,?,?,?,?,?)",
            (name, email, generate_password_hash(password), age_i, gender, now().isoformat(timespec="seconds")),
        )
        conn.commit()
    except Exception as exc:
        if "UNIQUE" in str(exc).upper():
            return render_template("register.html", error="An account with this email already exists.", form=data)
        logging.exception("Registration error")
        return render_template("register.html", error="Registration failed. Please try again.", form=data)
    finally:
        conn.close()
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    if not user or not check_password_hash(user["password_hash"], password):
        return render_template("login.html", error="Invalid email or password.", email=email)
    session.clear()
    session["user_id"] = user["id"]
    session["user_name"] = user["name"]
    return redirect(url_for("dashboard"))


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    if not require_login():
        return redirect(url_for("login"))
    conn = get_connection()
    rows = conn.execute(
        "SELECT id,disease,prediction,created_at FROM predictions WHERE user_id=? ORDER BY id DESC",
        (session["user_id"],),
    ).fetchall()
    stats = conn.execute(
        "SELECT COUNT(*) total, SUM(CASE WHEN prediction='YES' THEN 1 ELSE 0 END) positive FROM predictions WHERE user_id=?",
        (session["user_id"],),
    ).fetchone()
    conn.close()
    return render_template("dashboard.html", user=session["user_name"], reports=rows, stats=stats)


@app.route("/diabetes")
def diabetes():
    if not require_login():
        return redirect(url_for("login"))
    return render_template("diabetes.html")


@app.route("/dengue")
def dengue():
    if not require_login():
        return redirect(url_for("login"))
    return render_template("dengue.html")


@app.route("/typhoid")
def typhoid():
    if not require_login():
        return redirect(url_for("login"))
    return render_template("typhoid.html", model_note=typhoid_model_status())


@app.route("/image-analysis")
def image_analysis():
    if not require_login():
        return redirect(url_for("login"))
    return render_template("image_analysis.html")


def typhoid_model_status():
    try:
        model = load_model("typhoid_model (1).pkl")
        if not hasattr(model, "estimators_"):
            return "The supplied Typhoid model is an unfitted RandomForestClassifier. Prediction is disabled until a trained/fitted model is supplied."
        return "Model loaded."
    except Exception:
        return "The supplied Typhoid model is unavailable."


def patient_common(form):
    return {
        "PATIENT_NAME": form.get("patient_name", "").strip() or session["user_name"],
        "PATIENT_ID": f"USER-{session['user_id']}",
        "CONTACT": form.get("contact", "").strip() or "Not provided",
        "ADDRESS": form.get("address", "").strip() or "Not provided",
    }


def validate_nonnegative(*values):
    return all(v >= 0 for v in values)


@app.post("/api/predict/diabetes")
def api_diabetes():
    if not require_login():
        return jsonify(success=False, error="Authentication required."), 401
    try:
        age = int(request.form["age"])
        bmi = float(request.form["bmi"])
        glucose = float(request.form["blood_glucose_level"])
        gender = request.form["gender"]
        if not (0 <= age <= 120) or bmi <= 0 or glucose < 0 or gender not in ("Female", "Male", "Other"):
            raise ValueError

        gender_male = gender == "Male"
        gender_other = gender == "Other"
        features = [age, bmi, glucose, gender_male, gender_other]
        model = load_model("diabeties_model (2).pkl")
        prediction, confidence = predict_model(model, features, ["age", "bmi", "blood_glucose_level", "gender_Male", "gender_Other"])
        report_id = f"{now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6].upper()}"
        values = {
            "REPORT_ID": report_id, "REPORT_DATE": now().strftime("%d/%m/%Y"),
            "EVALUATION_DATETIME": now().strftime("%d/%m/%Y %H:%M"), "AGE": age, "SEX": gender,
            "CONFIDENCE": f"{confidence:.2f}%" if confidence is not None else "N/A", "PREDICTION": prediction,
            "BMI": bmi, "BLOOD_GLUCOSE": glucose, "GENDER_MALE": gender_male, "GENDER_OTHER": gender_other,
            **patient_common(request.form),
        }
        pdf = compile_latex("diabeties_report.tex", values, "diabeties")
        conn = get_connection()
        cur = conn.execute(
            "INSERT INTO predictions(user_id,disease,prediction,input_data,created_at,report_path) VALUES(?,?,?,?,?,?)",
            (session["user_id"], "Diabetes", prediction, json.dumps(features), now().isoformat(timespec="seconds"), str(pdf.relative_to(BASE_DIR))),
        )
        report_db_id = cur.lastrowid
        conn.commit(); conn.close()
        return jsonify(success=True, disease="Diabetes", prediction=prediction, confidence=confidence, report_id=report_db_id, download_url=url_for("download_report", report_id=report_db_id))
    except (KeyError, ValueError, TypeError):
        return jsonify(success=False, error="Please enter valid values in all required fields."), 400
    except Exception as exc:
        return jsonify(success=False, error=render_model_error(exc)), 400


@app.post("/api/predict/dengue")
def api_dengue():
    if not require_login():
        return jsonify(success=False, error="Authentication required."), 401
    try:
        age = int(request.form["age"])
        hb = float(request.form["hemoglobin"])
        wbc = float(request.form["wbc"])
        rbc = float(request.form["rbc_panel"])
        platelets = float(request.form["platelet_count"])
        sex = request.form["sex"]
        if not (0 <= age <= 120) or not validate_nonnegative(hb, wbc, rbc, platelets) or sex not in ("Female", "Male"):
            raise ValueError

        female = sex == "Female"; male = sex == "Male"
        features = [age, hb, wbc, rbc, platelets, female, male]
        model = load_model("dengue_model.pkl")
        prediction, confidence = predict_model(model, features, ["Age", "Haemoglobin", "WBC Count", "RBC PANEL", "Platelet Count", "Sex_Female", "Sex_Male"])
        report_id = f"{now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6].upper()}"
        values = {
            "REPORT_ID": report_id, "REPORT_DATE": now().strftime("%d/%m/%Y"),
            "EVALUATION_DATETIME": now().strftime("%d/%m/%Y %H:%M"), "AGE": age, "SEX": sex,
            "CONFIDENCE": f"{confidence:.2f}%" if confidence is not None else "N/A", "PREDICTION": prediction,
            "D_AGE": age, "D_HB": hb, "D_WBC": wbc, "D_RBC": rbc, "D_PLATELET": platelets,
            "D_SEX_FEMALE": female, "D_SEX_MALE": male, **patient_common(request.form),
        }
        pdf = compile_latex("dengue_report.tex", values, "dengue")
        conn = get_connection()
        cur = conn.execute(
            "INSERT INTO predictions(user_id,disease,prediction,input_data,created_at,report_path) VALUES(?,?,?,?,?,?)",
            (session["user_id"], "Dengue", prediction, json.dumps(features), now().isoformat(timespec="seconds"), str(pdf.relative_to(BASE_DIR))),
        )
        report_db_id = cur.lastrowid
        conn.commit(); conn.close()
        return jsonify(success=True, disease="Dengue", prediction=prediction, confidence=confidence, report_id=report_db_id, download_url=url_for("download_report", report_id=report_db_id))
    except (KeyError, ValueError, TypeError):
        return jsonify(success=False, error="Please enter valid values in all required fields."), 400
    except Exception as exc:
        return jsonify(success=False, error=render_model_error(exc)), 400


@app.post("/api/predict/typhoid")
def api_typhoid():
    if not require_login(): return jsonify(success=False,error="Authentication required."),401
    try:
        data=request.get_json(silent=True) or {}
        req=["age","fever_duration_days","white_blood_cell_count","platelet_count","gender","water_source","gastrointestinal_symptoms","neurological_symptoms","skin_manifestations","blood_culture_result","widal_test","typhidot_test","typhoid_vaccination_status","previous_history_of_typhoid","weather_condition","ongoing_infection_in_society"]
        if any(data.get(k) in (None,"") for k in req): raise ValueError("Please complete all required Typhoid fields.")
        age=float(data["age"]);fever=float(data["fever_duration_days"]);wbc=float(data["white_blood_cell_count"]);platelets=float(data["platelet_count"])
        if min(age,fever,wbc,platelets)<0: raise ValueError("Numeric values cannot be negative.")
        cols=["Age","Fever Duration (Days)","White Blood Cell Count","Platelet Count","Gender_Male","Water Source Type_Tap","Water Source Type_Untreated Supply","Water Source Type_Well","Gastrointestinal Symptoms_Constipation","Gastrointestinal Symptoms_Diarrhea","Neurological Symptoms_Delirium","Neurological Symptoms_Headache","Skin Manifestations_Yes","Blood Culture Result_Positive","Widal Test_Low O & H Antibody","Typhidot Test_IgM Positive","Typhidot Test_Negative","Typhoid Vaccination Status_Received","Previous History of Typhoid_Yes","Weather Condition_Hot & Dry","Weather Condition_Moderate","Weather Condition_Rainy & Wet","Ongoing Infection in Society_Dengue Outbreak","Ongoing Infection in Society_Seasonal Flu"]
        row={c:0 for c in cols};row.update({"Age":age,"Fever Duration (Days)":fever,"White Blood Cell Count":wbc,"Platelet Count":platelets})
        for c,v in [("Gender_Male",data["gender"]=="Male"),("Water Source Type_Tap",data["water_source"]=="Tap"),("Water Source Type_Untreated Supply",data["water_source"]=="Untreated Supply"),("Water Source Type_Well",data["water_source"]=="Well"),("Gastrointestinal Symptoms_Constipation",data["gastrointestinal_symptoms"]=="Constipation"),("Gastrointestinal Symptoms_Diarrhea",data["gastrointestinal_symptoms"]=="Diarrhea"),("Neurological Symptoms_Delirium",data["neurological_symptoms"]=="Delirium"),("Neurological Symptoms_Headache",data["neurological_symptoms"]=="Headache"),("Skin Manifestations_Yes",data["skin_manifestations"]=="Yes"),("Blood Culture Result_Positive",data["blood_culture_result"]=="Positive"),("Widal Test_Low O & H Antibody",data["widal_test"]=="Low O & H Antibody"),("Typhidot Test_IgM Positive",data["typhidot_test"]=="IgM Positive"),("Typhidot Test_Negative",data["typhidot_test"]=="Negative"),("Typhoid Vaccination Status_Received",data["typhoid_vaccination_status"]=="Received"),("Previous History of Typhoid_Yes",data["previous_history_of_typhoid"]=="Yes"),("Weather Condition_Hot & Dry",data["weather_condition"]=="Hot & Dry"),("Weather Condition_Moderate",data["weather_condition"]=="Moderate"),("Weather Condition_Rainy & Wet",data["weather_condition"]=="Rainy & Wet"),("Ongoing Infection in Society_Dengue Outbreak",data["ongoing_infection_in_society"]=="Dengue Outbreak"),("Ongoing Infection in Society_Seasonal Flu",data["ongoing_infection_in_society"]=="Seasonal Flu")]: row[c]=int(v)
        model=load_model("typhoid_model (1).pkl");X=pd.DataFrame([[row[c] for c in cols]],columns=cols);pred=model.predict(X)[0]
        prediction="YES" if str(pred).strip().lower() in {"1","yes","positive","true","typhoid","detected"} else "NO"
        confidence=float(max(model.predict_proba(X)[0]))*100 if hasattr(model,"predict_proba") else None
        rid=uuid.uuid4().hex[:10].upper(); values={"REPORT_ID":rid,"REPORT_DATE":now().strftime("%d/%m/%Y"),"PATIENT_NAME":str(data.get("patient_name") or "Not provided"),"PATIENT_ID":str(session["user_id"]),"AGE":str(int(age) if age.is_integer() else age),"SEX":str(data["gender"]),"CONTACT":"Not provided","EVALUATION_DATETIME":now().strftime("%d/%m/%Y %H:%M"),"PREDICTION":prediction,"CONFIDENCE":f"{confidence:.2f}%" if confidence is not None else "N/A"}
        for i,c in enumerate(cols,1): values[f"T_INPUT_{i}"]=str(row[c])
        pdf=compile_latex("typhoid_report.tex",values,"typhoid");conn=get_connection();cur=conn.execute("INSERT INTO predictions(user_id,disease,prediction,input_data,created_at,report_path) VALUES(?,?,?,?,?,?)",(session["user_id"],"Typhoid",prediction,json.dumps(row),now().isoformat(timespec="seconds"),str(pdf.relative_to(BASE_DIR))));dbid=cur.lastrowid;conn.commit();conn.close()
        return jsonify(success=True,disease="Typhoid",prediction=prediction,confidence=confidence,report_id=dbid,download_url=url_for("download_report",report_id=dbid),redirect=url_for("dashboard"))
    except (KeyError,ValueError,TypeError): return jsonify(success=False,error="Please enter valid values in all required Typhoid fields."),400
    except Exception as exc: return jsonify(success=False,error=render_model_error(exc)),400


@app.get("/api/history")
def api_history():
    if not require_login():
        return jsonify(success=False, error="Authentication required."), 401
    conn = get_connection()
    rows = conn.execute("SELECT id,disease,prediction,input_data,created_at FROM predictions WHERE user_id=? ORDER BY id DESC", (session["user_id"],)).fetchall()
    conn.close()
    return jsonify(success=True, history=[dict(r) for r in rows])


@app.get("/api/report/<int:report_id>")
def download_report(report_id):
    if not require_login():
        return jsonify(success=False, error="Authentication required."), 401
    conn = get_connection()
    row = conn.execute("SELECT report_path FROM predictions WHERE id=? AND user_id=?", (report_id, session["user_id"])).fetchone()
    conn.close()
    if not row:
        return jsonify(success=False, error="Report not found."), 404
    path = BASE_DIR / row["report_path"]
    if not path.is_file() or BASE_DIR not in path.resolve().parents:
        return jsonify(success=False, error="Report unavailable."), 404
    return send_file(path, as_attachment=True, download_name=path.name)


@app.get("/api/health")
def health():
    return jsonify(status="ok", application="HEAL-AS")


@app.errorhandler(413)
def too_large(_):
    return jsonify(success=False, error="Uploaded file is too large. Maximum size is 5 MB."), 413


if __name__ == "__main__":
    app.run(debug=os.environ.get("HEAL_AS_DEBUG", "0") == "1")
