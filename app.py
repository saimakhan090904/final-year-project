from flask import Flask, render_template, url_for, request, redirect, session
import os
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__, static_folder='static')
# Enable template auto-reload in development to ensure UI changes show up immediately
app.config['TEMPLATES_AUTO_RELOAD'] = True
try:
    app.jinja_env.auto_reload = True
except Exception:
    pass
# Secret key for session storage (used to persist last BMI/weight/height for prefilling)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
# from diet_recommender import (
#     load_dataset,
#     recommend_recipes,
#     to_recipe_output,
#     build_nutrition_target,
#     build_exclude_list_for_diet_type,
#     filter_restricted_foods,
# )
from risk_explainer import (
    compute_risk_contributions,
    contributions_to_data_uri,
    get_recommended_targets,
    comparison_chart_to_data_uri,
)
# from personalization import personalize_diet, compute_daily_targets
from inference_scaler import get_minmax_scaler, FEATURE_ORDER
# from medical_filters import RESTRICTION_CATALOG
_recipes_df = None


def _age_band(age: int) -> str:
    if age is None:
        return '40-59'
    if age <= 39:
        return '20-39'
    if age <= 59:
        return '40-59'
    return '60+'


def _bp_ranges_for_age(age: int):
    band = _age_band(age)
    if band == '20-39':
        return (90, 120), (60, 80)
    if band == '40-59':
        return (90, 130), (60, 85)
    return (90, 140), (60, 90)


def _status_from_range(value: float, low: float, high: float):
    if value < low:
        return 'Low', 'danger'
    if value > high:
        return 'High', 'danger'
    # Within band
    # Treat close-to-edges as borderline? Keep simple: normal inside
    return 'Normal', 'success'


def _safe_load_model(path: str):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _heuristic_probability(*, sysBP: float, diaBP: float, glucose: float, age: int,
                           totChol: float, prevalentHyp: int, diabetes: int,
                           BPMeds: int, bmi: float) -> float:
    # Use existing explainer scores (0..1) and map to a pseudo-probability.
    scores = [s for _, s in compute_risk_contributions({
        'sysBP': sysBP,
        'diaBP': diaBP,
        'glucose': glucose,
        'age': age,
        'totChol': totChol,
        'prevalentHyp': prevalentHyp,
        'diabetes': diabetes,
        'BPMeds': BPMeds,
        'bmi': bmi,
    })]
    if not scores:
        return 0.2
    # Average of top 6 contributors, softly scaled into [0.05, 0.95]
    scores.sort(reverse=True)
    top = scores[:6]
    raw = sum(top) / max(1, len(top))
    proba = 0.05 + 0.9 * min(1.0, max(0.0, raw))
    return float(proba)




def build_feature_rows(*, age: int, sysBP: float, diaBP: float, glucose: float, bmi: float,
                       totChol: float, prevalentHyp: int, gender: str, BPMeds: int, diabetes: int):
    rows = []

    # BP ranges depend on age band
    (sys_lo, sys_hi), (dia_lo, dia_hi) = _bp_ranges_for_age(age)
    sys_status, sys_cls = _status_from_range(sysBP, sys_lo, sys_hi)
    dia_status, dia_cls = _status_from_range(diaBP, dia_lo, dia_hi)
    rows.append({'feature': 'Systolic BP', 'range': f'{sys_lo}-{sys_hi} mmHg', 'value': sysBP, 'status': sys_status, 'cls': sys_cls})
    rows.append({'feature': 'Diastolic BP', 'range': f'{dia_lo}-{dia_hi} mmHg', 'value': diaBP, 'status': dia_status, 'cls': dia_cls})

    # Glucose
    if glucose < 70:
        g_status, g_cls = 'Low', 'danger'
        g_range = '70–99 mg/dL normal'
    elif glucose < 100:
        g_status, g_cls = 'Normal', 'success'
        g_range = '70–99 mg/dL normal'
    elif glucose < 126:
        g_status, g_cls = 'Borderline', 'warning'
        g_range = '100–125 prediabetic'
    else:
        g_status, g_cls = 'High', 'danger'
        g_range = '126+ diabetic'
    rows.append({'feature': 'Glucose', 'range': g_range, 'value': glucose, 'status': g_status, 'cls': g_cls})

    # Age (display only)
    rows.append({'feature': 'Age', 'range': '-', 'value': age, 'status': '-', 'cls': ''})

    # BMI
    if bmi < 18.5:
        b_status, b_cls = 'Low', 'danger'
        b_range = '18.5–24.9 normal'
    elif bmi < 25:
        b_status, b_cls = 'Normal', 'success'
        b_range = '18.5–24.9 normal'
    elif bmi < 30:
        b_status, b_cls = 'Borderline', 'warning'
        b_range = '25–29.9 overweight'
    else:
        b_status, b_cls = 'High', 'danger'
        b_range = '30+ obese'
    rows.append({'feature': 'BMI', 'range': b_range, 'value': round(bmi, 2), 'status': b_status, 'cls': b_cls})

    # Total cholesterol
    if totChol < 200:
        c_status, c_cls = 'Normal', 'success'
        c_range = '<200 desirable'
    elif totChol < 240:
        c_status, c_cls = 'Borderline', 'warning'
        c_range = '200–239 borderline'
    else:
        c_status, c_cls = 'High', 'danger'
        c_range = '240+ high'
    rows.append({'feature': 'Total Cholesterol', 'range': c_range, 'value': totChol, 'status': c_status, 'cls': c_cls})

    # Hypertension
    hyp_status = 'Normal' if prevalentHyp == 0 else 'Abnormal'
    hyp_cls = 'success' if prevalentHyp == 0 else 'danger'
    rows.append({'feature': 'Hypertension', 'range': 'No is normal', 'value': 'Yes' if prevalentHyp == 1 else 'No', 'status': hyp_status, 'cls': hyp_cls})

    # Gender
    rows.append({'feature': 'Gender', 'range': '-', 'value': gender.title() if isinstance(gender, str) else gender, 'status': '-', 'cls': ''})

    # Diabetes
    diab_status = 'Normal' if diabetes == 0 else 'Abnormal'
    diab_cls = 'success' if diabetes == 0 else 'danger'
    rows.append({'feature': 'Diabetes', 'range': 'No is normal', 'value': 'Yes' if diabetes == 1 else 'No', 'status': diab_status, 'cls': diab_cls})

    # BP Medication depends on BP
    bp_high = (sys_status == 'High') or (dia_status == 'High')
    if BPMeds == 0 and not bp_high:
        bpmed_status, bpmed_cls = 'Normal', 'success'
    elif BPMeds == 1 and bp_high:
        bpmed_status, bpmed_cls = 'Normal', 'success'
    else:
        bpmed_status, bpmed_cls = 'Borderline', 'warning'
    rows.append({'feature': 'BP Medication', 'range': 'No normal unless high BP', 'value': 'Yes' if BPMeds == 1 else 'No', 'status': bpmed_status, 'cls': bpmed_cls})

    return rows


def _high_risk_flags(*, age: int, sysBP: float, diaBP: float, glucose: float, bmi: float, totChol: float) -> tuple[list[str], int]:
    """Return (list of high-risk factor names, count) using medical cutoffs.

    - BP: age-banded cutoffs
    - Glucose: >= 126 mg/dL
    - BMI: >= 30
    - Total Cholesterol: >= 240
    """
    highs: list[str] = []
    (sys_lo, sys_hi), (dia_lo, dia_hi) = _bp_ranges_for_age(age)
    if sysBP > sys_hi:
        highs.append('Systolic BP')
    if diaBP > dia_hi:
        highs.append('Diastolic BP')
    if glucose >= 126:
        highs.append('Glucose')
    if bmi >= 30:
        highs.append('BMI')
    if totChol >= 240:
        highs.append('Cholesterol')
    return highs, len(highs)


@app.route("/")
def index():
    return render_template('heart.html')
@app.route("/home")
def home():
    # Prefill from last BMI calculation if available
    return render_template(
        'home.html',
        prefill_bmi=session.get('last_bmi'),
        prefill_age=session.get('last_age'),
        prefill_gender=session.get('last_gender'),
    )
@app.route("/stress")
def stress():
    return render_template('stress.html')
@app.route("/checkup")
def checkup():
    return render_template('checkup.html')
@app.route("/fitness")
def fitness():
    return render_template('fitness.html')
# @app.route("/diet")
# def diet():
#     return render_template('diet.html')
@app.route("/sleep")
def sleep():
    return render_template('sleep.html')

@app.route("/analysis")
def analysis():
    return render_template('analysis.html')
@app.route("/bmi")
def bmi():
    return render_template('bmi.html')

@app.route('/results', methods=['POST','GET'])
def results():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    weight = float(request.form['Weight'])
    height = float(request.form['Height'])


    bmi= weight / ((height / 100) ** 2)

    # Persist latest values in session so other pages can auto-fill
    try:
        session['last_bmi'] = round(bmi, 2)
        session['last_weight'] = weight
        session['last_height'] = height
        session['last_age'] = age
        session['last_gender'] = 'male' if sex == 0 else 'female'
    except Exception:
        # Session might be unavailable in some environments; ignore failures
        pass


    if (bmi < 18.5):
        return render_template('Underweight.html',bmi=bmi, weight=weight, height=height, age=age, sex=sex)
    elif (bmi >= 18.5 and bmi < 24.5):
        return render_template('Normal.html',bmi=bmi, weight=weight, height=height, age=age, sex=sex)
    elif (bmi >= 24.5 ):
        return render_template('Overweight.html',bmi=bmi, weight=weight, height=height, age=age, sex=sex)

@app.route('/result', methods=['POST', 'GET'])

def result():
    sysBP = float(request.form['sysBP'])
    glucose = float(request.form['glucose'])
    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    totCHol = float(request.form['totChol'])
    diaBP = float(request.form['diaBP'])
    prevalentHyp = int(request.form['prevalentHyp'])
    male = int(request.form['male'])
    # Map UI value (0=Male,1=Female) to dataset semantics (male: 1=Male, 0=Female)
    male_model = 1 if male == 0 else 0
    BPMeds = int(request.form['BPMeds'])
    diabetes = int(request.form['diabetes'])
    # Build feature frame in canonical order
    x = pd.DataFrame([[sysBP, glucose, age, totCHol, diaBP, prevalentHyp, diabetes, male_model, BPMeds, bmi]],
                     columns=FEATURE_ORDER, dtype=float)

    # x = np.array([sysbp, glucose, age, cigsperday, totchol, diabp, prevalentHyp,
    #               male, bpmeds,diabetes])
    print(x)
    model = _safe_load_model('pickle files/randomf.pkl')

    # mp = None
    # scaler_path = os.path.join(os.path.dirname(__file__), 'randomf')
    # with open('rf_pickle', 'rb') as g:
    #     mp = pickle.load(g)

    # Ensure inference-time scaling consistent with training
    try:
        scaler, _ = get_minmax_scaler()
        x = pd.DataFrame(scaler.transform(x.values), columns=FEATURE_ORDER)
    except Exception:
        pass

   # model_path = os.path.join(os.path.dirname(__file__), 'models/rfc.sav')
    #clf = jb.load(model_path)

    # Probabilistic prediction; lower decision threshold to increase sensitivity
    proba = None
    if model is not None:
        try:
            proba = float(model.predict_proba(x)[0][1])
        except Exception:
            try:
                proba = float(model.predict(x)[0])
            except Exception:
                proba = None
    if proba is None:
        # Fallback to heuristic probability when the pickled model is incompatible
        proba = _heuristic_probability(
            sysBP=sysBP, diaBP=diaBP, glucose=glucose, age=age, totChol=totCHol,
            prevalentHyp=prevalentHyp, diabetes=diabetes, BPMeds=BPMeds, bmi=bmi,
        )
    # Load tuned threshold if available
    threshold = 0.3
    try:
        import json
        from pathlib import Path
        tpath = Path('models') / 'threshold.json'
        if tpath.exists():
            threshold = float(json.loads(tpath.read_text()).get('threshold', threshold))
    except Exception:
        pass
    y = 1 if proba >= threshold else 0
    print({'proba': proba, 'threshold': threshold, 'y': y})

  # No heart disease
    # Safety net override: if 3+ features are High, force Risk
    highs, num_high = _high_risk_flags(age=age, sysBP=sysBP, diaBP=diaBP, glucose=glucose, bmi=bmi, totChol=totCHol)
    if num_high >= 3:
        y = 1

    if y == 0:
        if male == 0:
            gender = 'male'
        else:
            gender = 'female'
        if prevalentHyp == 0:
            p = 'No'
        else:
            p = 'Yes'
        if diabetes == 0:
            d = 'No'
        else:
            d = 'Yes'
        if BPMeds == 0:
            b = 'No'
        else:
            b = 'Yes'
        contrib = compute_risk_contributions({
            'sysBP': sysBP,
            'diaBP': diaBP,
            'glucose': glucose,
            'age': age,
            'totChol': totCHol,
            'prevalentHyp': prevalentHyp,
            'diabetes': diabetes,
            'BPMeds': BPMeds,
            'bmi': bmi,
        })
        # Improved comparison chart: current vs recommended
        targets = get_recommended_targets(age=age, weight_kg=None, bmi=bmi, has_diabetes=diabetes, has_hypertension=prevalentHyp, total_chol=totCHol)
        feature_rows = build_feature_rows(age=age, sysBP=sysBP, diaBP=diaBP, glucose=glucose, bmi=bmi,
                                          totChol=totCHol, prevalentHyp=prevalentHyp, gender=gender,
                                          BPMeds=BPMeds, diabetes=diabetes)
        return render_template('nodisease.html',age=age,gender=gender,bmi=bmi,sysBP=sysBP,diaBP=diaBP,glucose=glucose,
                             totCHol=totCHol,p=p,d=d,b=b, feature_rows=feature_rows)
    else:
    #heart disease
      if male==0:
          gender='male'
      else:
          gender='female'
      if prevalentHyp == 0:
          p='No'
      else:
          p='Yes'
      if diabetes==0:
          d='No'
      else:
          d='Yes'
      if BPMeds==0:
          b='No'
      else:
          b = 'Yes'
      contrib = compute_risk_contributions({
          'sysBP': sysBP,
          'diaBP': diaBP,
          'glucose': glucose,
          'age': age,
          'totChol': totCHol,
          'prevalentHyp': prevalentHyp,
          'diabetes': diabetes,
          'BPMeds': BPMeds,
          'bmi': bmi,
      })
      targets = get_recommended_targets(age=age, weight_kg=None, bmi=bmi, has_diabetes=diabetes, has_hypertension=prevalentHyp, total_chol=totCHol)
      feature_rows = build_feature_rows(age=age, sysBP=sysBP, diaBP=diaBP, glucose=glucose, bmi=bmi,
                                        totChol=totCHol, prevalentHyp=prevalentHyp, gender=gender,
                                        BPMeds=BPMeds, diabetes=diabetes)
      # Show confirmation page before diet preferences
      return render_template('heartdisease_detected.html', age=age, gender=gender, bmi=bmi, sysBP=sysBP, diaBP=diaBP, glucose=glucose,
                             totCHol=totCHol, p=p, d=d, b=b, feature_rows=feature_rows, high_flags=highs, probability=proba, threshold=threshold)


# @app.route('/diet/preferences', methods=['GET'])
# def diet_preferences():
#     # Prefill the preferences form from query params (used by both positive and preventive flows)
#     age = request.args.get('age', type=int) or session.get('last_age')
#     gender = request.args.get('gender', type=str) or session.get('last_gender')
#     bmi = request.args.get('bmi', type=float) or session.get('last_bmi')
#     weight = request.args.get('weight', type=float) or session.get('last_weight')
#     height = request.args.get('height', type=float) or session.get('last_height')
#     totCHol = request.args.get('totChol', type=float)
#     p = request.args.get('p', default='No')  # Yes/No
#     d = request.args.get('d', default='No')  # Yes/No
# 
#     return render_template(
#         'diet_preferences.html',
#         age=age, gender=gender, bmi=bmi, weight=weight, height=height,
#         totCHol=totCHol, p=p, d=d
#     )


# @app.route('/diet/recommend', methods=['POST'])
# def diet_recommendations():
#     global _recipes_df
#     if _recipes_df is None:
#         # Try to load the dataset from the diet project
#         preferred = [
#             os.path.join('Diet-Recommendation-System-main', 'Data', 'dataset.csv'),
#             os.path.join('..', 'Diet-Recommendation-System-main', 'Data', 'dataset.csv'),
#             os.path.join('Data', 'dataset.csv'),
#         ]
#         _recipes_df = load_dataset(preferred_paths=preferred)
# 
#     diet_type = request.form.get('diet_type', 'veg')
#     alcohol = int(request.form.get('alcohol', '0'))
#     weight_goal = request.form.get('weight_goal', 'maintain')
#     activity_level = request.form.get('activity_level', 'moderate')
#     smoking = request.form.get('smoking', '0')
#     sleep_quality = request.form.get('sleep_quality', 'good')
# 
#     # Health info
#     bmi = float(request.form.get('bmi'))
#     tot_chol = float(request.form.get('totChol'))
#     diabetes = int(request.form.get('diabetes', '0'))
#     hypertension = int(request.form.get('hypertension', '0'))
# 
#     # Compute daily targets when inputs are provided
#     weight_val = float(request.form.get('weight', request.args.get('weight', 0)) or 0)
#     height_val = float(request.form.get('height', request.args.get('height', 0)) or 0)
#     age_val = int(request.form.get('age', request.args.get('age', 0)) or 0)
#     gender_val = request.form.get('gender', request.args.get('gender', ''))
#     cal_target, prot_target = compute_daily_targets(
#         weight_kg=weight_val or None,
#         height_cm=height_val or None,
#         age=age_val or None,
#         gender=gender_val or None,
#         activity_level=activity_level,
#         weight_goal=weight_goal,
#     )
# 
#     # Build nutrition target from health context, using per-meal targets derived from daily goals when available
#     nutrition_target = build_nutrition_target(
#         bmi=bmi, tot_chol=tot_chol, diabetes=diabetes, hypertension=hypertension, alcohol=alcohol,
#         protein_target_g=prot_target,
#         calorie_target_kcal=cal_target
#     )
# 
#     # Build include/exclude filters
#     allergies_raw = request.form.get('allergies', '')
#     allergies = [a.strip() for a in allergies_raw.split(',') if a.strip()]
#     exclude_by_diet = build_exclude_list_for_diet_type(diet_type)
#     exclude_all = list(set([*allergies, *exclude_by_diet]))
# 
#     # Recommend
#     recs_df = recommend_recipes(
#         dataset=_recipes_df,
#         nutrition_input=nutrition_target,
#         include_ingredients=[],
#         exclude_ingredients=exclude_all,
#         params={'n_neighbors': 10, 'return_distance': False},
#     )
#     recipes = to_recipe_output(recs_df) or []
# 
#     # Build user condition flags for filtering
#     user_conditions: list[str] = []
#     if diabetes == 1:
#         user_conditions.append('diabetes')
#     if hypertension == 1:
#         user_conditions.append('hypertension')
#     if tot_chol is not None and float(tot_chol) > 200.0:
#         user_conditions.append('cholesterol')
# 
#     # Apply medical-condition-based filtering to recipes (case-insensitive)
#     recipes = filter_restricted_foods(user_conditions, recipes)
# 
#     # Build prediction_output context for risk detection
#     prediction_output = {
#         'bmi': bmi,
#         'totChol': tot_chol,
#         'diabetes': diabetes,
#     }
# 
#     # If your ML pipeline computes a risk string, pass it via query/form as 'risk_reason'
#     risk_reason_override = request.form.get('risk_reason', request.args.get('risk_reason'))
# 
#     _personalized = personalize_diet(
#         base_diet=recipes,
#         prediction_output=prediction_output,
#         weight_goal=weight_goal,
#         activity_level=activity_level,
#         alcohol=alcohol,
#         smoking=smoking,
#         sleep_quality=sleep_quality,
#         weight_kg=weight_val,
#         height_cm=height_val,
#         age=age_val,
#         gender=gender_val,
#         risk_reason_override=risk_reason_override,
#     )
#     # Backward compatibility: support both 3- and 5-value returns
#     if isinstance(_personalized, tuple) and len(_personalized) == 5:
#         final_recipes, risk_reason, global_notes, cal_target, prot_target = _personalized
#     else:
#         final_recipes, risk_reason, global_notes = _personalized
#         cal_target, prot_target = None, None
# 
#     # Build health banner messages with ingredient suggestions
#     health_messages = []
#     if bmi is not None and bmi >= 30:
#         health_messages.append('Obesity detected: focus on high-fiber meals (vegetables, legumes, whole grains) and lean proteins; avoid sugary drinks.')
#     if hypertension == 1:
#         health_messages.append('Blood pressure is high: follow DASH; limit sodium (<1500 mg/day), emphasize fruits/vegetables, beans, whole grains.')
#     if tot_chol is not None and tot_chol >= 240:
#         health_messages.append('Cholesterol is high: add oats, barley, legumes, nuts/seeds, olive oil; include omega-3 sources (flax/chia/walnuts or fish).')
#     if diabetes == 1:
#         health_messages.append('Glucose elevated/diabetes: choose low-GI options, more fiber; avoid refined sugar and sweetened beverages.')
#     if alcohol == 1:
#         health_messages.append('Alcohol use: reduce to <2 drinks/week and hydrate well; add potassium-rich foods.')
#     try:
#         smoking_flag = int(smoking)
#     except Exception:
#         smoking_flag = 0
#     if smoking_flag == 1:
#         health_messages.append('Smoking: increase antioxidant-rich foods (berries, citrus, leafy greens, cruciferous vegetables).')
# 
#     return render_template('diet_results.html', recipes=final_recipes, risk_reason=risk_reason, global_notes=global_notes,
#                            cal_target=cal_target, prot_target=prot_target, health_messages=health_messages)


@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/QuitSmoking')
def QuitSmoking():
    return render_template('QuitSmoking.html')

# @app.route('/diet/medical-filters', methods=['GET'])
# def medical_filters_page():
#     # Read multi-select conditions from query string
#     selected = request.args.getlist('conditions')
#     selected = [c.strip().lower() for c in selected]
# 
#     # Build sections for the template
#     sections = {}
#     label_map = {
#         'diabetes': 'Diabetes: Avoid These Foods',
#         'high_bp': 'High Blood Pressure: Avoid These Foods',
#         'cholesterol': 'High Cholesterol: Avoid These Foods',
#     }
#     # Normalize synonyms
#     normalized = []
#     for c in selected:
#         if c in ('hypertension', 'high blood pressure', 'bp', 'highbp'):
#             normalized.append('high_bp')
#         else:
#             normalized.append(c)
#     for key in ['diabetes', 'high_bp', 'cholesterol']:
#         if key in normalized:
#             items = RESTRICTION_CATALOG.get(key, [])
#             sections[key] = {
#                 'title': label_map.get(key, key.title()),
#                 'items': items,
#             }
#     return render_template('medical_filters.html', sections=sections, selected=set(normalized))

# Debug endpoint to verify which project instance and templates are being used
@app.route('/__debug')
def __debug_info():
    try:
        from pathlib import Path
        tpl_dir = Path(app.root_path) / 'templates'
        diet_tpl = tpl_dir / 'diet_results.html'
        content = ''
        exists = diet_tpl.exists()
        if exists:
            content = diet_tpl.read_text(encoding='utf-8', errors='ignore')
        return {
            'cwd': os.getcwd(),
            'flask_root': app.root_path,
            'templates_path': str(tpl_dir),
            'diet_results_path': str(diet_tpl),
            'diet_results_exists': exists,
            'diet_results_has_fit_score_text': ('fit_score' in content) if exists else None,
        }
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == "__main__":
    # Helpful startup logs to verify the running project and template directory
    try:
        print({
            'cwd': os.getcwd(),
            'flask_root': app.root_path,
            'templates_path': os.path.join(app.root_path, 'templates'),
            'static_path': os.path.join(app.root_path, 'static'),
        })
    except Exception:
        pass
    # Run with debug to auto-reload templates and code during development
    port = 0
    try:
        port = int(os.environ.get('PORT') or os.environ.get('FLASK_RUN_PORT') or 5000)
    except Exception:
        port = 5000
    print({'port': port})
    app.run(debug=True, port=port)
    