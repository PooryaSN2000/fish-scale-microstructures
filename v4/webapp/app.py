import os
import sys
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '../..'))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Paths to models
V3_MODEL = os.path.join(PROJECT_ROOT, "v3/checkpoints/best_model.pth")
V3_PROTO = os.path.join(PROJECT_ROOT, "v3/checkpoints/prototypes.pth")
V4_MODEL = os.path.join(PROJECT_ROOT, "v4/checkpoints/best_model.pth")
V4_PROTO = os.path.join(PROJECT_ROOT, "v4/checkpoints/prototypes.pth")

classifier_v3 = None
classifier_v4 = None

def init_classifiers():
    global classifier_v3, classifier_v4
    # 1. Load v3
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "v3/src"))
        import inference as inf_v3
        if os.path.exists(V3_MODEL) and os.path.exists(V3_PROTO):
            classifier_v3 = inf_v3.FishClassifier(model_path=V3_MODEL, prototypes_path=V3_PROTO)
            print(f"[Arena] Classifier v3 loaded ({len(classifier_v3.class_names)} species).")
        sys.path.pop(0)
    except Exception as e:
        print(f"[Arena Warning] Could not load v3: {e}")

    # 2. Load v4
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "v4/src"))
        for m in ['inference', 'config', 'model', 'dataset', 'gradcam']:
            sys.modules.pop(m, None)
        import inference as inf_v4
        if os.path.exists(V4_MODEL) and os.path.exists(V4_PROTO):
            classifier_v4 = inf_v4.FishClassifierV4(model_path=V4_MODEL, prototypes_path=V4_PROTO)
            print(f"[Arena] Classifier v4 loaded ({len(classifier_v4.class_names)} species).")
    except Exception as e:
        print(f"[Arena Warning] Could not load v4: {e}")

init_classifiers()

app = Flask(__name__)
app.json.sort_keys = False
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    species_v3 = classifier_v3.class_names if classifier_v3 else []
    species_v4 = classifier_v4.class_names if classifier_v4 else []
    return render_template(
        'index.html',
        species_v3=species_v3,
        species_v4=species_v4,
        v3_count=len(species_v3),
        v4_count=len(species_v4)
    )

@app.route('/predict', methods=['POST'])
def predict():
    global classifier_v3, classifier_v4
    if classifier_v4 is None or classifier_v3 is None:
        init_classifiers()

    if 'file' not in request.files:
        return jsonify({'error': 'No file part provided.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    auto_crop = request.form.get('auto_crop', 'true').lower() == 'true'
    mode = request.form.get('mode', 'compare')  # 'compare', 'v4', or 'v3'

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Run v4 inference
        res_v4 = None
        if classifier_v4 is not None and mode in ['compare', 'v4']:
            t0 = time.time()
            p4, conf4, _, sim4, all_sims4, meta4 = classifier_v4.predict(filepath, auto_crop=auto_crop, return_metadata=True)
            cam4_b64 = classifier_v4.explain(filepath)
            t_v4 = time.time() - t0

            sim_pct4 = {k: round(max(0.0, v) * 100, 1) for k, v in all_sims4.items()}
            sorted_sims4 = sorted(sim_pct4.items(), key=lambda x: x[1], reverse=True)
            top5_v4 = dict(sorted_sims4[:5])
            overall_sim4 = round(max(0.0, sim4) * 100, 1)

            mem4 = "in_group" if overall_sim4 >= 75.0 else ("borderline" if overall_sim4 >= 60.0 else "out_of_group")
            res_v4 = {
                "prediction": p4,
                "overall_similarity": overall_sim4,
                "confidence": round(conf4 * 100, 1),
                "membership": mem4,
                "is_uncertain": meta4.get("is_uncertain", False),
                "uncertainty_reason": meta4.get("uncertainty_reason", ""),
                "margin": meta4.get("margin", 0.0),
                "top_matches": top5_v4,
                "all_similarities": sim_pct4,
                "crop_overlay_image": meta4.get("crop_overlay_image"),
                "gradcam_image": cam4_b64,
                "execution_time": f"{t_v4:.2f}s",
                "species_count": len(classifier_v4.class_names),
                "model_badge": "v4: MixStyle + CB-ArcFace + TTA (۵۷ گونه)",
                "features": ["Domain Generalization (MixStyle)", "Class-Balanced ArcFace", "Spherical-TTA (۵ زاویه)", "۵۷ گونه زیستی"]
            }

        # Run v3 inference
        res_v3 = None
        if classifier_v3 is not None and mode in ['compare', 'v3']:
            t0 = time.time()
            p3, conf3, _, sim3, all_sims3, meta3 = classifier_v3.predict(filepath, auto_crop=auto_crop, return_metadata=True)
            cam3_b64 = classifier_v3.explain(filepath)
            t_v3 = time.time() - t0

            sim_pct3 = {k: round(max(0.0, v) * 100, 1) for k, v in all_sims3.items()}
            sorted_sims3 = sorted(sim_pct3.items(), key=lambda x: x[1], reverse=True)
            top5_v3 = dict(sorted_sims3[:5])
            overall_sim3 = round(max(0.0, sim3) * 100, 1)

            mem3 = "in_group" if overall_sim3 >= 75.0 else ("borderline" if overall_sim3 >= 60.0 else "out_of_group")
            res_v3 = {
                "prediction": p3,
                "overall_similarity": overall_sim3,
                "confidence": round(conf3 * 100, 1),
                "membership": mem3,
                "is_uncertain": meta3.get("is_uncertain", False),
                "uncertainty_reason": meta3.get("uncertainty_reason", ""),
                "margin": meta3.get("margin", 0.0),
                "top_matches": top5_v3,
                "all_similarities": sim_pct3,
                "crop_overlay_image": meta3.get("crop_overlay_b64"),
                "gradcam_image": cam3_b64,
                "execution_time": f"{t_v3:.2f}s",
                "species_count": len(classifier_v3.class_names),
                "model_badge": "v3: ConvNeXt-Tiny + ArcFace (۴۰ گونه)",
                "features": ["Standard ArcFace", "Single-View Embedding", "۴۰ گونه هدف"]
            }

        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

        # Consensus & Comparison analysis
        consensus = None
        if res_v3 and res_v4:
            is_agree = (res_v3["prediction"] == res_v4["prediction"])
            v4_is_new_species = (res_v4["prediction"] not in classifier_v3.class_names)
            sim_diff = round(res_v4["overall_similarity"] - res_v3["overall_similarity"], 1)

            consensus = {
                "agree": is_agree,
                "v3_pred": res_v3["prediction"],
                "v4_pred": res_v4["prediction"],
                "v3_sim": res_v3["overall_similarity"],
                "v4_sim": res_v4["overall_similarity"],
                "sim_diff": sim_diff,
                "v4_is_new_species": v4_is_new_species,
                "verdict_title_fa": "همگرایی کامل دو مدل" if is_agree else "اختلاف نظر تشخیصی بین دو نسخه",
                "verdict_title_en": "Complete Model Consensus" if is_agree else "Diagnostic Divergence Between Versions",
                "verdict_desc_fa": f"هر دو مدل با تشابه بالا گونه «{res_v4['prediction']}» را شناسایی کردند." if is_agree else (
                    f"مدل v4 گونه «{res_v4['prediction']}» را پیش‌بینی کرده که از ۱۷ گونه جدید تحت پوشش v4 است و در دامنه مدل v3 وجود ندارد!" if v4_is_new_species else
                    f"مدل v3 گونه «{res_v3['prediction']}» ({res_v3['overall_similarity']}٪) و مدل v4 گونه «{res_v4['prediction']}» ({res_v4['overall_similarity']}٪) را پیشنهاد کردند. پایداری v4 به دلیل Spherical-TTA بالاتر است."
                ),
                "verdict_desc_en": f"Both versions independently agreed on '{res_v4['prediction']}'." if is_agree else (
                    f"v4 predicted '{res_v4['prediction']}', one of the 17 rare taxa newly modeled in v4 (outside v3's 40-species domain)." if v4_is_new_species else
                    f"v3 predicted '{res_v3['prediction']}' ({res_v3['overall_similarity']}%) vs v4 '{res_v4['prediction']}' ({res_v4['overall_similarity']}%)."
                )
            }

        return jsonify({
            "mode": mode,
            "consensus": consensus,
            "v3": res_v3,
            "v4": res_v4
        })

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=False)
