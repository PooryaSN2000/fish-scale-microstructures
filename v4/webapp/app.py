import os
import sys
import time
import json
import uuid
import shutil
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

# Ensure environment stability
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '../..'))
SRC_DIR = os.path.join(PROJECT_ROOT, "v4/src")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "v4/checkpoints")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference import FishClassifierV4
from retrain_active_learning import ActiveLearningEngineV4

# Initialize single dedicated v4 inference engine and active learning manager
MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
PROTOS_PATH = os.path.join(CHECKPOINTS_DIR, "prototypes.pth")

classifier_v4 = None
active_learning = None

def init_v4():
    global classifier_v4, active_learning
    if os.path.exists(MODEL_PATH) and os.path.exists(PROTOS_PATH):
        classifier_v4 = FishClassifierV4(model_path=MODEL_PATH, prototypes_path=PROTOS_PATH)
        print(f"[AquaLens v4] Loaded dedicated classifier with {len(classifier_v4.class_names)} species.")
    else:
        print(f"[AquaLens v4 Warning] Model or prototypes checkpoint not found.")
    
    al_dir = os.path.join(BASE_DIR, "active_learning_data")
    active_learning = ActiveLearningEngineV4(active_learning_dir=al_dir, checkpoints_dir=CHECKPOINTS_DIR)

init_v4()

app = Flask(__name__)
app.json.sort_keys = False
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    species_list = sorted(classifier_v4.class_names) if classifier_v4 else []
    al_stats = active_learning.get_stats() if active_learning else {}
    return render_template(
        'index.html',
        species_list=species_list,
        species_count=len(species_list),
        al_stats=al_stats
    )

@app.route('/predict', methods=['POST'])
def predict():
    if classifier_v4 is None:
        init_v4()
    if classifier_v4 is None:
        return jsonify({'error': 'Classifier v4 could not be initialized.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part provided.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    auto_crop = request.form.get('auto_crop', 'true').lower() == 'true'

    sample_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    filename = f"{sample_id}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        t0 = time.time()
        pred_species, conf, _, sim, all_sims, meta = classifier_v4.predict(
            filepath, auto_crop=auto_crop, return_metadata=True
        )
        cam_b64 = classifier_v4.explain(filepath, auto_crop=auto_crop)
        inference_time = time.time() - t0

        sim_pct = {k: round(max(0.0, v) * 100, 1) for k, v in all_sims.items()}
        sorted_sims = sorted(sim_pct.items(), key=lambda x: x[1], reverse=True)
        
        # Build Top 5 matches with detailed 2-line presentation metadata
        top5_list = []
        for rank, (sp_name, sp_sim) in enumerate(sorted_sims[:5], start=1):
            genus = sp_name.split()[0] if sp_name else ""
            top5_list.append({
                "rank": rank,
                "species_name": sp_name,
                "similarity": sp_sim,
                "genus": genus
            })

        overall_sim = round(max(0.0, sim) * 100, 1)

        # Calibrated membership thresholds based on 163-sample empirical test distribution
        if overall_sim >= 70.0:
            membership = "in_group"
            membership_title_fa = "عضو قطعی ۵۷ گونه هدف"
            membership_title_en = "Definite In-Domain Species Match"
            membership_desc_fa = "تشابه ساختاری ریزساختار پولک با پایگاه داده زیستی در سطح تطابق بالا و قطعی قرار دارد."
            membership_desc_en = "Scale microstructure demonstrates strong invariant alignment with the species prototype."
        elif overall_sim >= 55.0:
            membership = "borderline"
            membership_title_fa = "تطابق بینابینی / گونه‌های خواهر"
            membership_title_en = "Borderline / Sibling Taxon Affinity"
            membership_desc_fa = "تشابه در محدوده حدفاصل قرار دارد؛ نیازمند بازبینی خطوط رشد توسط متخصص یا نمونه‌برداری مجدد."
            membership_desc_en = "Moderate similarity detected. Suggesting sibling taxon or optical blur on circuli."
        else:
            membership = "out_of_group"
            membership_title_fa = "خارج از پایگاه ۵۷ گونه یا تصویر غیرپولک"
            membership_title_en = "Out-of-Distribution / Non-Target Sample"
            membership_desc_fa = "میزان تطابق کمتر از آستانه اطمینان است. احتمالاً نمونه خارج از ۵۷ گونه است."
            membership_desc_en = "Similarity is below the confidence margin, indicating an out-of-domain fish species."

        genus = pred_species.split()[0] if pred_species else ""

        response_data = {
            "sample_id": sample_id,
            "filename": filename,
            "prediction": pred_species,
            "genus": genus,
            "overall_similarity": overall_sim,
            "confidence": round(conf * 100, 1),
            "membership": membership,
            "membership_title_fa": membership_title_fa,
            "membership_title_en": membership_title_en,
            "membership_desc_fa": membership_desc_fa,
            "membership_desc_en": membership_desc_en,
            "is_uncertain": meta.get("is_uncertain", False),
            "uncertainty_reason": meta.get("uncertainty_reason", ""),
            "margin": meta.get("margin", 0.0),
            "top_matches": top5_list,
            "crop_overlay_image": meta.get("crop_overlay_image"),
            "gradcam_image": cam_b64,
            "execution_time": f"{inference_time:.2f}s",
            "species_count": len(classifier_v4.class_names),
            "model_badge": "v4: MixStyle + CB-ArcFace + 7-View Spherical-TTA",
            "features": [
                "Domain Invariance (MixStyle)",
                "Class-Balanced ArcFace",
                "7-View Spherical-TTA",
                "57 Covered Biological Taxa",
                "Active Learning Ready"
            ]
        }
        return jsonify(response_data)

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Records expert verification or label correction for active learning.
    """
    if active_learning is None:
        return jsonify({'error': 'Active learning engine not initialized.'}), 500

    data = request.get_json() or request.form
    sample_id = data.get('sample_id')
    filename = data.get('filename')
    status = data.get('status')  # 'confirmed' or 'corrected'
    confirmed_species = data.get('confirmed_species')
    notes = data.get('notes', '')

    if not filename or not status or not confirmed_species:
        return jsonify({'error': 'Missing required fields (filename, status, confirmed_species).'}), 400

    src_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(src_path):
        return jsonify({'error': 'Sample file no longer exists in temporary storage.'}), 404

    # Copy sample to verified active learning storage
    dest_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sample_id}_{secure_filename(filename)}"
    dest_path = os.path.join(active_learning.verified_samples_dir, dest_filename)
    shutil.copy2(src_path, dest_path)

    # Append to feedback log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "sample_id": sample_id,
        "sample_filename": dest_filename,
        "status": status,
        "confirmed_species": confirmed_species,
        "notes": notes
    }

    log_file = os.path.join(active_learning.active_learning_dir, "feedback_log.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return jsonify({
        'success': True,
        'message': f"Sample successfully recorded for species '{confirmed_species}'.",
        'stats': active_learning.get_stats()
    })

@app.route('/sync_al', methods=['POST'])
def sync_active_learning():
    """
    Performs fast (<1.5s) hyperspherical EMA synchronization of prototypes
    using the accumulated expert feedback.
    """
    if active_learning is None or classifier_v4 is None:
        return jsonify({'error': 'System components not initialized.'}), 500

    t0 = time.time()
    res = active_learning.sync_prototypes_online(classifier_v4, ema_alpha=0.15)
    res['elapsed_time'] = f"{time.time() - t0:.2f}s"
    res['stats'] = active_learning.get_stats()
    return jsonify(res)

@app.route('/al_stats', methods=['GET'])
def get_al_stats():
    if active_learning is None:
        return jsonify({})
    return jsonify(active_learning.get_stats())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=False)
