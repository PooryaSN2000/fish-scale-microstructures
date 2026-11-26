import os
import sys
import io
import time
import json
import uuid
import base64
import shutil
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '../..'))
SRC_DIR = os.path.join(PROJECT_ROOT, "multiclass_35species/src")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "multiclass_35species/checkpoints")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import load_rgb_image
from inference import FishClassifierV4
from retrain_active_learning import ActiveLearningEngineV4

MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
PROTOS_PATH = os.path.join(CHECKPOINTS_DIR, "prototypes.pth")

classifier_v4 = None
active_learning = None


def init_v4():
    global classifier_v4, active_learning
    if os.path.exists(MODEL_PATH) and os.path.exists(PROTOS_PATH):
        classifier_v4 = FishClassifierV4(model_path=MODEL_PATH, prototypes_path=PROTOS_PATH)
        print(f"[AquaLens v4.1] Loaded classifier with {len(classifier_v4.class_names)} species.")
    else:
        print("[AquaLens v4.1 Warning] Model or prototypes checkpoint not found.")

    al_dir = os.path.join(BASE_DIR, "active_learning_data")
    active_learning = ActiveLearningEngineV4(active_learning_dir=al_dir, checkpoints_dir=CHECKPOINTS_DIR)


def _image_to_preview_b64(image_path: str) -> str:
    img = load_rgb_image(image_path)
    img.thumbnail((800, 800))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


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
        custom_species=sorted(classifier_v4.custom_species) if classifier_v4 else [],
        al_stats=al_stats,
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
    filename = f"{sample_id}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        t0 = time.time()
        pred_species, conf, all_probs, sim, all_sims, meta = classifier_v4.predict(
            filepath, auto_crop=auto_crop, return_metadata=True
        )
        cam_b64 = classifier_v4.explain(filepath, auto_crop=auto_crop)
        preview_b64 = _image_to_preview_b64(filepath)
        inference_time = time.time() - t0

        sim_pct = {k: round(max(0.0, v) * 100, 1) for k, v in all_sims.items()}
        sorted_sims = sorted(sim_pct.items(), key=lambda x: x[1], reverse=True)

        top5_list = []
        for rank, (sp_name, sp_sim) in enumerate(sorted_sims[:5], start=1):
            genus = sp_name.split()[0] if sp_name else ""
            top5_list.append({
                "rank": rank,
                "species_name": sp_name,
                "similarity": sp_sim,
                "probability": round(all_probs.get(sp_name, 0.0) * 100, 1),
                "genus": genus,
                "is_custom": sp_name in classifier_v4.custom_species,
            })

        top3_candidates = top5_list[:3]
        overall_sim = round(max(0.0, sim) * 100, 1)
        n_species = len(classifier_v4.class_names)
        membership = meta.get("membership", "out_of_group")
        in_domain_score = meta.get("in_domain_score", overall_sim)
        maha_dist = meta.get("maha_dist")
        maha_cutoff = meta.get("maha_cutoff")

        if membership == "in_group":
            membership_title_fa = f"تطابق تاییدشده در منیفولد ({n_species} گونه)"
            membership_title_en = f"Verified In-Domain Match ({n_species} Taxa)"
            membership_desc_fa = f"هم‌ترازی قوی با پیش‌نمونه و فاصله ماهالانوبیس مجاز (امتیاز منیفولد: {in_domain_score}٪)."
            membership_desc_en = f"Strong prototype alignment and manifold fit (In-Domain Score: {in_domain_score}%)."
        elif membership == "borderline":
            membership_title_fa = "تطابق مرزی / رقابت در ۳ گزینه برتر (Top-3)"
            membership_title_en = "Borderline / Review Top-3 Candidates"
            membership_desc_fa = f"نمونه نزدیک به مرز منیفولد است (امتیاز منیفولد: {in_domain_score}٪). گزینه‌های ۱ تا ۳ برتر را بررسی کرده یا در صورت نیاز گونه جدید ثبت کنید."
            membership_desc_en = f"Near manifold boundary (Score: {in_domain_score}%). Review Top-3 shortlist or register as a novel species."
        else:
            membership_title_fa = f"گونه ناشناخته / خارج از {n_species} گونه فعلی (OOD)"
            membership_title_en = f"Novel / Out-of-Distribution Sample (Outside {n_species} Taxa)"
            membership_desc_fa = f"تصویر خارج از منیفولد آماری گونه‌های شناخته‌شده است (امتیاز منیفولد: {in_domain_score}٪). می‌توانید بدون آموزش مجدد، همین تصویر را ثبت کنید."
            membership_desc_en = f"Outside learned taxonomic manifold (In-Domain Score: {in_domain_score}%). Register directly as a novel species without retraining."

        genus = pred_species.split()[0] if pred_species else ""

        return jsonify({
            "sample_id": sample_id,
            "filename": filename,
            "prediction": pred_species,
            "genus": genus,
            "overall_similarity": overall_sim,
            "in_domain_score": in_domain_score,
            "maha_dist": maha_dist,
            "maha_cutoff": maha_cutoff,
            "confidence": round(conf * 100, 1),
            "membership": membership,
            "membership_title_fa": membership_title_fa,
            "membership_title_en": membership_title_en,
            "membership_desc_fa": membership_desc_fa,
            "membership_desc_en": membership_desc_en,
            "is_uncertain": meta.get("is_uncertain", False),
            "is_custom_species": meta.get("is_custom_species", False),
            "uncertainty_reason": meta.get("uncertainty_reason", ""),
            "margin": meta.get("margin", 0.0),
            "top3_candidates": top3_candidates,
            "top3_cumulative_confidence": meta.get("top3_cumulative_confidence", 0.0),
            "top3_consensus_genus": meta.get("top3_consensus_genus", genus),
            "top3_consensus_count": meta.get("top3_consensus_count", 1),
            "top_matches": top5_list,
            "preview_image": preview_b64,
            "crop_overlay_image": meta.get("crop_overlay_image"),
            "gradcam_image": cam_b64,
            "execution_time": f"{inference_time:.2f}s",
            "species_count": n_species,
            "species_list": sorted(classifier_v4.class_names),
            "custom_species": sorted(classifier_v4.custom_species),
            "model_badge": "v4.1: ScaleDetector + MixStyle + FDA + CB-ArcFace + 7-View Spherical-TTA",
        })

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    if active_learning is None:
        return jsonify({'error': 'Active learning engine not initialized.'}), 500

    data = request.get_json() or request.form
    sample_id = data.get('sample_id')
    filename = data.get('filename')
    status = data.get('status')
    confirmed_species = data.get('confirmed_species')
    notes = data.get('notes', '')

    if not filename or not status or not confirmed_species:
        return jsonify({'error': 'Missing required fields (filename, status, confirmed_species).'}), 400

    src_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(src_path):
        return jsonify({'error': 'Sample file no longer exists in temporary storage.'}), 404

    dest_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sample_id}_{secure_filename(filename)}"
    dest_path = os.path.join(active_learning.verified_samples_dir, dest_filename)
    shutil.copy2(src_path, dest_path)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "sample_id": sample_id,
        "sample_filename": dest_filename,
        "status": status,
        "confirmed_species": confirmed_species,
        "notes": notes,
    }
    log_file = os.path.join(active_learning.active_learning_dir, "feedback_log.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return jsonify({
        'success': True,
        'message': f"Sample successfully recorded for species '{confirmed_species}'.",
        'stats': active_learning.get_stats(),
    })


@app.route('/register_species', methods=['POST'])
def register_species():
    """
    Zero-retraining Open-Set Few-Shot Registration:
    Accepts either an existing uploaded `filename` OR 1..K uploaded `files` alongside `species_name`,
    builds a hyperspherical prototype in <1s, and updates prototypes.pth + live classifier.
    """
    if active_learning is None or classifier_v4 is None:
        return jsonify({'error': 'System components not initialized.'}), 500

    species_name = ""
    paths = []

    if request.is_json:
        data = request.get_json()
        species_name = (data.get('species_name') or '').strip()
        filename = data.get('filename')
        if filename:
            p = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(p):
                paths.append(p)
    else:
        species_name = (request.form.get('species_name') or '').strip()
        filename = request.form.get('filename')
        if filename:
            p = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(p):
                paths.append(p)
        uploaded_files = request.files.getlist('files')
        for f in uploaded_files:
            if f and f.filename:
                uid = str(uuid.uuid4())[:8]
                dest = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_{secure_filename(f.filename)}")
                f.save(dest)
                paths.append(dest)

    if not species_name:
        return jsonify({'success': False, 'error': 'Species name is required.'}), 400
    if not paths:
        return jsonify({'success': False, 'error': 'At least one scale image is required.'}), 400

    t0 = time.time()
    res = active_learning.register_new_species(classifier_v4, species_name, paths)
    res['elapsed_time'] = f"{time.time() - t0:.2f}s"
    res['species_list'] = sorted(classifier_v4.class_names)
    res['stats'] = active_learning.get_stats()
    return jsonify(res), (200 if res.get('success') else 400)


@app.route('/remove_custom_species', methods=['POST'])
def remove_custom_species():
    if active_learning is None or classifier_v4 is None:
        return jsonify({'error': 'System components not initialized.'}), 500

    data = request.get_json() or {}
    species_name = data.get('species_name')  # None removes all custom species
    res = active_learning.remove_custom_species(classifier_v4, species_name=species_name)
    res['species_list'] = sorted(classifier_v4.class_names)
    res['stats'] = active_learning.get_stats()
    return jsonify(res), (200 if res.get('success') else 400)


@app.route('/sync_al', methods=['POST'])
def sync_active_learning():
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
