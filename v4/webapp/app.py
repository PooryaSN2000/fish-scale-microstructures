import os
import sys
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Add v4/src to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'src'))
sys.path.append(SRC_DIR)

from inference import FishClassifierV4
import base64
import csv
import json
import uuid
from datetime import datetime

CHECKPOINT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'checkpoints'))
ACTIVE_LEARNING_DIR = os.path.join(BASE_DIR, 'active_learning_data')
os.makedirs(ACTIVE_LEARNING_DIR, exist_ok=True)

app = Flask(__name__)
app.json.sort_keys = False
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the v4 classifier
model_file = os.path.join(CHECKPOINT_DIR, "best_model.pth")
proto_file = os.path.join(CHECKPOINT_DIR, "prototypes.pth")

try:
    if os.path.exists(model_file) and os.path.exists(proto_file):
        classifier = FishClassifierV4(model_path=model_file, prototypes_path=proto_file)
        print(f"[AquaLens v4 WebApp] Classifier initialized with {len(classifier.class_names)} species.")
    else:
        print("[AquaLens v4 WebApp] Notice: Checkpoints will be loaded once training is complete.")
        classifier = None
except Exception as e:
    print(f"[AquaLens v4 WebApp] Notice: Loading error ({e}). Model can be initialized post-training.")
    classifier = None

@app.route('/')
def index():
    species_list = classifier.class_names if classifier else []
    return render_template('index.html', species_list=species_list, version="v4")

@app.route('/predict', methods=['POST'])
def predict():
    global classifier
    if classifier is None:
        if os.path.exists(model_file) and os.path.exists(proto_file):
            classifier = FishClassifierV4(model_path=model_file, prototypes_path=proto_file)
        else:
            return jsonify({'error': 'v4 model checkpoints not found. Run training script first.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part provided.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    auto_crop = request.form.get('auto_crop', 'false').lower() == 'true'

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            start_time = time.time()
            pred_class, confidence, all_probs, max_sim, all_sims, meta = classifier.predict(
                filepath, 
                auto_crop=auto_crop, 
                return_metadata=True
            )
            execution_time = time.time() - start_time

            # Generate Grad-CAM explainability heatmap
            cam_overlay_b64 = classifier.explain(filepath)

            # Clean up uploaded file
            os.remove(filepath)

            sim_percentages = {k: round(max(0.0, v) * 100, 1) for k, v in all_sims.items()}
            overall_similarity = round(max(0.0, max_sim) * 100, 1)

            sorted_sims = sorted(sim_percentages.items(), key=lambda x: x[1], reverse=True)
            top_5_matches = dict(sorted_sims[:5])

            if overall_similarity >= 75.0:
                membership = "in_group"
            elif overall_similarity >= 60.0:
                membership = "borderline"
            else:
                membership = "out_of_group"

            return jsonify({
                'prediction': pred_class,
                'overall_similarity': overall_similarity,
                'confidence': round(confidence * 100, 1),
                'membership': membership,
                'is_uncertain': meta["is_uncertain"],
                'uncertainty_reason': meta["uncertainty_reason"],
                'margin': meta["margin"],
                'top_matches': top_5_matches,
                'all_similarities': sim_percentages,
                'crop_overlay_image': meta["crop_overlay_image"],
                'gradcam_image': cam_overlay_b64,
                'execution_time': f"{execution_time:.2f}s"
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=False)
