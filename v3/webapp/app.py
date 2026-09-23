import os
import sys
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from inference import FishClassifier

import base64
import csv
import json
import uuid
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'checkpoints'))
ACTIVE_LEARNING_DIR = os.path.join(BASE_DIR, 'active_learning_data')
os.makedirs(ACTIVE_LEARNING_DIR, exist_ok=True)

app = Flask(__name__)
app.json.sort_keys = False
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the 40-class classifier
try:
    classifier = FishClassifier(
        model_path=os.path.join(CHECKPOINT_DIR, "best_model.pth"), 
        prototypes_path=os.path.join(CHECKPOINT_DIR, "prototypes.pth")
    )
    print(f"FishClassifier initialized successfully with {len(classifier.class_names)} species.")
except Exception as e:
    print(f"Notice: Model checkpoint will be loaded once trained ({e}).")
    classifier = None

@app.route('/')
def index():
    species_list = classifier.class_names if classifier else []
    return render_template('index.html', species_list=species_list)

@app.route('/predict', methods=['POST'])
def predict():
    if classifier is None:
        return jsonify({'error': 'Classifier not initialized. Please ensure best_model.pth and prototypes.pth exist in checkpoints/'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part provided.'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            start_time = time.time()
            pred_class, confidence, all_probs, max_sim, all_sims = classifier.predict(filepath)
            execution_time = time.time() - start_time
            
            # Generate Grad-CAM explainability heatmap before cleanup
            cam_overlay_b64 = classifier.explain(filepath)
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            # Map cosine similarities [-1, 1] to percentages [0, 100]
            sim_percentages = {k: round(max(0.0, v) * 100, 1) for k, v in all_sims.items()}
            overall_similarity = round(max(0.0, max_sim) * 100, 1)
            
            # Top-5 closest species
            sorted_sims = sorted(sim_percentages.items(), key=lambda x: x[1], reverse=True)
            top_5_matches = dict(sorted_sims[:5])
            
            # Categorize membership status into 3 distinct tiers:
            if overall_similarity >= 75.0:
                membership = "in_group"
                status_title = "Inside 40-Species Target Group (عضو ۴۰ گونه شناخته‌شده)"
                status_desc = f"این تصویر با تشابه ساختاری بالای {overall_similarity}٪ متعلق به ۴۰ گونه هدف است."
            elif overall_similarity >= 60.0:
                membership = "borderline"
                status_title = "Borderline / Moderate Similarity (تشابه متوسط / مرزی)"
                status_desc = f"این تصویر دارای تشابه متوسط ({overall_similarity}٪) با الگوها است. ممکن است گونه‌ای بسیار نزدیک یا تصویری با وضوح کمتر باشد."
            else:
                membership = "out_of_group"
                status_title = "Not in Group (خارج از ۴۰ گونه هدف)"
                status_desc = f"این تصویر به ۴۰ گونه تحت پوشش تعلق ندارد (حداکثر تشابه فقط {overall_similarity}٪ است)."
            
            # Format the output for the frontend
            return jsonify({
                'success': True,
                'membership': membership,
                'status_title': status_title,
                'status_desc': status_desc,
                'overall_similarity': overall_similarity,
                'prediction': pred_class,
                'confidence': round(confidence * 100, 2),
                'top_matches': top_5_matches,
                'all_similarities': sim_percentages,
                'gradcam_image': cam_overlay_b64,
                'execution_time': f"{execution_time:.3f}s"
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Active Learning Endpoint:
    Records fishery expert confirmation or correction for uploaded scales,
    saving validated images into active_learning_data/ for iterative model retraining.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON payload received.'}), 400
            
        image_data = data.get('image_data')
        predicted_species = data.get('predicted_species', 'Unknown')
        confirmed_species = data.get('confirmed_species', predicted_species)
        status = data.get('status', 'confirmed')  # 'confirmed' or 'corrected'
        expert_notes = data.get('notes', '')
        
        if not image_data or not confirmed_species:
            return jsonify({'error': 'Missing image_data or confirmed_species.'}), 400
            
        # Parse base64 image data
        if ',' in image_data:
            header, encoded = image_data.split(',', 1)
        else:
            encoded = image_data
        img_bytes = base64.b64decode(encoded)
        
        # Target species folder
        species_folder = os.path.join(ACTIVE_LEARNING_DIR, 'verified_samples', secure_filename(confirmed_species))
        os.makedirs(species_folder, exist_ok=True)
        
        timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp_str}_{unique_id}.jpg"
        save_path = os.path.join(species_folder, filename)
        
        with open(save_path, 'wb') as f:
            f.write(img_bytes)
            
        # Append to JSONL log
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'filename': filename,
            'relative_path': os.path.relpath(save_path, ACTIVE_LEARNING_DIR),
            'predicted_species': predicted_species,
            'confirmed_species': confirmed_species,
            'status': status,
            'expert_notes': expert_notes
        }
        jsonl_path = os.path.join(ACTIVE_LEARNING_DIR, 'feedback_log.jsonl')
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
        # Append to CSV summary log
        csv_path = os.path.join(ACTIVE_LEARNING_DIR, 'feedback_log.csv')
        csv_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(['timestamp', 'filename', 'predicted_species', 'confirmed_species', 'status', 'expert_notes'])
            writer.writerow([
                log_entry['timestamp'],
                filename,
                predicted_species,
                confirmed_species,
                status,
                expert_notes
            ])
            
        return jsonify({
            'success': True,
            'message': f"تصویر با موفقیت در بانک گونه‌های تاییدشده '{confirmed_species}' ذخیره شد.",
            'saved_file': filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
