import os
import sys
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from inference import FishClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'checkpoints'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the classifier (loads model and prototypes)
try:
    classifier = FishClassifier(
        model_path=os.path.join(CHECKPOINT_DIR, "best_model.pth"), 
        prototypes_path=os.path.join(CHECKPOINT_DIR, "prototypes.pth")
    )
    print("FishClassifier initialized successfully.")
except Exception as e:
    print(f"Error initializing classifier: {e}")
    classifier = None

import time

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if classifier is None:
        return jsonify({'error': 'Classifier not initialized properly.'}), 500
        
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
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            # Map cosine similarities [-1, 1] to percentages [0, 100]
            sim_percentages = {k: round(max(0.0, v) * 100, 1) for k, v in all_sims.items()}
            overall_similarity = round(max(0.0, max_sim) * 100, 1)
            
            # Categorize membership status into 3 distinct, reliable tiers:
            if overall_similarity >= 75.0:
                membership = "in_group"
                status_title = "Inside Group (عضو گروه Lutjanus)"
                status_desc = f"این تصویر با تشابه ساختاری بالای {overall_similarity}٪ متعلق به گروه ماهی‌های Lutjanus است."
            elif overall_similarity >= 60.0:
                membership = "borderline"
                status_title = "Borderline / Moderate Similarity (تشابه متوسط / مرزی)"
                status_desc = f"این تصویر دارای تشابه متوسط ({overall_similarity}٪) است. ممکن است گونه‌ای نزدیک یا تصویری با وضوح کمتر باشد."
            else:
                membership = "out_of_group"
                status_title = "Not in Group (خارج از این گروه)"
                status_desc = f"این تصویر به این گروه تعلق ندارد (حداکثر تشابه فقط {overall_similarity}٪ است)."
            
            # Format the output for the frontend
            return jsonify({
                'success': True,
                'membership': membership,
                'status_title': status_title,
                'status_desc': status_desc,
                'overall_similarity': overall_similarity,
                'prediction': pred_class,
                'confidence': round(confidence * 100, 2),
                'similarities': sim_percentages,
                'all_probabilities': {k: round(v * 100, 2) for k, v in all_probs.items()},
                'execution_time': f"{execution_time:.3f}s"
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
