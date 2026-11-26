import os
import sys
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from inference import FishClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'checkpoints'))

try:
    classifier = FishClassifier(
        model_path=os.path.join(CHECKPOINT_DIR, "best_model.pth"),
        prototypes_path=os.path.join(CHECKPOINT_DIR, "prototypes.pth"),
    )
    print("FishClassifier initialized successfully.")
except Exception as e:
    print(f"Error initializing classifier: {e}")
    classifier = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


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

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        start_time = time.time()
        res = classifier.predict_detailed(filepath, return_visuals=True)
        execution_time = time.time() - start_time
        if os.path.exists(filepath):
            os.remove(filepath)

        membership = res["membership"]
        overall_similarity = res["overall_similarity"]
        in_domain_score = res["in_domain_score"]
        maha_dist = res.get("maha_dist")

        if membership == "in_group":
            status_title = "Inside Group (عضو قطعی گروه Lutjanus)"
            status_desc = (
                f"این تصویر با تشابه کسینوسی {overall_similarity}٪ و تطابق منیفولد {in_domain_score}٪ "
                f"(فاصله ماهالانوبیس {maha_dist}) متعلق به خانواده Lutjanus تایید شد."
            )
        elif membership == "borderline":
            status_title = "Borderline / Moderate Similarity (تشابه مرزی یا گونه نزدیک)"
            status_desc = (
                f"تشابه کسینوسی {overall_similarity}٪ و امتیاز منیفولد {in_domain_score}٪ در محدوده مرزی است "
                f"؛ لطفاً لیست ۳ گزینه برتر (Top-3) را بررسی کنید."
            )
        else:
            status_title = "Not in Group (خارج از ۵ گونه Lutjanus)"
            status_desc = (
                f"الگوی ویژگی‌های تصویر خارج از توزیع ۵ گونه هدف است "
                f"(تشابه {overall_similarity}٪ | امتیاز منیفولد {in_domain_score}٪ | فاصله {maha_dist})."
            )

        res.update({
            'success': True,
            'status_title': status_title,
            'status_desc': status_desc,
            'execution_time': f"{execution_time:.3f}s",
        })
        return jsonify(res)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
