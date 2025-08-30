from flask import Flask, request, render_template, redirect, send_from_directory
import pandas as pd
import joblib
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load ML models
ml_models = {
    "RandomForest": joblib.load("models/rf_model.pkl"),
    "GaussianNB": joblib.load("models/nb_model.pkl"),
    "LightGBM": joblib.load("models/lgbm_model.pkl"),
    "CatBoost": joblib.load("models/catboost_model.pkl"),
    "XGBoost": joblib.load("models/xgb_model.pkl"),
}

# Load DL models
dl_models = {
    "FNN": tf.keras.models.load_model("models/fnn_model.h5"),
    "LSTM": tf.keras.models.load_model("models/lstm_model.h5")
}

# Required features for all models
all_features = [
    'parent_protein_id', 'protein_seq', 'peptide_seq', 'chou_fasman', 'emini',
    'kolaskar_tongaonkar', 'parker', 'isoelectric_point', 'aromaticity',
    'hydrophobicity', 'stability', 'minhash_signature', 'Exact Match Exists',
    'Appears as Subsequence (MHC Binding)', 'Closest Peptide', 'Path Length'
]

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        selected_model = request.form.get("model")

        if file.filename.endswith('.csv') and selected_model:
            # Save uploaded file
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            # Load and preprocess data
            df = pd.read_csv(filepath)

            # Rename underscored features to match trained model features
            rename_map = {
                'Exact_Match_Exists': 'Exact Match Exists',
                'Appears_as_Subsequence_(MHC_Binding)': 'Appears as Subsequence (MHC Binding)',
                'Closest_Peptide': 'Closest Peptide',
                'Path_Length': 'Path Length'
            }
            df.rename(columns=rename_map, inplace=True)

            # Get required features
            model = ml_models.get(selected_model) or dl_models.get(selected_model)
            if hasattr(model, "feature_names_in_"):
                required_features = list(model.feature_names_in_)
            else:
                required_features = all_features

            missing = [feat for feat in required_features if feat not in df.columns]
            if missing:
                return f"Missing features for model '{selected_model}': {missing}"

            input_df = df[required_features].copy()

            # Predict
            if selected_model == "LSTM":
                input_data = input_df.select_dtypes(include=[np.number]).values
                input_data = input_data.reshape((input_data.shape[0], 1, input_data.shape[1]))
                preds = dl_models[selected_model].predict(input_data)
                predictions = (preds > 0.5).astype(int).flatten()
            elif selected_model in dl_models:
                input_data = input_df.select_dtypes(include=[np.number]).values
                preds = dl_models[selected_model].predict(input_data)
                predictions = (preds > 0.5).astype(int).flatten()
            else:
                predictions = ml_models[selected_model].predict(input_df)

            # Attach predictions
            df['prediction'] = predictions
            result_filename = f'{selected_model}_results.csv'
            result_path = os.path.join('uploads', result_filename)
            df.to_csv(result_path, index=False)

            return render_template(
                'index.html',
                tables=[df.to_html(classes='data table table-bordered', index=False, justify="center")],
                titles=df.columns.values,
                model_name=selected_model,
                download_link=f"/download/{result_filename}"
            )

    return redirect('/')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('uploads', filename, as_attachment=True)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
