from flask import Flask, request, render_template, redirect, send_from_directory, url_for
import pandas as pd
import joblib
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key_here'

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
def welcome():
    return render_template('welcome.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        selected_model = request.form.get("model")

        if file.filename.endswith('.csv') and selected_model:
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            rename_map = {
                'Exact_Match_Exists': 'Exact Match Exists',
                'Appears_as_Subsequence_(MHC_Binding)': 'Appears as Subsequence (MHC Binding)',
                'Closest_Peptide': 'Closest Peptide',
                'Path_Length': 'Path Length'
            }
            df.rename(columns=rename_map, inplace=True)

            model = ml_models.get(selected_model) or dl_models.get(selected_model)
            if hasattr(model, "feature_names_in_"):
                required_features = list(model.feature_names_in_)
            else:
                required_features = all_features

            missing = [feat for feat in required_features if feat not in df.columns]
            if missing:
                return f"Missing features for model '{selected_model}': {missing}"

            input_df = df[required_features].copy()

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

            df['prediction'] = predictions
            display_df = df.drop(columns=['target'], errors='ignore')
            result_filename = f'{selected_model}_results.csv'
            result_path = os.path.join('uploads', result_filename)
            df.to_csv(result_path, index=False)

            # 🌟 Enhanced Epitope Ranking using Biological Scores
            ranked_df = df[df['prediction'] == 1].copy()

            for feature in ['kolaskar_tongaonkar', 'emini', 'parker']:
                if feature in ranked_df.columns:
                    min_val = ranked_df[feature].min()
                    max_val = ranked_df[feature].max()
                    ranked_df[feature + '_norm'] = (ranked_df[feature] - min_val) / (max_val - min_val + 1e-8)

            ranked_df['epitope_score'] = (
                0.5 * ranked_df.get('kolaskar_tongaonkar_norm', 0) +
                0.3 * ranked_df.get('emini_norm', 0) +
                0.2 * ranked_df.get('parker_norm', 0)
            )

            ranking_result = ranked_df[['peptide_seq', 'epitope_score']].groupby('peptide_seq').mean().reset_index()
            ranking_result = ranking_result.sort_values(by='epitope_score', ascending=False)

            # Save ranking result with model name in the filename
            ranking_csv_path = os.path.join('uploads', f'{selected_model}_epitope_ranking.csv')
            ranking_result.to_csv(ranking_csv_path, index=False)

            # Plot Top 10 for model-specific ranking
            plt.figure(figsize=(10, 6))
            sns.barplot(x='epitope_score', y='peptide_seq', data=ranking_result.head(10), palette='Greens_r')
            plt.title(f"Top 10 Ranked Epitopes - {selected_model}")
            plt.xlabel("Composite Epitope Score")
            plt.ylabel("Peptide Sequence")
            plt.tight_layout()
            ranking_plot_path = os.path.join('uploads', f'{selected_model}_ranking_plot.png')
            plt.savefig(ranking_plot_path)
            plt.close()

            table_html = display_df.to_html(classes='table table-bordered table-striped', index=False)

            return render_template(
                'predict.html',
                model_name=selected_model,
                table=table_html,
                download_link=f"/download/{result_filename}",
                show_ranking_link=True,
                ranking_file=f'{selected_model}_epitope_ranking.csv',
                ranking_plot=f'{selected_model}_ranking_plot.png'
            )
    return redirect('/')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('uploads', filename, as_attachment=False)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
