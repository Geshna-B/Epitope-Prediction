# Epitope Prediction for Vaccine Development

## Overview
**Epitope Prediction for Vaccine Development** is a Machine Learning (ML) and Deep Learning (DL) project that predicts **B-cell epitopes** from protein sequences using curated datasets from IEDB and UniProt.  
The system extracts key sequence features (hydrophobicity, antigenicity, accessibility, etc.), applies advanced ML/DL models, and ranks epitopes based on immunological relevance.  
A **web-based interface** enables users to upload sequences, run predictions, and visualize results, offering a fast and scalable solution to support vaccine design and immunotherapy research.  

## Objectives
- Develop robust ML/DL models for **B-cell epitope prediction**.  
- Use curated datasets from **IEDB & UniProt**.  
- Implement **feature engineering** (hydrophobicity, aromaticity, stability, accessibility).  
- Apply **algorithms** like Dijkstra’s, Min Hashing, Trie for better sequence representation.  
- Compare ML models: Random Forest, XGBoost, LightGBM, CatBoost, GaussianNB.  
- Train DL models: **Feedforward Neural Network (FNN)** and **LSTM**.  
- Build a **web-based tool** for user-friendly predictions and visualization.  
- Rank epitopes based on **biological scores** (antigenicity, hydrophobicity, accessibility).  

## System Architecture
1. **Data Collection** → IEDB + UniProt datasets (`input_bcell.csv`, `input_sars.csv`, `input_covid.csv`).  
2. **Preprocessing** → Encoding sequences, normalization, handling missing values, class imbalance.  
3. **Feature Engineering** → Sequence-based & physicochemical features.  
4. **Model Training** → ML & DL models with cross-validation.  
5. **Prediction & Ranking** → Classify epitopes and rank based on immune relevance.  
6. **Visualization** → Tabular predictions + ranked epitope plots.  
7. **Web Interface** → Upload sequences, select models, download results.  

## Dataset
- **input_bcell.csv** → 14,387 entries (B-cell epitope training set).  
- **input_sars.csv** → 520 entries (SARS protein peptides).  
- **input_covid.csv** → Target dataset (COVID sequences without labels).  
- Features include **Chou-Fasman, Emini, Kolaskar-Tongaonkar, Parker indices**, plus hydrophobicity, stability, and other physicochemical properties.
  
## Models Implemented
### Machine Learning
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  
- Gaussian Naïve Bayes  

### Deep Learning
- Feedforward Neural Network (FNN)  
- Long Short-Term Memory (LSTM)  

## Results
- **Best Models** → LSTM & XGBoost (highest accuracy & F1-score).  
- Feature importance revealed **Kolaskar-Tongaonkar & Emini indices** as top predictors.  
- Visualizations included **ROC curves, feature distributions, and ranked epitope plots**.  

## Future Work
- Extend prediction to **T-cell epitopes**.  
- Incorporate **MHC binding affinity** for deeper biological relevance.  
- Use **Graph Neural Networks (GNNs)** for peptide-protein graph representations.  
- Integrate **ProtBERT/AlphaFold embeddings** via transfer learning.  
- Real-time epitope prediction for vaccine design.  
- Deploy as a **cloud-based platform** for researchers.  

## Project Structure

Epitope-Prediction/

├── data/                # input datasets (bcell, sars, covid).

├── src/                 # source code (ML/DL models, feature extraction).

├── notebooks/           # Jupyter notebooks for EDA and experiments.

├── models/              # trained model files.

├── webapp/              # web-based interface code.

├── requirements.txt     # dependencies.

└── README.md            # project documentation.

## Installation

git clone https://github.com/your-username/Epitope-Prediction.git
cd Epitope-Prediction
pip install -r requirements.txt

## Usage

### Run Prediction Script

python src/predict.py --input data/input_test.csv --model lstm

### Run Web App

streamlit run webapp/app.py


Upload your **CSV peptide sequences**, choose a model, and visualize ranked epitope predictions.

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.


## Team

* **Geshna B**
* **Katikala Dedeepya**
* **Malavika S Prasad**
* **Vada Gouri Hansika Reddy**

Supervisors: **Dr. Vinith R & Dr. Manoj Bhat**
Department of Artificial Intelligence, Amrita Vishwa Vidyapeetham

