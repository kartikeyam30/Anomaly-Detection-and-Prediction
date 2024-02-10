# Unsupervised Anomaly Detection and Prediction

## Project Overview
This project aims to create an unsupervised anomaly detection algorithm followed by anomaly prediction using LSTM and an LSTM-based Transformer model. The anomaly detection component is an ensemble model that incorporates HDBSCAN, Isolation Forest (iForest), and an Autoencoder. Specifically, the ensemble involves running HDBSCAN and iForest on selected features as Model 1, then on all features as Model 2, and finally running the Autoencoder on all features. The outputs from these models are scored using an anomaly scoring system. This score is then used for predicting future instances of anomalies using both LSTM and LSTM-based Transformer models.

## Technology Stack
- Python
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn for data visualization
- Scikit-learn for model building and evaluation
- HDBSCAN, Isolation Forest for anomaly detection
- Keras/TensorFlow for LSTM and Autoencoder models

## Setup Instructions
1. Clone the repository to your local machine.
2. Ensure Python 3.x is installed.
3. Install the required dependencies using `pip install -r requirements.txt`.

## Usage
- Run `eda.py` for initial exploratory data analysis.
- Execute `light_model.py` to perform feature engineering and initial model training.
- Use `clustering.py` to apply the unsupervised clustering models (HDBSCAN, iForest, Autoencoder).
- Run `anomaly_prob.py` to score anomalies based on the model outputs.
- For future anomaly predictions, execute `light_timeseries.py` for LSTM predictions and `light_timeseries_transformer.py` for LSTM-Transformer predictions.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit pull requests with your proposed changes.

## License
This project is open-sourced under the MIT License. See the LICENSE file for more details.
