# 🏠 House Price Prediction — ANN

A deep learning project that predicts the **sale price of a house** based on features like size, location, quality, and number of rooms using an Artificial Neural Network (ANN).

## 📌 Overview
- **Algorithm:** Artificial Neural Network (ANN)
- **Dataset:** Kaggle House Prices Dataset (1460 samples, 80+ features)
- **Task:** Regression (Predict SalePrice in USD)
- **Result:** MAE ~$22,000 on test data

## 🛠️ Tech Stack
- Python, NumPy, Pandas
- TensorFlow, Keras
- Scikit-learn
- Jupyter Notebook

## 📁 Files
- `data/train.csv` — Training dataset
- `data/test.csv` — Test dataset
- `notebooks/Ann_house_Project.ipynb` — Jupyter notebook with full code
- `requirements.txt` — Python dependencies

## 🚀 Run
```bash
pip install -r requirements.txt
```
Then open `notebooks/Ann_house_Project.ipynb` in Jupyter or Google Colab and run all cells.

## 📊 Screenshots

### Model Summary
![Model Summary](screenshots/Screenshot%20(1).png)

### Training Loss Curves
![Training Curves](screenshots/Screenshot%20(2).png)

### Final Test MAE
![Final MAE](screenshots/Screenshot%20(3).png)