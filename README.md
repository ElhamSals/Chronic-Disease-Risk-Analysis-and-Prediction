# Chronic Disease Risk Analysis and Prediction

## 📌 Project Overview
This project analyzes trends in chronic disease indicators in the United States and develops a neural network model to predict disease risk levels.  

The dataset comes from the **CDC U.S. Chronic Disease Indicators (CDI)**, which includes 115 health indicators across all U.S. states and territories.

The work focuses on:
- Identifying trends and correlations in health indicators.
- Performing geographic and cluster analysis.
- Predicting chronic disease risk levels using a neural network.
- Providing visualizations to aid healthcare decision-making.

---

## 📂 Dataset
**Source:** [CDC Chronic Disease Indicators](https://data.cdc.gov/Chronic-Disease-Indicators/U-S-Chronic-Disease-Indicators/hksd-2xuw/about_data)  
**Key Features:**
- Multiple health indicators (e.g., diabetes, cancer, cardiovascular health, alcohol use, mental health).
- Geographic coverage: All U.S. states and some territories.
- Temporal coverage for trend analysis.

---

## 📊 Data Analysis Steps
1. **Trend Analysis**
   - Identify changes in health indicators over time (e.g., alcohol use, immunization).
2. **Geographic Analysis**
   - Compare states’ performance in health indicators.
3. **Correlation Analysis**
   - Check relationships between different indicators (e.g., Diabetes & Health Status = 1.00 correlation).
4. **Time Series Analysis**
   - Predict future trends for specific indicators.
5. **Visualization**
   - Use charts, line graphs, and heatmaps.
6. **Cluster Analysis**
   - Group states based on similarities in health profiles.

---

## 🤖 Neural Network Model
1. **Preprocessing**
   - Scale `datavalue` column using `MinMaxScaler`.
   - Apply K-Means clustering to label states as **Low**, **Moderate**, or **High** risk.
2. **Train/Test Split**
   - 80% training data, 20% testing data.
3. **Architecture**
   - 3 Dense layers: 32 → 16 → 3 units.
   - Final activation: `softmax`.
   - Loss: `sparse_categorical_crossentropy`.
   - Optimizer: `adam`.
4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score.
   - Noted that the dataset is **imbalanced** → performance for some classes is low.
   - **No resampling** was used for unbalanced data since the aim was to simulate real-world conditions for the course project.  
     **Future Work:** Add more balanced samples from real-world data.
5. **Visualization**
   - Display predicted risk classifications for each state.

---

## 🛠 Tools & Libraries
- **Python**: Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, Plotly.
- **Jupyter Notebook / Google Colab** for experimentation.

---

## 🚀 How to Run
1. **Clone the repository** (if uploaded to GitHub):
   ```bash
   git clone https://github.com/your-username/chronic-disease-prediction.git
   cd chronic-disease-prediction
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the analysis notebook**:
   ```bash
   jupyter notebook Test.ipynb
   ```
4. **Input dataset**:
   - Place `US_ChronicDiseaseIndicators.csv` or `.xlsx` in the project directory.
5. **View outputs**:
   - Plots and metrics will be displayed inside the notebook.

---

## 📈 Results Summary
- Clear trends found for various indicators (e.g., sharp drop in immunization in 2021 due to COVID-19).
- High correlations between some indicators (up to 1.00).
- Model achieved ~72% accuracy but struggled with minority classes due to dataset imbalance.
- Visualizations effectively highlighted state-level disparities.

---

## 📌 Future Work
- Improve **data balancing** by collecting more samples from real patients.
- Experiment with **SMOTE** or other resampling techniques.
- Enhance model complexity or try other architectures (e.g., Random Forest, Gradient Boosting).
- Deploy the model as a **web-based interactive dashboard**.

---

