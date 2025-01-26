# Bank Marketing Campaign Analysis with Machine Learning

## 📊 Project Overview

This data science project focuses on analyzing bank marketing campaigns to predict whether a client will subscribe to a term deposit. Using machine learning techniques, specifically a Random Forest Classifier, we build a predictive model to help banks optimize their marketing strategies and improve campaign effectiveness.

## 🎯 Objectives

- Analyze patterns in customer responses to bank marketing campaigns
- Build a predictive model for term deposit subscriptions
- Identify key factors influencing customer decisions
- Provide insights for optimizing future marketing campaigns

## 📑 Dataset Description

The dataset contains information about direct marketing campaigns (phone calls) of a Portuguese banking institution. 

### Features:

#### Bank Client Data:
- `age`: Age of the client (numeric)
- `job`: Type of job (categorical)
- `marital`: Marital status (categorical)
- `education`: Education level (categorical)
- `default`: Has credit in default? (binary)
- `balance`: Average yearly balance in euros (numeric)
- `housing`: Has housing loan? (binary)
- `loan`: Has personal loan? (binary)

#### Campaign Data:
- `contact`: Contact communication type (categorical)
- `day`: Last contact day of the month (numeric)
- `month`: Last contact month of year (categorical)
- `campaign`: Number of contacts performed during this campaign (numeric)
- `pdays`: Days since the client was last contacted (numeric)
- `previous`: Number of contacts performed before this campaign (numeric)
- `poutcome`: Outcome of the previous marketing campaign (categorical)

#### Target Variable:
- `deposit`: Has the client subscribed to a term deposit? (binary: 'yes', 'no')

## 🛠️ Technical Implementation

### Data Preprocessing

1. **Data Cleaning**
   - Removal of the 'duration' column to prevent data leakage
   - Handling missing values
   - Data type conversions

2. **Feature Engineering**
   - Standardization of numerical features using StandardScaler
   - One-hot encoding for categorical variables
   - Feature selection based on importance

### Model Development

#### Random Forest Classifier
- Implementation using scikit-learn
- Hyperparameters:
  - random_state = 42
  - Default parameters for other settings

### Model Evaluation
- Split ratio: 70% training, 30% testing
- Metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

## 🔧 Technologies Used

### Core Technologies
- Python 3.x
- Jupyter Notebook

### Key Libraries
- **Data Processing**
  - pandas (data manipulation)
  - numpy (numerical operations)
  
- **Machine Learning**
  - scikit-learn
    - RandomForestClassifier
    - StandardScaler
    - OneHotEncoder
    - train_test_split
    
- **Visualization**
  - matplotlib
  - seaborn

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.x installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

### Installation Steps

1. Clone the repository:
   ```bash
   git clone [your-repository-url]
   cd [repository-name]
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

5. Open `yassine (1).ipynb` and run the cells in sequence

## 📈 Usage Example

```python
# Load and preprocess data
df_bank = pd.read_csv('bank.csv')
df_bank = df_bank.drop('duration', axis=1)

# Feature engineering
scaler = StandardScaler()
numerical_features = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
df_bank[numerical_features] = scaler.fit_transform(df_bank[numerical_features])

# Train model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
predictions = rf_classifier.predict(X_test)
```

## 📊 Results and Insights

The Random Forest Classifier model provides:
- Prediction of term deposit subscription likelihood
- Feature importance rankings
- Insights into customer behavior patterns

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ✍️ Author

**Yassine**

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- Portuguese Banking Institution for providing the data
- All contributors and maintainers

## 📞 Contact

For any queries or suggestions, please open an issue in the repository.
