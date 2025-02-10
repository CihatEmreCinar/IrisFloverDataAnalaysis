
---

# Iris Classification Model

This project is designed to classify flowers from the **Iris dataset** using machine learning models. The dataset includes features like **sepal length**, **sepal width**, **petal length**, and **petal width**, and the target is to predict the species of the flower. This notebook focuses on:

1. **Data Preprocessing**
2. **Model Training and Optimization**
3. **Model Evaluation**
4. **Model Improvement Techniques**

## Dependencies

To run this project, you will need the following libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

You can install them using pip:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Steps

### 1. Data Preprocessing
   - **Load the Data**: We load the Iris dataset from a CSV file.
   - **Label Encoding**: The categorical variable `species` is encoded into numeric values using **LabelEncoder**.
   - **Feature Scaling**: The numerical features are standardized using **StandardScaler** to ensure they are on the same scale for machine learning models.

### 2. Model Selection and Training
   - We use **Logistic Regression** with **Regularization** (L2 and L1) to train a classification model.
   - We also explore an **alternative model** with **Random Forest Classifier**.

### 3. Hyperparameter Optimization
   - **GridSearchCV** is used to find the best hyperparameters (e.g., regularization strength, solver) for the Logistic Regression model.

### 4. Model Evaluation
   - **Cross-Validation**: We use **Stratified KFold** for cross-validation to ensure robust evaluation.
   - **Performance Metrics**: We calculate **accuracy**, **classification report**, **confusion matrix**, and **ROC AUC curve** to evaluate model performance.

### 5. Model Improvement
   - We implement techniques like regularization and hyperparameter tuning to improve the model's performance.
   - Finally, the best model is selected based on **test accuracy** and further evaluated using the **confusion matrix** and **ROC curve**.

## How to Run

1. **Download the Dataset**: Ensure the dataset is available as `IRIS.csv` in the same directory.
2. **Run the Code**: Run the Python script or Jupyter notebook. Make sure all dependencies are installed.
3. **View Results**: The final model performance will be displayed, including metrics like accuracy and the classification report.

## Example Output

The model evaluation will display:

- **Test Accuracy**: The final test accuracy of the best model.
- **Classification Report**: Precision, recall, and F1-score for each class.
- **Confusion Matrix**: Visualizes the true positive, false positive, true negative, and false negative values.
- **ROC AUC Curve**: A plot showing the trade-off between true positive rate and false positive rate.

## Contributing

Feel free to contribute by submitting issues or pull requests.

