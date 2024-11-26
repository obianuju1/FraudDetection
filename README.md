# Fraud Detection Model Development

## Project Overview

This project focuses on detecting fraudulent financial transactions using machine learning techniques. The goal was to develop models that can effectively identify fraud in an imbalanced dataset of transactions, while minimizing errors in predicting legitimate transactions.

### Key Features:
- **Class Imbalance Handling**: Addressed the imbalance between fraudulent and non-fraudulent transactions using techniques like SMOTE, undersampling, and class weights.
- **Model Development**: Built and evaluated several machine learning models including Logistic Regression, Shallow Neural Networks (using TensorFlow), Random Forest, Gradient Boosting, and Linear SVC.
- **Performance Evaluation**: Models were evaluated using key metrics like Precision, Recall, and F1-score, with **Logistic Regression** and **Linear SVC** providing the best performance on the balanced dataset.

## Tools and Libraries

- **Machine Learning Libraries**: 
  - **TensorFlow** (for Neural Networks)
  - **Scikit-learn**
  - **XGBoost**
  - **Keras**
  - **LightGBM**
  
- **Data Preprocessing**: 
  - **Pandas**
  - **NumPy**
  - **RobustScaler**
  - **StandardScaler**
  - **Imbalanced-learn** (for handling class imbalance)

- **Model Evaluation**: 
  - **Precision**
  - **Recall**
  - **F1-Score**
  - **ROC-AUC**

- **Data Visualization**: 
  - **Matplotlib**
  - **Seaborn**

## Dataset

The dataset contains transaction records, including both fraudulent and non-fraudulent transactions. Due to the nature of fraud detection problems, the dataset is highly imbalanced, with far fewer fraudulent transactions. The dataset was preprocessed and split into training, testing, and validation sets, with the training set balanced to improve model performance.

### Data Preprocessing:
- **Missing Data**: Handled through mean imputation for numerical features.
- **Feature Scaling**: Applied **RobustScaler** and **StandardScaler** to scale the features.
- **Class Imbalance Handling**: Employed **SMOTE**, undersampling, and class weights to balance the dataset.

## Models Developed

1. **Logistic Regression**  
   - Balanced between Precision and Recall, achieving an **F1-score of 0.90** on a balanced dataset.
  
2. **Shallow Neural Networks (using TensorFlow)**  
   - Used for detecting fraud with a focus on capturing non-linear patterns.
   - Achieved an **F1-score of 0.87** on the balanced dataset.
  
3. **Random Forest Classifier**  
   - A robust model for handling imbalanced datasets, with an **F1-score of 0.89** on the balanced dataset.

4. **Gradient Boosting Classifier**  
   - High precision but lower recall, resulting in an **F1-score of 0.79** on the balanced dataset.
  
5. **Linear SVC**  
   - Achieved an **F1-score of 0.90** on the balanced dataset, with a strong recall and good precision.

## Results

### Balanced Dataset Performance:

| Model                 | Precision | Recall | F1-Score |
|-----------------------|-----------|--------|----------|
| Logistic Regression    | 0.97      | 0.83   | 0.90     |
| Shallow Neural Networks| 1.00      | 0.76   | 0.87     |
| Random Forest          | 0.97      | 0.82   | 0.89     |
| Gradient Boosting      | 1.00      | 0.65   | 0.79     |
| Linear SVC             | 0.98      | 0.83   | 0.90     |

### Imbalanced Dataset Performance:

| Model                 | Precision | Recall | F1-Score |
|-----------------------|-----------|--------|----------|
| Logistic Regression    | 0.82      | 0.58   | 0.68     |
| Shallow Neural Networks| 0.63      | 0.71   | 0.67     |
| Random Forest          | 0.79      | 0.46   | 0.58     |
| Gradient Boosting      | 0.54      | 0.54   | 0.54     |
| Linear SVC             | 0.07      | 0.96   | 0.17     |

## Conclusion

- **Best Performers**: Logistic Regression and Linear SVC showed the best performance overall, particularly in terms of balancing Precision and Recall, with both achieving **F1-scores of 0.90** on the balanced dataset.
- **Shallow Neural Networks**: High precision but lower recall, making them less effective for this particular problem when recall is prioritized.
- **Linear SVC**: Achieved very high recall (0.96) on the imbalanced dataset but suffered from very low precision, indicating a trade-off between false positives and false negatives.

## How to Run the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/fraud-detection.git
    cd fraud-detection
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the project:
    ```bash
    python fraud_detection.py
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This updated version has proper Markdown formatting using `#` for the various levels of headings. It’s ready to be added to your GitHub repository or project documentation. Let me know if you'd like to make any additional adjustments!
