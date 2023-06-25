# financial-inclusion
Financial Inclusion describing if a person will likely or unlikely have a bank account in East Africa.
Financial Inclusion Prediction
This Python script performs a prediction on financial inclusion based on a dataset containing various features. It uses machine learning techniques, specifically a Random Forest Classifier, to predict whether an individual has a bank account or not. The script also generates a CSV file with the predicted results.

Requirements
Python 3.x
Libraries: pandas, numpy, matplotlib, seaborn, sklearn
Dataset
The script assumes the presence of the following CSV files in the same directory:

train.csv: Contains the training data.
test.csv: Contains the test data.
VariableDefinitions.csv: Provides the definitions of the variables used in the dataset.
Usage
Ensure that the required CSV files are in the same directory as the script.
Run the script using the Python interpreter.
Steps
Data Loading: The script reads the training and test data from the respective CSV files.
Data Exploration: Basic information about the training data is displayed, such as the shape, size, and column names.
Data Preprocessing: The script performs some data preprocessing steps, such as dropping unnecessary columns and handling missing values.
Data Visualization: Several data visualizations are created using matplotlib and seaborn to explore the dataset.
Data Encoding: Categorical variables are one-hot encoded, and numerical variables are scaled using the StandardScaler.
Train-Test Split: The dataset is split into training and testing sets using a 80:20 ratio.
Model Training: A Random Forest Classifier is trained on the training data.
Prediction: The trained model is used to predict the bank account status on the test data.
Output Generation: The predicted results are saved in a CSV file named "mark x.csv".
Result Analysis: Accuracy and F1-score are calculated to evaluate the model's performance.
Data Cleaning: Some data cleaning steps are applied to remove specific data points from the training data.
Final Output: The cleaned test data is used to generate the final CSV file with the predicted results.
For more details on the code implementation, please refer to the inline comments in the script.

Note: Ensure that all required libraries are installed before running the script.
