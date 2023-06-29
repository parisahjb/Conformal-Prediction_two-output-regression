# Conformal-Prediction_two-output-regression
Input parameters of TrainModel function: 

  

 'file_name': This argument specifies the name of the CSV file containing the dataset that will be used to train the machine learning model. The file should be in the same directory as the Python script that calls the TrainModel() function.(10 inputs + two targets[Stiffness, Strength]) 

'scaler': This argument specifies the type of scaling to be applied to the dataset. The default value is 'std', which stands for standard scaling. Other possible values include 'minmax' for min-max scaling and 'robust' for robust scaling. 

'method': This argument specifies the method to be used for uncertainty quantification. 'WCV+' and 'CV+' work better when outliers are present, while 'CV' is better when we remove outliers from our dataset. 

'confidence': This argument specifies the confidence level for the calculation of the prediction intervals. The default value is 0.95, which corresponds to a 95% confidence level. There is a 95% chance, the true values of test points fall within the PI. 

'test_size': This argument specifies the proportion of the dataset to be used for testing the machine learning model. The default value is 0.2, which corresponds to a 80/20 split between the training and testing datasets. 

'frac': This argument is used to randomly select a subset of the test points for which to plot the prediction intervals. If it is set to 1, it plots all the prediction intervals for test points. Setting a lower value for frac can help make the plot more readable by selecting a representative subset of test points to display. 

  

In summary, the TrainModel() function is used to train a machine learning model on a given dataset, with optional data preprocessing. The function returns the trained model and the associated prediction intervals for each test point. 

 
 

Input parameters of Solution function: 

 

The Solution function takes four arguments: file_name, want, and method. 

'file_name': This argument specifies the name of the CSV file containing the dataset that was used to train the model. 

'want': This argument specifies the name of the CSV file containing the test data for which prediction intervals are to be calculated. 

'method': This argument specifies the method that was used to train the model and will be used for uncertainty quantification in the prediction intervals. 

'confidence': Your desired confidence level. 

The Solution function reads in the trained model from the specified file, preprocesses the test data in want in the same way as the training data, and then uses the trained model to calculate prediction intervals for each test point using the specified method. 

The output of the Solution function can be downloaded as a CSV file containing the original test data and the corresponding prediction intervals of both targets for each test point. 

 
