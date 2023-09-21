![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/88fdba06-5be9-46fc-8115-d5f2768db485)
In the beginning of the code, we will be importing related libraries that is the basic libraries, 
preprocessing libraries, model training libraries, under sampling and oversampling libraries 
and lastly libraries for checking accuracy.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/5e905940-c0bb-4b75-9f6c-fb4cb7a3be85)
In this line of code, it reads CSV file named ‘onlinefraud.csv’ using the pandas library and 
assigns the result data to a variable called ‘Data’. We can see that the data was displayed the 
first five rows of the ‘Data’ Data Frame. 

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/179735ff-209f-45d4-8fb6-58d6e478b37d)
The code executes the data type for each of the features

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/0e4bca08-dafa-4d05-a8cd-7cf5de93bcfe)
Data.isnull().sum() checks if there are any null values in our dataset

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/8cb63158-f27b-4e98-be49-8be3321a773e)
This code filters the 'data' Data Frame to only include rows where the value in the 'isFraud' 
column is equal to 1. By passing this boolean mask as an index to the 'data' Data Frame, 
data[data['isFraud']==1] returns a new Data Frame that contains only the rows where 
'isFraud' is equal to 1. This effectively filters the dataset to show only the rows associated 
with fraudulent transactions or activities

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/5c7e0b91-efaf-457c-b480-7121f904dc93)
This code creates a countplot using the seaborn library to visualize the distribution of the 
'type' column in the 'data' Data Frame. We saw that the cash_out type has the highest count 
and Debit type has the lowest count.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/b5300d34-af4b-4aca-b54c-3b5a31746e2e)
This code creates a countplot using the seaborn library to visualize the distribution of the 
'isFraud' column in the 'data' DataFrame. This plot will show the count of fraud and nonfraud instances in the 'isFraud' column of the dataset.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/856f9333-e6a8-4468-9bd3-9f75424cde0a)
From the code above, we got to figure out the number of datasets that are ‘fraud’ and ‘not 
fraud’. As we can see, the data is imbalanced since the amount of data that is fraud (8213) is 
so little compared to data that is not fraud (6 354 407)

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/8c6d0254-663a-4d20-9ad7-8bc1609ab604)
This code has the same function as the previous code, it only outputs the number of datasets 
that are ‘fraud’ and ‘not fraud’ in percentage form.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/b0461958-48be-4afb-8d98-35cd6843b8d9)
This code is for selecting all the columns that have numerical values for us to check the 
outliers and skewness of the values and insert it into a list named numerical.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/d12eb030-de11-42f0-99fd-b263cd03f517)
This code is for creating boxplots which use to check outliers in each column in the numerical 
list from the previous code. From the boxplot that we have created, we can see the skewness 
of the values in the graph.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/be4a591a-da57-4128-a381-ad742c25689c)
This code is to drop all the data that we do not need.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/cb456660-98ce-4f0f-a051-a97cb9abf2ae)
This code is utilizing the one hot encoding to represent the type of transaction method that 
the users have used ie. cash out, debit, payment, and transfer by changing it from string to 
one hot vectors as a method of classification into fixed classes of the type of payment mode.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/764cf813-bdad-4a16-8ece-0e791a1a2310)
The code above shows that we use RobustScaler to standardise our data. The way it works 
is by removing and replacing the outliers. It replaces the outliers with the mean value, or the 
average of the class that it wants to take. Our flaw is that we could not use our data that we 
have obtained as raw as it is due to the presence of outliers. Hence is why we use 
RobustScaler to make our data more presentable and also helps to class our data by range. 
This is so that it is easier for us to implement our data in our model.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/9a834ae8-7a6c-453e-a412-35ac0d1d9786)
Here we will be splitting the datasets into two parts x and y, where x contains all the columns 
except for ‘isFraud’, and y contains only the ‘isFraud’ column.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/5717d1b5-d2f5-4f22-acbb-fc6dab3651ee)
We use this line of code to check the columns that are contained inside the variable x and we 
can see that the column ‘isFraud’ has been dropped.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/0d81cce6-a2cc-4bd4-b3ba-152455eafe7d)
Here we want to display the filtered datasets by selecting rows where ‘isFraud’ column value 
is 1, indicating fraud.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/78058f0f-b550-4a89-a6f4-5468bc58eef1)
This part of the code imports the ExtraTreeRegressor class from the skleanr.ensemble 
module. We then make an instance of the class and fitted the model with x and y data to train 
it. Then, we printed the feature importance values of each feature so that we can see the 
distribution of contributions by each feature.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/dc0a635d-f4ca-4276-9474-276e7115ec3f)
We then store the feature importance attributes as pandas Series with feature names as the 
index in a variable name feat_importances. Now we want to make a horizontal bar plot with 
the feature_importances, we use nlargest method with parameter 10 to retrieve top 10 
values from the feature Series we made based on their magnitude in this case being their 
importance values.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/a96be147-1946-44ad-97be-8905ea7a4cd5)
Now we will be implementing the train test split technique to train our data. We split the x 
and y data into training (X_train, y_train) and testing (X_test, y_test) sets. Then, we will be 
making an instance of the StratifiedKFold class and pass in the necessary parameters.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/848733ed-72e8-4da9-9abd-d190e91a36ac)
We then proceed with making an instance of the LogisticRegression class as we are going to 
be implementing a randomized search for hyperparameters tuning of a Logistic Regression 
model using cross-validation. We need to specify the hyperparameters we are going to used, 
so we initialized variable param with a range of ‘C’ value from 0.1 to 10 so the 
hyperparameter tuning process can explore different magnitude of ‘C’. We then make an 
instance of the RandomizedSearchCV class we are going to be using and passing in the 
parameters necessary. Finally, we then fitted our randomize search model with the X_train 
and y_train data.

![image](https://github.com/AimanZaharin/CreditCardFraudDetectionProject/assets/92364588/6a558709-5ed3-4d9e-8d42-d6c191e26c2a)
After finishing training our model with the dataset, we are going to use the predict method 
on our trained model to make a prediction on the test dataset (X_test) and store the 
prediction in y_pred. For us to analyse the results of our data analytics process, we will print 
three things that will take y_test and y_pred as its parameters: confusin matrix, accuracy 
score and classification report.













































