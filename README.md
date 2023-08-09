# Snappfood Sentiment Analysis using Naive Bayes Classifier
This project aims to perform sentiment analysis on Snappfood user comments using a Naive Bayes Classifier built from scratch. The dataset consists of 70,000 comments labeled as either "Happy" (Positive) or "Sad" (Negative).

## Project Steps
**1. Loading the Dataset**
The dataset containing Snappfood user comments with their corresponding labels is provided.

**2. Data Preprocessing**
Perform the following text preprocessing steps using the hazm library for Persian language processing:
Remove HTML tags, URLs, and non-alphanumeric characters from the dataset using regex functions.
Remove stopwords using the hazm stopwords list.
Normalize the text by converting it to lowercase and removing extra spaces.
Tokenize the text into individual words or tokens.
Apply stemming or lemmatization to reduce words to their base form.
Remove any remaining punctuation marks or special characters.

**3. Encoding Labels and Making Train-Test Splits**
Encode the labels "Happy" and "Sad" as numeric values (e.g., 1 and 0) for ease of processing.
Split the preprocessed dataset into a training set and a test set, typically using an 80:20 or 70:30 ratio.

**4. Building the Naive Bayes Classifier from Scratch**
Implement the Naive Bayes algorithm to build a sentiment classifier:
Calculate the prior probabilities of each class (Happy and Sad) by counting the number of occurrences of each class in the training set.
Calculate the likelihood probabilities of each word occurring in each class by counting the occurrences of each word in the corresponding class.
Apply Laplace smoothing to handle unseen words in the test set.

**5. Fitting the Model on Training Set and Evaluating Accuracies on the Test Set**
Apply the trained Naive Bayes classifier to the test set:
Preprocess the test comments using the same preprocessing steps as in the training phase.
Classify each test comment using the Naive Bayes formula and the probabilities obtained from the training phase.
Evaluate the accuracy of the model by comparing the predicted labels with the true labels of the test set.
Calculate additional metrics such as precision, recall, and F1-score to assess the performance of the classifier.
## Dependencies
The project uses the hazm library for text preprocessing in Persian language.
Other dependencies include pandas, numpy, scikit-learn, and nltk.

## Contributing
If you would like to contribute to this project, please follow the standard guidelines for contributions, such as opening an issue for bug fixes or feature requests and creating a pull request for proposed changes.
