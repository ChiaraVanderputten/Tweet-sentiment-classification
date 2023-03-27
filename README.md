# Tweet Sentiment Classification

The project aims to classify the sentiment of a tweet as positive or negative using the DSL2122 January dataset. The correlation between the sentiment of a tweet and the user who writes it is strong and was taken into account in building the model. The correlation between the sentiment's distribution of tweets and the days of the week was also explored.

The article discusses the importance of pre-processing data extracted online, especially from social networks, as it may contain unnecessary data and a large number of features that reduce the accuracy of predictions. Pre-processing steps included converting tweet text to lowercase, eliminating punctuation, numbers, user tags, and URLs, as well as lemmatization, stemming, and stopword elimination. TfidfVectorizer from sklearn was used to transform the pre-processed text into a feature vector, and three different vectorizers were used for tweet text, day of the week, and user. 


In the article three different models to predict sentiment based on the pre-processed features was tested: Logistic Regression, Random Forest Classifier, and LinearSVC. Including day of the week and user features in addition to text the F1 score improve from 0.77 to 0.82. Then performing a grid search the best hyperparameters for each model have been found.
