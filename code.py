import pandas as pd
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from scipy.sparse import csr_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import re
from sklearn import preprocessing
import string
from scipy.sparse import hstack
from wordcloud import WordCloud, STOPWORDS
pd.options.mode.chained_assignment = None  # default='warn'


# ANALYSIS ON TWEET DISTRIBUTION OVER MONTH
def plot_tweet_over_month(sample, date_list, sentiment_list):
    # set height of bar
    month_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    month_neg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(len(sample)):
        if "JAN" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                month_pos[0] += 1
            else:
                month_neg[0] += 1

        if "FEB" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                month_pos[1] += 1
            else:
                month_neg[1] += 1

        if "MAR" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                month_pos[2] += 1
            else:
                month_neg[2] += 1

        if "APR" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                month_pos[3] += 1
            else:
                month_neg[3] += 1

        if "MAY" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                month_pos[4] += 1
            else:
                month_neg[4] += 1

        if "JUN" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                month_pos[5] += 1
            else:
                month_neg[5] += 1

        if "JUL" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                month_pos[6] += 1
            else:
                month_neg[6] += 1

        if "AUG" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                month_pos[7] += 1
            else:
                month_neg[7] += 1

        if "SEPT" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                month_pos[8] += 1
            else:
                month_neg[8] += 1

        if "OCT" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                month_pos[9] += 1
            else:
                month_neg[9] += 1

        if "NOV" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                month_pos[10] += 1
            else:
                month_neg[10] += 1

        if "DEC" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                month_pos[11] += 1
            else:
                month_neg[11] += 1

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(month_pos))
    br2 = [x + barWidth for x in br1]
    # Make the plot
    plt.bar(br1, month_pos, color='r', width=barWidth,
            edgecolor='grey', label='POSITIVE TWEET')
    plt.bar(br2, month_neg, color='g', width=barWidth,
            edgecolor='grey', label='NEGATIVE TWEET')
    # Adding Xticks
    plt.xlabel('Month', fontweight='bold', fontsize=15)
    plt.ylabel('#tweet', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(month_pos))],
               ['JAN', 'FEB', 'MAR', 'APR', 'MAY','JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DIC'])
    plt.legend()
    plt.show()


# ANALYSIS ON TWEET DISTRIBUTION OVER DAYS OF THE WEEK
def plot_tweet_over_dof(sample, date_list, sentiment_list):
    # set height of bar
    day_pos = [0, 0, 0, 0, 0, 0, 0]
    day_neg = [0, 0, 0, 0, 0, 0, 0]

    for i in range(len(sample)):
        if "MON" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                day_pos[0] += 1
            else:
                day_neg[0] += 1

        if "TUE" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                day_pos[1] += 1
            else:
                day_neg[1] += 1

        if "WED" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                day_pos[2] += 1
            else:
                day_neg[2] += 1

        if "THU" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                day_pos[3] += 1
            else:
                day_neg[3] += 1

        if "FRI" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                day_pos[4] += 1
            else:
                day_neg[4] += 1

        if "SAT" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                day_pos[5] += 1
            else:
                day_neg[5] += 1

        if "SUN" in str(date_list[i]).upper():
            if sentiment_list[i] == 1:
                day_pos[6] += 1
            else:
                day_neg[6] += 1

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(day_pos))
    br2 = [x + barWidth for x in br1]
    # Make the plot
    plt.bar(br1, day_pos, color='r', width=barWidth,
            edgecolor='grey', label='POSITIVE TWEET')
    plt.bar(br2, day_neg, color='g', width=barWidth,
            edgecolor='grey', label='NEGATIVE TWEET')
    # Adding Xticks
    plt.xlabel('Month', fontweight='bold', fontsize=15)
    plt.ylabel('#tweet', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(day_pos))],
               ['MON', 'TUE', 'WED', 'THU', 'FRI','SAT', 'SUN'])
    plt.legend()
    plt.show()


# USERS ANALYSIS
def scatter_plot_user_sentiment(sample, user_list, sentiment_list):
    users = user_list
    value = []

    for i in range(len(users)):
        value.append([0, 0])  # positive, negative

    user_pos_neg = dict(zip(users, value))

    for i in range(len(sample)):
        if sentiment_list[i] == 1:  # positive
            user_pos_neg[str(user_list[i])][0] += 1
        else:
            user_pos_neg[str(user_list[i])][1] += 1  # negative

    x = []
    y = []

    for user in user_pos_neg.keys():
        x.append(user_pos_neg[str(user)][0])
        y.append(user_pos_neg[str(user)][1])

    plt.xlabel('#positive', fontweight='bold', fontsize=15)
    plt.ylabel('#negative', fontweight='bold', fontsize=15)
    plt.scatter(x, y, c ="blue")
    plt.show()


# WORDCLOUD

def show_word_cloud(text_list):
    bigstring = " ".join(tweet for tweet in text_list)
    plt.figure(figsize=(12, 12))
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',collocations=False,width=1200,height=1000).generate(bigstring)
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()


# READ AND LOAD DATASET
dev = pd.read_csv("development.csv")
eval = pd.read_csv("evaluation.csv")
dataset = dev.drop_duplicates(ignore_index=True, subset=['ids']) # remove duplicates by tweet ids

df = pd.concat([dataset, eval], sort=False, ignore_index=True)
df['dof'] = ""
for i in range(len(df)):
    df['dof'][i] = (str(df['date'][i]).split()[0]).lower()


train_valid_mask = ~df["sentiment"].isna()

dataset = df[train_valid_mask]
eval = df[~train_valid_mask]


# DATA EXPLORATION
# PLOT ON THE ORIGINAL DATASET
#plot_tweet_over_month(dev, dev['date'], dev['sentiment'])
#plot_tweet_over_dof(dev, dev['date'], dev['sentiment'])
#scatter_plot_user_sentiment(dev, dev['user'], dev['sentiment'])
#show_word_cloud(dev['text'])

# PLOT ON A 30% SAMPLE OF THE ORIGINAL DATASET
#plot_tweet_over_month(dev_sample, dev_sample['date'], dev_sample['sentiment'])
#plot_tweet_over_dof(dev_sample, dev_sample['date'], dev_sample['sentiment'])
#scatter_plot_user_sentiment(dev_sample, dev_sample['user'], dev_sample['sentiment'])
#show_word_cloud(dev_sample['text'])

del dataset['date']
del dataset['flag']
del dataset['ids']
del eval['date']
del eval['flag']
del eval['ids']

# PREPROCESSING

dataset['text'] = dataset['text'].str.lower()  # lowercasing text
eval['text'] = eval['text'].str.lower()  # lowercasing text

# remove punctuations
punctuations_list = string.punctuation


def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


dataset['text']= dataset['text'].apply(lambda x: cleaning_punctuations(x))
eval['text']= eval['text'].apply(lambda x: cleaning_punctuations(x))


# remove URL
def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)


dataset['text'] = dataset['text'].apply(lambda x: cleaning_URLs(x))
eval['text'] = eval['text'].apply(lambda x: cleaning_URLs(x))


# remove numbers
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)


dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))
eval['text'] = eval['text'].apply(lambda x: cleaning_numbers(x))

# Getting tokenization of tweet text

dataset['text'] = dataset['text'].apply(lambda x: x.split())
eval['text'] = eval['text'].apply(lambda x: x.split())

# Applying Stemming
st = nltk.PorterStemmer()


def stemming_on_text(data):
    return  [st.stem(word) for word in data]


dataset['text'] = dataset['text'].apply(lambda x: stemming_on_text(x))
eval['text'] = eval['text'].apply(lambda x: stemming_on_text(x))

# Applying Lemmatizer
lm = nltk.WordNetLemmatizer()


def lemmatizer_on_text(data):
    return " ".join([lm.lemmatize(word) for word in data])


dataset['text'] = dataset['text'].apply(lambda x: lemmatizer_on_text(x))
eval['text'] = eval['text'].apply(lambda x: lemmatizer_on_text(x))

# Stopwords
additional = ['rt', 'rts', 'retweet', 'amp', 'quot','lol','im','go','day','wa','thi','u']
stop = set().union(stopwords.words('english'),additional)


def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop])


dataset['text'] = dataset['text'].apply(lambda text: cleaning_stopwords(text))
eval['text'] = eval['text'].apply(lambda text: cleaning_stopwords(text))


positive = dataset[dataset['sentiment'] == 1]
negative = dataset[dataset['sentiment'] == 0]

#show_word_cloud(positive['text'])
#show_word_cloud(negative['text'])


# SPLIT DATASET
def split_dataset(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('dataset split done')
    return X_train, X_test, y_train, y_test


# MODEL SELECTION

def model_Evaluate(model, X_test, y_test):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)


# Plot the ROC-AUC Curve for model
def plot_ROC_AUC(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC CURVE')
    plt.legend(loc="lower right")
    plt.show()


# SPLIT DATASET FOR EVALUATE OUR MODELS
#X_train, X_test, y_train, y_test = split_dataset(dataset['text'], dataset['sentiment'])

# FEATURE ENGINEERING
vectorizer = TfidfVectorizer(ngram_range=(1,2))
vectorizer2 = TfidfVectorizer()
vectorizer3 = TfidfVectorizer()
train_text = vectorizer.fit_transform(dataset['text'])
train_dof = vectorizer2.fit_transform(dataset['dof'])
train_user = vectorizer3.fit_transform(dataset['user'])
X_train = hstack([train_text, train_dof,train_user])

test_text = vectorizer.transform(eval['text'])
test_dof = vectorizer2.transform(eval['dof'])
test_user = vectorizer3.transform(eval['user'])

X_test = hstack([test_text, test_dof,test_user])


# HYPERPARAMETER TUNING FOR TFIDFVECTORIZER AND FOR MODEL

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC()), #LogisticRegression()),
])
parameters = {
    #'tfidf__max_df': [0.4, 0.5],
    #'tfidf__min_df': [0.001, 0.002, 0.003, 20],
    'tfidf__ngram_range': [(1,1),(1, 2)],
    #'clf__max_depth': [10, 20, 30, 50],
    #'clf__min_samples_leaf': [10, 20, 30],
    #'clf__min_samples_split': [5, 8, 10, 12],
    #'clf__n_estimators': [100],
    'tfidf__max_features':[1000,2000,5000,10000,20000],
    'clf__C': [1],
    #'clf__max_iter': [15000],
    'clf__random_state': [42]
}

#grid_search_tune = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=3)
#grid_search_tune.fit(X_train2, y_train2)

#print("Best parameters set:")
#print(grid_search_tune.best_estimator_.steps)


# GRID SEARCH AND EVALUATION PREDICT

# LOGISTIC REGRESSION

# HYPERPARAMETERS FOR LOGISTIC REGRESSION

solver = ['liblinear']
C = [0.5,5,10,100]
random_state = [42]
multi_class = ['ovr']
max_iter = [10,30,300,2000]
# Create the random grid
random_lr_grid = { 'C': C,
                   'multi_class': multi_class,
                    'solver': solver,
                   'random_state': random_state,
                   'max_iter': max_iter
                    }

#lr = LogisticRegression()
#lr_random = GridSearchCV(refit=True, estimator=lr, param_grid=random_lr_grid, cv=3, verbose=3, scoring='f1_macro', n_jobs= -1)
#lr_random.fit(X_train,y_train)
#print(lr_random.best_estimator_,lr_random.best_params_)
#y_pred_lr = lr_random.predict(X_test)

#model_Evaluate(lr_random, X_test, y_test)
#plot_ROC_AUC(y_test, y_pred_lr)



# LINEAR SVC

# HYPERPARAMETER FOR LINEAR SVC
C = [0.05, 0.6, 1, 5]
multi_class = ['ovr']
random_state = [42]
# Create the random grid
random_svc_grid = {'C': C,
                    'multi_class': multi_class,
                   'random_state': random_state,
                    }


#lsvc = LinearSVC()
#lsvc_random = GridSearchCV(refit=True, estimator=lsvc, param_grid=random_svc_grid, cv=3, verbose=3, scoring='f1_macro', n_jobs= -1)
#lsvc_random.fit(X_train,y_train)
#print(lsvc_random.best_estimator_,lsvc_random.best_params_)
#y_pred_lsvc = lsvc_random.predict(X_test)

#model_Evaluate(lsvc_random, X_test, y_test)
#plot_ROC_AUC(y_test, y_pred_lsvc)


lsvc_best = LinearSVC(C=0.6,random_state=42)
lsvc_best.fit(X_train,dataset['sentiment'])  # dataset['sentiment'])
y_pred_lsvc_best = lsvc_best.predict(X_test)
#model_Evaluate(lsvc_best, X_test, y_test)

#plot_ROC_AUC(y_test, y_pred_lsvc_best)

# RANDOM FOREST CLASSIFIER

# HYPERPARAMETER FOR RANDOM FOREST
# Number of trees in random forest
n_estimators = [50,250,500] #int(x) for x in np.linspace(start = 200, stop = 600, num = 5)]
# Number of features to consider at every split
max_features = ['sqrt' , 'log2']
# Maximum number of levels in tree
max_depth = [50,500,1000] # int(x) for x in np.linspace(60, 110, num = 5)]
#max_depth.append(None)
# Minimum number of samples required to split a node
#min_samples_split = [5, 10,15]
# Minimum number of samples required at each leaf node
#min_samples_leaf = [4,8]
# Method of selecting samples for training each tree
#bootstrap = [True]  #, False]
# Create the random grid
random_rf_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
 #              'min_samples_split': min_samples_split,
  #             'min_samples_leaf': min_samples_leaf,
   #            'bootstrap': bootstrap
                              }

#rf = RandomForestClassifier(max_depth=500, n_estimators=500, max_features='sqrt')
#rf_random = GridSearchCV(refit=True, estimator=rf, param_grid=random_rf_grid, cv=3, verbose=3, scoring='f1_macro', n_jobs= -1)
#rf.fit(X_train, y_train)
#print(rf_random.best_estimator_, rf_random.best_params_, rf_random.cv_results_)
#y_pred_rf = rf.predict(X_test)

#model_Evaluate(rf_random, X_test, y_test)
#plot_ROC_AUC(y_test, y_pred_rf)

# LOAD FINAL EVALUATION

results = pd.DataFrame(y_pred_lsvc_best)
results.to_csv('file.csv', index_label="Id", header=['Predicted'])