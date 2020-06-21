import pandas as pd 
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
 
warnings.filterwarnings("ignore")


def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def featureEngineer(sentiments):
	htmlSyntax = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
	sentiments['review'] = sentiments['review'].apply(lambda x: re.sub('\ |\!|\/|\;|\:|\=|\"|\(|\)|\:|\]|\[|\<|\>|\{|\}|\'|\?|\.|\,|\|',' ', str(x)))
	sentiments['review'] = sentiments['review'].apply(lambda x: re.sub('\d+',' ', str(x)))	
	sentiments['review'] = sentiments['review'].apply(lambda x: re.sub(htmlSyntax,' ', str(x)))	
	sentiments['review'] = sentiments['review'].apply(lambda x: re.sub('\s+',' ',str(x).strip().lower()))
	sentiments['review'] = sentiments['review'].apply(lambda x: word_tokenize(x))
	sentenceList = []
	sentenceListTest = []
	for i in sentiments['review']:
		wordList = ''	
		for k in i:
			if k not in stopwordsList:
				wordList = wordList + reduce_lengthening(lemmatizer.lemmatize(k)) + ' '
		sentenceList.append(wordList.strip().lower())
	#wordList.clear()

	sentiments['review'] = pd.Series(sentenceList)
	sentiments['review'] = sentiments['review'].apply(lambda x: re.sub('\s+', ' ', str(x).strip()))
	return sentiments


sentimentData = pd.read_csv('IMDB Dataset.csv', header = 0)

stopwordsList = set(list(set(stopwords.words('english')))+['br'])
# print(list(stopwordsList))
lemmatizer = WordNetLemmatizer()

t0 = time.time()
sentimentData = featureEngineer(sentimentData)
t1 = time.time()
print('Time Take is:', t1 - t0)
sentimentData['sentiment'] = sentimentData['sentiment'].map({'positive': 1, 'negative': 0})

dataset_train, dataset_test, train_data_label, test_data_label = train_test_split(sentimentData['review'], sentimentData['sentiment'], test_size=0.20)

train = pd.DataFrame(dataset_train)
test = pd.DataFrame(dataset_test)

train.columns = ['review']
test.columns = ['review']
concatenatedReview = pd.concat([train['review'], test['review']])
vectorizer = TfidfVectorizer(strip_accents = 'unicode', analyzer = 'word',token_pattern=r'\w{1,}',ngram_range = (1,3), stop_words = 'english', sublinear_tf = True, max_features = 20000)
transformer = vectorizer.fit(concatenatedReview)

train_ngrams = transformer.transform(train['review'])
test_ngrams = transformer.transform(test['review'])

print(test_ngrams.shape)
parameters = {'C':np.logspace(-10, 3,20, endpoint=True, base = 2.71828).tolist()}
model = LinearSVC(penalty = 'l2', max_iter = 10000, class_weight = 'balanced')
clf = GridSearchCV(model, parameters, n_jobs = -1)
clf.fit(test_ngrams, test_data_label)
print('Optimal Lambda: ',clf.best_params_)

classifier = LinearSVC(C = clf.best_params_['C'], penalty = 'l2', max_iter = 10000, class_weight = 'balanced').fit(train_ngrams, train_data_label)
predictedValue = classifier.predict(test_ngrams)
print(classifier.score(train_ngrams, train_data_label))
print(classifier.score(test_ngrams, test_data_label))

tn, fp, fn, tp = confusion_matrix(test_data_label, predictedValue).ravel()
sensitivity = tp/(tp+fn) 
Specificity = tn/(tn+fp)
fpr, tpr, threshold = roc_curve(test_data_label, predictedValue)
rc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % rc_auc)
plt.legend(loc = 'lower right')	
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print('True Negative',tn, 'False Positive',fp, 'False Negative',fn, 'True Positive',tp)
print('Sensitivity', sensitivity, 'Specificity', Specificity)

