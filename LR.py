import numpy as np
import nltk
from nltk.corpus import wordnet
import string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def main():
    Dtrain = open("TrainingData", 'r')
    Dlabels = open("TrainingLabels", 'r')
    Dlabels= np.loadtxt(Dlabels, delimiter="\n")

    ######### Preprocessing Steps ##############

            ### Convert to Lower Case ###
    Dtrain  = lowerCase(Dtrain)
            ### Depunctuating ###
    Dtrain = depunctuate(Dtrain)
            ### Splitting Dtrain ###
    DtrainPos = splitData(1, Dtrain, Dlabels)
    DtrainNeg = splitData(0, Dtrain, Dlabels)
            ### Converting for LR input ###
    Dtrain = np.concatenate((np.array(DtrainPos), np.array(DtrainNeg)), axis=0)
    Dtrain = Dtrain.tolist()
    vectorizer1 = CountVectorizer()
    vectorizer1.fit(Dtrain)
    vector1 = vectorizer1.transform(Dtrain)
    X = vector1.toarray()

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, [1] * len(DtrainPos) + [0] * len(DtrainNeg), test_size=0.30)
            ### LR Model ###
    clf = LogisticRegression(random_state=0, max_iter=10000).fit(Xtrain, Ytrain)
    y_pred = clf.predict(Xtest)
    print('Logistic Regression Accuracy: %s' % accuracy_score(y_pred, Ytest))

def lowerCase(StringArray):
    lowerArray = []
    for i in StringArray:
        lowerArray.append(i.lower())
    return lowerArray
def depunctuate(StringArray):
    exclude = set(string.punctuation)
    depStringArray = []
    for i in StringArray:
        s = ''.join(ch for ch in i if ch not in exclude)
        depStringArray.append(s)
    return depStringArray

def splitData(Div, StringArray, Labels):
    splitArray = []
    for i, x in enumerate(StringArray):
        if Labels[i] == Div:
            splitArray.append(x)
    return splitArray

if __name__ == "__main__":
    main()