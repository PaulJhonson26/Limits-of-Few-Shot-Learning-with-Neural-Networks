import numpy as np
import nltk
from nltk.corpus import wordnet
import string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    Dtrain = open("TrainingData", 'r')
    Dlabels = open("TrainingLabels", 'r')
    Dlabels= np.loadtxt(Dlabels, delimiter="\n")

    ######### Preprocessing Steps ##############

            ### Convert to Lower Case ###
    Dtrain  = lowerCase(Dtrain)
            ### Depunctuating ###
    Dtrain = depunctuate(Dtrain)
            ### Removing Stopwords ###
    Dtrain  = removeStop(Dtrain)  ###Btw might be fixable but this thing takes ages!
            ### Lemmatizing ###
    Dtrain  = Lemma(Dtrain)
            ### Creating Ngrams ###
    Ngrams = createNgrams(Dtrain)
            ### Splitting Dtrain ###
    DtrainPos = splitData(1, Dtrain, Dlabels)
    DtrainNeg = splitData(0, Dtrain, Dlabels)
    ######## Converting for LR input ########
            ### Ngrams ###
    Xtrain, Xtest, Ytrain, Ytest = NgramModel(Dtrain, Dlabels, 200) #Either Use Ngram or TFIDF not both
            ### TFIDFing
#    Xtrain, Xtest, Ytrain, Ytest = tfidfModel(Dtrain, Dlabels, 200)
    clf = LogisticRegression(random_state=0, max_iter=10000).fit(Xtrain, Ytrain)
    y_pred = clf.predict(Xtest)

    print('Final Logistic Regression Accuracy: %s' % accuracy_score(y_pred, Ytest))

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
def removeStop(StringArray):
    tokens = []
    for i in StringArray: #### Tokenize each sentence
        tokens.append(word_tokenize(i))
    tokensRS = []
    for i in tokens: #### Remove stop words
        tokensRS.append([word for word in i if not word in stopwords.words()])
    reString = []
    for i in tokensRS: ### put back into string format
        reString.append((" ").join(i))
    return reString

def createNgrams(StringArray):
    tokensArray = []
    for i in StringArray:
        tokensArray.append(word_tokenize(i))
    Ngrams = []
    for i in tokensArray:
        Ngrams.append(list(ngrams(i, 3)))
    for i in Ngrams:### Fixing an issue where there were a lot of empty entries
        if i ==  "":
            Ngrams.remove(i)
    return Ngrams

def Lemma(Dtrain):
    nltk.download("averaged_perceptron_tagger")
    nltk.download("wordnet")
    lemmatizer = nltk.WordNetLemmatizer()
    tokensArray = []
    for i in Dtrain:
        tokensArray.append(word_tokenize(i))
    Lemmatized = []
    for k, s in enumerate(tokensArray):
        Sentence = []
        for w in s:
            Sentence.append(lemmatizer.lemmatize(w, get_wordnet_pos(w)))
            NTsentence = (" ").join(Sentence)
        Lemmatized.append(NTsentence)
    return Lemmatized
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def NgramModel(Dtrain, Dlabels, iter):
    BestX = []
    bestPred = 0
    DtrainPos = splitData(1, Dtrain, Dlabels)
    DtrainNeg = splitData(0, Dtrain, Dlabels)
    ### Converting for LR input ###
    Dtrain = np.concatenate((np.array(DtrainPos), np.array(DtrainNeg)), axis=0)
    Dtrain = Dtrain.tolist()
    vectorizer1 = CountVectorizer(ngram_range=(1, 7))

    for i in range(iter):
        vectorizer1.fit(Dtrain)
        vector1 = vectorizer1.transform(Dtrain)
        X = vector1.toarray()

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, [1] * len(DtrainPos) + [0] * len(DtrainNeg), test_size=0.20)
        ### LR Model ###
        clf = LogisticRegression(random_state=0, max_iter=10000).fit(Xtrain, Ytrain)
        y_pred = clf.predict(Xtest)
        predScore = accuracy_score(y_pred, Ytest)
       #print('Accuracy: %s' % accuracy_score(y_pred, Ytest))
        if predScore > bestPred:
            bestPred = predScore
            BestX = [Xtrain, Xtest, Ytrain, Ytest]
            print('Logistic Regression Accuracy: %s' % accuracy_score(y_pred, Ytest))

    return BestX
def tfidfModel(Dtrain, Dlabels, iter):
    BestX = []
    bestPred = 0
    DtrainPos = splitData(1, Dtrain, Dlabels)
    DtrainNeg = splitData(0, Dtrain, Dlabels)
    ### Converting for LR input ###
    Dtrain = np.concatenate((np.array(DtrainPos), np.array(DtrainNeg)), axis=0)
    Dtrain = Dtrain.tolist()
    for i in range(iter):
        tf_idf = TfidfVectorizer()
        tf_idf.fit(Dtrain)
        vector = tf_idf.transform(Dtrain)
        X = vector.toarray()
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, [1] * len(DtrainPos) + [0] * len(DtrainNeg), test_size=0.20)
        ### LR Model ###
        clf = LogisticRegression(random_state=0, max_iter=10000).fit(Xtrain, Ytrain)
        y_pred = clf.predict(Xtest)
        predScore = accuracy_score(y_pred, Ytest)
        # print('Accuracy: %s' % accuracy_score(y_pred, Ytest))
        if predScore > bestPred:
            bestPred = predScore
            BestX = [Xtrain, Xtest, Ytrain, Ytest]
            print('Logistic Regression Accuracy: %s' % accuracy_score(y_pred, Ytest))
    return BestX
if __name__ == "__main__":
    main()