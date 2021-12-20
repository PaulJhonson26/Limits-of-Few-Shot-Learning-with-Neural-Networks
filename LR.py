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
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os
from sklearn.metrics import f1_score

# load the Stanford GloVe model


def main():
    Dtrain = open("TrainingData", 'r')
    Dlabels = open("TrainingLabels", 'r')
    Dlabels= np.loadtxt(Dlabels, delimiter="\n")
    ### Glove ###
    glove_filename = 'glove.6B.50d.txt'
    # Variable for data directory
    glove_path = os.path.abspath(glove_filename)
    word2vec_output_file = glove_filename + '.word2vec'
    glove2word2vec(glove_path, word2vec_output_file)
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    ######### Preprocessing Steps ##############

            ### Convert to Lower Case ###
    Dtrain  = lowerCase(Dtrain)
            ### Depunctuating ###
    Dtrain = depunctuate(Dtrain)
            ### Removing Stopwords ###
#    Dtrain  = removeStop(Dtrain)  ###Btw might be fixable but this thing takes ages!
            ### Lemmatizing ###
 #   Dtrain  = Lemma(Dtrain)
            ### Creating Ngrams ###
 #   Ngrams = createNgrams(Dtrain)
            ### Splitting Dtrain ###
    DtrainPos = splitData(1, Dtrain, Dlabels)
    print(DtrainPos)
    DtrainNeg = splitData(0, Dtrain, Dlabels)
    ######## Converting for LR input ########
            ### Ngrams ###
    Xtrain, Xtest, Ytrain, Ytest = NgramModel(Dtrain, Dlabels, 3) #Either Use Ngram or TFIDF not both
            ### TFIDFing
#    Xtrain, Xtest, Ytrain, Ytest = tfidfModel(Dtrain, Dlabels, 200)
            ### Glove ###
    # word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(glove_filename)
    # vectorizer = Word2VecVectorizer(model)
    #
    # Xtrain, Xtest, Ytrain, Ytest = train_test_split(Dtrain, [1] * len(DtrainPos) + [0] * len(DtrainNeg), test_size=0.50)
    # for i, sent in enumerate(Xtrain):
    #     avg = sentenceAvg(sent, word_to_vec_map)
    #     #print(avg)
    #     Xtrain[i] = avg
    # #Get the sentence embeddings for the train dataset
    # #Xtrain = vectorizer.fit_transform(Xtrain)
    # Get the sentence embeddings for the test dataset
    # Xtest = vectorizer.transform(Xtest)
            ### Glvoe End ###
    clf = LogisticRegression(random_state=0, max_iter=10000).fit(Xtrain, Ytrain)
    y_pred = clf.predict(Xtest)

    print('Final Logistic Regression Accuracy: %s' % accuracy_score(y_pred, Ytest))
    print('Final Logistic Regression F1 score: %s' % f1_score(y_pred, Ytest))

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
    vectorizer1 = CountVectorizer(ngram_range=(1, 5))

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

def sentenceAvg(sentence, wordVectorMap):
    words = sentence.lower().split()
    avg = np.zeros(50)
    for w in words:
        if w in wordVectorMap:
            avg+= wordVectorMap[w]
    avg = avg/ len(words)
    return avg


def read_glove_vecs(glove_file):
    with open(glove_file, encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map
class Word2VecVectorizer:
  def __init__(self, model):
    print("Loading in word vectors...")
    self.word_vectors = model
    print("Finished loading in word vectors")

  def fit(self, data):
    pass

  def transform(self, data):
    # determine the dimensionality of vectors
    v = self.word_vectors.get_vector('king')
    self.D = v.shape[0]

    X = np.zeros((len(data), self.D))
    n = 0
    emptycount = 0
    for sentence in data:
      tokens = sentence.split()
      vecs = []
      m = 0
      for word in tokens:
        try:
          # throws KeyError if word not found
          vec = self.word_vectors.get_vector(word)
          vecs.append(vec)
          m += 1
        except KeyError:
          pass
      if len(vecs) > 0:
        vecs = np.array(vecs)
        X[n] = vecs.mean(axis=0)
      else:
        emptycount += 1
      n += 1
    print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
    return X


  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)
if __name__ == "__main__":
    main()