import nltk
import numpy
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from dbAccesser import get_tweets

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers=classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf

documents_f = open("documents.pickle","rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("word_features5k.pickle","rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

#featuresets = [(find_features(rev),category) for (rev,category) in documents]
featuresets_save = open("featuresets.p","rb")
featuresets = pickle.load(featuresets_save)
featuresets_save.close()

classifier_f = open("classifier.p","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

MNB_classifier_f = open("MNB_classifier.p","rb")
MNB_classifier = pickle.load(MNB_classifier_f)
MNB_classifier_f.close()

B_classifier_f = open("B_classifier.p","rb")
B_classifier = pickle.load(B_classifier_f)
B_classifier_f.close()

LR_classifier_f = open("LR_classifier.p","rb")
LR_classifier = pickle.load(LR_classifier_f)
LR_classifier_f.close()

SGD_classifier_f = open("SGD_classifier.p","rb")
SGD_classifier = pickle.load(SGD_classifier_f)
SGD_classifier_f.close()

LinearSVC_classifier_f = open("LinearSVC_classifier.p","rb")
LinearSVC_classifier = pickle.load(LinearSVC_classifier_f)
LinearSVC_classifier_f.close()

NuSVC_classifier_f = open("NuSVC_classifier.p","rb")
NuSVC_classifier = pickle.load(NuSVC_classifier_f)
NuSVC_classifier_f.close()

voted_classifier = VoteClassifier(classifier,SGD_classifier,LinearSVC_classifier
                                  ,NuSVC_classifier, MNB_classifier,
                                  B_classifier, LR_classifier)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

def main():
    npAR = get_tweets()
    results = []
    for i in npAR:
        i_ascii = ''.join((lv for lv in i if ord(lv) < 128))
        
        #print(i_ascii, sentiment(i_ascii))
        results.append(sentiment(i_ascii))
    sumPos = 0
    weightedSumPos = 0
    for i in results:
        if i[0] == 'pos':
            sumPos = sumPos+1
            weightedSumPos = weightedSumPos + i[1]
    print("Percentage of positive tweets:",sumPos/1000)
    print("Weighted percentage of positive tweets:",weightedSumPos/1000)
    
    
main()
    

