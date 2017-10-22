REQUIRES:
nltk
numpy
random
pickle
sklearn
statistics
dbAccesser

VIEW SENTIMENT_MOD.PY

This sentiment mod is my half of a research project I did last semester.  The full project and research paper can be found in /Gustavus-Code/394FinalSamAndMichael.zip

This sentiment mod takes in a string and uses sentiment analysis to take an educated guess as to the sentiment of the string.  For example:
>>> sentiment_mod.sentiment("I love Chrissy")
('pos', 1.0)
The mod classifies the text as positive with a confidence rating of 100%.  The main classifier is made up of 6 NLTK classifiers that were trained on 10000 pos/neg tagged movie reviews.  The main classifier then runs the string on the 6 individual classifiers then uses voting to assign a classification and confidence.  The classifier itself is discussed more in depth in the research paper previously mentioned.

If you run in command line call sentiment("inputstring") and the classifier will make its best guess at your sentiment
