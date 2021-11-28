import numpy as np
import pandas as pd
import string
from dale_chall import DALE_CHALL

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize

import nltk
nltk.download('stopwords')

dtypes = {"sentence": "string", "token": "string", "complexity": "float64"}

train = pd.read_excel('./data/train.xlsx', dtype=dtypes, keep_default_na=False)
test = pd.read_excel('./data/test.xlsx', dtype=dtypes, keep_default_na=False)

# Verificare - Train Data & Test Data
# print('train data: ', train.shape)
# print('test data: ', test.shape)

# Functie - Cuvantul se afla sau nu in lista DALE CHALL


def is_dale_chall(word):
    return int(word.lower() in DALE_CHALL)


# Functie - Prima litera a cuvantului este mica sau mare
def is_title(word):
    return int(word.istitle())


# Functie - Lungimea cuvantului
def length(word):
    return len(word)


# Functie - Numarul de vocale pentru cuvantul dat
def nr_vowels(word):
    vows = 'aeiouy'
    sum = 0

    for w in word:
        if w.lower() in vows:
            sum += 1

    return sum


# Functie - Numarul de consoane pentru cuvantul dat
def nr_cons(word):
    vows = 'aeiouy'
    sum = 0

    for w in word:
        if w.lower() not in vows:
            sum += 1

    return sum


# Functie - Numarul de silabe pentru cuvantul dat
'''
    Daca o litera este vocala, iar litera urmatoare este consoana, sum + 1 (Ex. Beautiful)
    Daca un cuvant se termina in "e" sau "le", iar lungimea cuvantului este mai mare decat 2, sum - 1 ()
        - Faptul ca acel cuvant se termina in e, o vocala, nu semnifica faptul unei noi silabe
    Daca ultima litera a cuvantului contine una dintre vocalele "aiouy", sum + 1
    Orice cuvant este format din minim o silaba
'''


def nr_syllables(word):
    sum = 0
    vowels = 'aeiouy'

    for w in range(len(word) - 1):
        if word[w] in vowels and word[w+1] not in vowels:
            sum += 1

    if word.endswith('e') or (word.endswith('le') and len(word) > 2):
        sum -= 1

    if word[len(word) - 1] in 'aiouy':
        sum += 1

    if sum == 0:
        sum = 1

    return sum


# Functie - Numarul de sinonime pentru cuvantul dat
def synsets(word):
    syns = wordnet.synsets(word)

    return len(syns)


# Functie - Categorisirea corpusurilor
# S-ar putea sa conteze daca un cuvant este din biomed, deoarece acesta are mai multe sanse sa fie un cuvant complex decat unul
# Care apartine "europarl"
def corpus_feature(corpus):
    corp_dict = {}
    corp_dict['bible'] = [1, 0, 0]
    corp_dict['europarl'] = [0, 1, 0]
    corp_dict['biomed'] = [0, 0, 1]
    return corp_dict[corpus]

# Functie - Reuniunea tuturor functiilor care au ca parametru un cuvant


def get_word_structure_features(word):
    features = []
    features.append(nr_syllables(word))
    features.append(is_dale_chall(word))
    features.append(length(word))
    features.append(nr_vowels(word))
    features.append(is_title(word))
    features.append(nr_cons(word))
    features.append(synsets(word))

    return np.array(features)


# ------------------ PREPROCESSING - Formatarea textului ------------------

# Functie - Eliminarea semnelor de punctuatie cu ajutorul librariei string
def removePunctuation(text):
    result = ""

    for t in text:
        if t not in string.punctuation:
            result += t

    return result


# Functie - Transformarea textului in lowercase
def lowercaseText(text):
    return text.lower()

# Functie - Eliminarea cuvintelor de legatura (Ex: me ,my, she, himself...)


def removeStopWords(tokens):
    filtered_words = [
        word for word in tokens if word not in stopwords.words('english')]

    return filtered_words

# ------------------ SENTENCE'S FEAUTRES ------------------

# Functie - Lungimea frazei


def sentenceLength(tokens):
    return len(tokens)


# Functie - Numarul de cuvinte care apartin listei DALE CHALL din fraza
def commonWords(words):
    sum = 0

    for word in words:
        if is_dale_chall(word):
            sum += 1

    return sum

# Functie - Numarul de sinonime care se regasesc in toata fraza


def synsetsSentence(words):
    sum = 0

    for word in words:
        sum += synsets(word)

    print(sum)
    return sum

# Functie - Numarul de cuvinte care nu fac parte din lista stopwords


def numberOfWords(initSen, presSen):
    return len(initSen) - len(presSen)

# Functie - Reuniunea tuturor functiilor care au legatura cu "sentence"


def get_sentence_details(sentence):
    # PreProcessing
    newSentence = removePunctuation(sentence)
    newSentence = lowercaseText(newSentence)

    # Tokenization
    tokens = word_tokenize(newSentence)
    initSen = tokens

    # Eliminarea stopwords din libraria nltk
    tokens = removeStopWords(tokens)

    senFeatures = []
    senFeatures.append(sentenceLength(tokens))
    senFeatures.append(commonWords(tokens))
    senFeatures.append(synsetsSentence(tokens))
    senFeatures.append(numberOfWords(initSen, tokens))

    return np.array(senFeatures)


# Reuniunea tuturor functiilor care au legatura fie cu corpus, sentence sau token intr-o singura lista
def featurize(row):
    word = row['token']
    all_features = []
    all_features.extend(corpus_feature(row['corpus']))
    all_features.extend(get_word_structure_features(word))
    all_features.extend(get_sentence_details(row['sentence']))
    return np.array(all_features)


# Crearea unei matrice care contine informatii despre fiecare linie din datele primite
def featurize_df(df):
    nr_of_features = len(featurize(df.iloc[0]))
    nr_of_examples = len(df)
    features = np.zeros((nr_of_examples, nr_of_features))
    for index, row in df.iterrows():
        row_ftrs = featurize(row)
        features[index, :] = row_ftrs
    return features


# Testare - Date reale
# X_train = featurize_df(train)
# y_train = train['complex'].values
# X_test = featurize_df(test)

# Testare - Date Fictive
X = train[['corpus', 'sentence', 'token']]
Y = train['complex'].values

X_tr = featurize_df(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_tr, Y, test_size=0.1, random_state=12)

# NAIVE
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

# Testare - Date Fictive
print(balanced_accuracy_score(y_test, y_pred))

cross_validation_scores = cross_val_score(gnb, X_train, y_train, cv=10)

sum = 0

for score in cross_validation_scores:
    sum += score

print(cross_validation_scores)

print(sum / 10)

conf_matrix = confusion_matrix(y_test, y_pred)

print(conf_matrix)

# Testare - Date reale - Creare CSV cu predictiile modelului
# df = pd.DataFrame()
# df['id'] = test.index + len(train) + 1
# df['complex'] = y_pred
# df.to_csv('submission.csv', index=False)
