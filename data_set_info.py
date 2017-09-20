import nltk
import time

from nltk import word_tokenize,sent_tokenize,pos_tag
# from nltk.corpus import stopwords

# write negative/positive words to a file
pos_bank = open('pos_word_bank.txt', 'w')
neg_bank = open('neg_word_bank.txt', 'w')

# numerical ranking of reviews
corpus_rankings = open("sentiment_labels.txt", "r").read()
corpus_rankings_delimited = corpus_rankings.split('\n') # split on entries

# text of the reviews
corpus_reviews = open("dictionary.txt", "r").read()
corpus_reviews_delimited = corpus_reviews.split('\n') # split on entries

# crash if we have bad data
# assert(len(corpus_reviews_delimited) == len(corpus_rankings_delimited))

# add rankings
print('********** Constructing initial structures **********')
t0 = time.time()
ranking_review_fields_use = []
for ranking in corpus_rankings_delimited:
    ranking_fields = ranking.split('|')
    if not len(ranking_fields) == 1:
        ranking_review_fields_use.append(ranking_fields[1])

# add reviews
for review in corpus_reviews_delimited:
    review_fields = review.split('|')
    if not len(review_fields) == 1:
        ranking_review_fields_use[int(review_fields[1])] = {'ranking' : float(ranking_review_fields_use[int(review_fields[1])]), 'review': review_fields[0]}

t1 = time.time()
print('[*] Completed initial construction in ' + str(t1 - t0) + ' seconds')

# tokenize reviews
print('******************* Tokenizing reviews *******************')
t2 = time.time()
for index, binding in enumerate(ranking_review_fields_use):
    review = nltk.word_tokenize(binding['review']);
    ranking_review_fields_use[index] = {'ranking' : float(ranking_review_fields_use[index]['ranking']), 'review' : review}

t3 = time.time()
print('[*] Completed tokenization in ' + str(t3 - t2) + ' seconds')

# remove stopwords (no-stopwords array is separate from actual tokenized words)
print('******************* Removing stopwords *******************')
t4 = time.time()
STOP_TYPES = ['DET', 'CNJ']
no_stopwords = []
for bindings in ranking_review_fields_use:
    tokens = nltk.pos_tag(bindings['review'])
    useful_words = [word for word, wordtype in tokens if wordtype not in STOP_TYPES]
    no_stopwords.append({'ranking': float(bindings['ranking']), 'review' : useful_words})

t5 = time.time()
print('[*] Removed all stopwords in ' + str(t5 - t4) + ' seconds')

# At this point, we have no stop words. Start by just splitting in
# half. Aggregate all words (and associated frequencies) with associated
# ratings of <= 5.0
print('************** Assembling negative keywords **************')
t6 = time.time()
words_negative = {}
for review in no_stopwords:
    if (review['ranking'] < 5.0):
        for word in review['review']:
            if word in words_negative:
                words_negative[word] += 1
            else:
                words_negative[word] = 0

for word in words_negative:
    neg_bank.write(word)
    neg_bank.write('\n')

t7 = time.time()
print('[*] Found negative keywords in ' + str(t7 - t6) + ' seconds')
