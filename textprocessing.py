# http://www.nltk.org/

"""
Text processing in Python.

Stemming is a technique used in Natural Language Processing
to reduce different inflected forms of words to a single invariant root form

Classes:
- class Text:
  initialize text as a list + staticmethod function which calculate the Edit Distance of two words

- class TextProcessing:
  initialize text and process it (remove punctuation,stemming)
  + can treat text as a list and set, calculate frequency of word in text

- class TextSimilarity:
  calculate Jaccard Similarity of two text, and this text can be treated as a list and set

- class MinHashSimilarity:
  let us estimate the Jaccard Similarity by the MinHash Algorithm (smaller accuracy but faster)


Functions:
- MinHashLshIterator:
  function takes the main text and check Jaccard Similarity with list of texts according to LSH formula
"""


try:
    from nltk.corpus import stopwords
    from string import punctuation   # punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    from stemming.porter2 import stem
    import editdistance
    import numpy as np
    import random
    from datasketch import MinHash, MinHashLSH
    import sys
except:
    sys.exit("Program requires modules")



class Text():
    def __init__(self, text, title):
        self.title = title
        self.text = text

    def __len__(self):
        return len(self.text)

    @staticmethod
    def edit_distance(x,y):
        dist = editdistance.eval(x,y)
        return dist



class TextProcessing(Text):
    def __init__(self, text, title):
        super().__init__(text,title)
        self.text_processing()

    def text_processing(self):
        # remove punctuation
        self.text = [word.lower() for word in self.text if not word in punctuation and word != '']
        # remove stopwords
        self.text =  [word for word in self.text if not word in stopwords.words('english')]
        # stemming of the text
        self.text = [stem(word) for word in self.text]

    @property
    def text_as_set(self):
        return set(self.text)

    @property
    def text_as_list(self):
        return list(self.text)

    def __repr__(self):
        return f"Length of text {self.title}: {len(self.text)}. Length of text as a set {len(set(self.text))}"

    def smart_string(func):
        def inner(self, x):
            if isinstance(x, str) == False:
                raise ValueError("inputs must be strings")
            return func(self, x)
        return inner

    @smart_string
    def frequency(self, word):
        return self.text.count(word)



class TextSimilarity(TextProcessing):
    def __init__(self,text,title):
        super().__init__(text,title)

    def smart_length(func):
        def inner(a, b):
            if len(a) == 0 or len(b) == 0:
                raise ValueError("inputs can not be empty")
            return func(a, b)
        return inner

    @staticmethod
    @smart_length
    def set_jaccard_similarity(a, b):
        return round(len(set(a&b))/len(set(a|b)),4)

    @staticmethod
    @smart_length
    def list_jaccard_similarity(a, b):
        common = []
        len_of_bags = len(a+b)
        for i in a:
            if i in b:
                common.append(i)
                b.remove(i)
        return round(len(common)/len_of_bags,4)



class MinHashSimilarity():
    def __init__(self, first_text, first_title, second_text, second_title, minhashes_number):
        self.first_text = TextProcessing(first_text, first_title).text_as_set
        self.second_text = TextProcessing(second_text, second_title).text_as_set
        self.words_set = list(self.first_text|self.second_text)
        self.minhashes_number = minhashes_number

    @property
    def signature_matrix(self):
        texts = [self.first_text, self.second_text]
        permutations = [i for i in range(len(self.words_set))]
        hash_matrix = np.zeros((self.minhashes_number, 2))
        hash_index = 0
        for hash in range(self.minhashes_number):
            j = 0
            permutation = np.random.permutation(permutations)
            for text in texts:
                for i in permutation:
                    # we make a sign in the SignatureMatrix for the first word which is in text
                    # this is the idea of MinHash
                    if self.words_set[i] in text:
                        hash_matrix[hash_index][j] = i
                        j+=1
                        break
            hash_index += 1
        return hash_matrix

    def minhash_jaccard_similarity(self):
        counter = 0
        _ =  self.signature_matrix
        for i in range(self.minhashes_number):
            if _[i,0] == _[i,1]:
                counter+=1
        return round(counter/self.minhashes_number, 2)



def minhash_lsh_iterator(main_text, texts_list, permutations_number):
    sets = [set(i) for i in texts_list]
    sets.insert(0, set(main_text))

    signature_matrix = [MinHash(num_perm=300) for _ in sets]
    i = 0
    while i < len(signature_matrix):
        for text in sets[i]:
            signature_matrix[i].update(text.encode('utf8'))
        i+=1
    thresh = 0.5
    lsh = MinHashLSH(threshold=thresh, num_perm=permutations_number)
    for i in range(len(signature_matrix)):
        lsh.insert(f"texts_list[{i}]", signature_matrix[i])
    results = lsh.query(signature_matrix[0])
    print(f"Approximate neighbours of the main text with Jaccard similarity > {thresh}", results)
