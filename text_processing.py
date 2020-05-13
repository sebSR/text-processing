"""
Text processing in Python


NLTK - Natural Language Toolkit
Starting texts:
text1: Moby Dick by Herman Melville 1851
text2: Sense and Sensibility by Jane Austen 1811
text3: The Book of Genesis
text4: Inaugural Address Corpus
text5: Chat Corpus
text6: Monty Python and the Holy Grail
text7: Wall Street Journal
text8: Personals Corpus
text9: The Man Who Was Thursday by G . K . Chesterton 1908
"""

try:
    import nltk
    from nltk.corpus import stopwords
    # punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    from string import punctuation
    # Stemming is a technique used in Natural Language Processing
    # to reduce different inflected forms of words to a single invariant root form
    from stemming.porter2 import stem
    from nltk.book import *
    import editdistance
    import numpy as np
    import random
    from datasketch import MinHash, MinHashLSH
except:
    print('Program requires modules')
    exit(1)


# Root class for text
class Text():
    def __init__(self,text,title=''):
        self.title = title
        self.text = [word for word in text]

    # dunder method return length of text
    def __len__(self):
        return len(self.text)

    @staticmethod
    def EditDistance(x,y):
        return editdistance.eval(x,y)


# child class for text processing
class TextProcessing(Text):
    def __init__(self,text,title=''):
        super().__init__(text,title)
        # remove punctuation
        self.text = [word.lower() for word in self.text  if not word in punctuation and word != '']
        # remove stopwords
        self.text =  [word for word in self.text if not word in stopwords.words('english')]
        # stemming of the text
        self.text = [stem(word) for word in self.text]

    @property
    def TextAsSet(self):
        return set(self.text)

    @property
    def TextAsList(self):
        return list(self.text)

    def __repr__(self):
        return f"Length of text: {len(self.text)}. Length of text as a set {len(set(self.text))}"

    def frequency(self,word):
        return f"The word {word} appears {self.text.count(word)} times in the text"


# child class for text similarity
class TextSimilarity(TextProcessing):
    def __init__(self,text,title):
        super().__init__(text,title)

    @staticmethod
    def SetJaccardSimilarity(A,B):
        if len(A) == 0 or len(B) == 0:
            raise ValueError("set can not be empty")
        else:
            return round(len(set(A&B))/len(set(A|B)),4)

    @staticmethod
    def ListJaccardSimilarity(A,B):
        if len(A) == 0 or len(B) == 0:
            raise ValueError("set can not be empty")
        else:
            common = []
            len_of_bags = len(A+B)
            for i in A:
                if i in B:
                    common.append(i)
                    B.remove(i)
            return round(len(common)/len_of_bags,4)

# class for MinHash -> The Signature Matrix for both texts
class MinHash():
    def __init__(self,textOne,textTwo,numberOfMinhash):
        self.textOne = TextProcessing(textOne).TextAsSet
        self.textTwo = TextProcessing(textTwo).TextAsSet
        self.setOfWords = list(self.textOne|self.textTwo)
        self.numberOfMinhash = numberOfMinhash

    @property
    def SignatureMatrix(self):
        texts = [self.textOne,self.textTwo]
        permutations = [i for i in range(len(self.setOfWords))]
        M = np.zeros((self.numberOfMinhash,2))
        hashIndex = 0
        for hash in range(self.numberOfMinhash):
            j = 0
            permutation = np.random.permutation(permutations)
            for text in texts:
                for i in permutation:
                    # we make sign in the SignatureMatrix for the first word which is in text
                    # this is the idea of MinHash
                    if self.setOfWords[i] in text:
                        M[hashIndex][j] = i
                        j+=1
                        break
            hashIndex += 1
        return M

    def MinhashJaccardSimilarity(self):
        counter = 0
        M = self.SignatureMatrix
        for i in range(self.numberOfMinhash):
            if M[i,0] == M[i,1]:
                counter+=1
        return round(counter/100,2)




def testOfJaccardSimilarity():
    tmp = MinHash(text1,text5,100)
    print(f"Jaccard Similarity estimate by MinHash: {tmp.MinhashJaccardSimilarity()}")
    MobyDick = TextProcessing(text1)
    ChatCorpus = TextProcessing(text5)
    print(f"Jaccard Similarity: {TextSimilarity.SetJaccardSimilarity(MobyDick.TextAsSet,ChatCorpus.TextAsSet)}")

if __name__ == '__main__':
    testOfJaccardSimilarity()
