try:
    import nltk
    from nltk.book import *
    import textprocessing as tp
    from textprocessing import *
except:
    print('Program requires modules')
    exit(1)

"""
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



def TextClass_example(x, y):
    result = tp.Text.edit_distance(x, y)
    print(f"Edit distance of \'{x}\' and \'{y}\': {result}")


def TextProcessingClass_example(text, title):
    result = TextProcessing(text, title)
    print(result)


def TextSimilarityClass_example(first_text, first_title, second_text, second_title):
    first = tp.TextProcessing(first_text, first_title)
    second = tp.TextProcessing(second_text, second_title)
    result = tp.TextSimilarity.set_jaccard_similarity(first.text_as_set, second.text_as_set)
    print(f"Jaccard Similarity (sets) of \'{first.title}\' and \'{second.title}\': {result}")


def MinHashSimilarityClass_example(first_text, first_title, second_text, second_title, permutations_number):
    tmp = tp.MinHashSimilarity(first_text, first_title, second_text, second_title, permutations_number)
    result = tmp.minhash_jaccard_similarity()
    print(f"Minhash Jaccard Similarity for the texts with function above: {result}")



if __name__ == '__main__':
    permutations_number = 300

    TextClass_example('one', 'two')

    TextProcessingClass_example(text1, 'Moby Dick')

    TextSimilarityClass_example(text1, 'Moby Dick', text2, 'Sense and Sensibility')

    MinHashSimilarityClass_example(text1, 'Moby Dick',  text2, 'Sense and Sensibility', permutations_number)

    minhash_lsh_iterator(text1, [text1,text2,text3,text4], permutations_number)




"""
OUTPUT:

Edit distance of 'one' and 'two': 3

Length of text Moby Dick: 113865. Length of text as a set 10443

Jaccard Similarity (sets) of 'Moby Dick' and 'Moby Dick': 0.2804

MinhashJaccardSimilarity for the texts with function above: 0.28

Approximate neighbours of the main textx with Jaccard similarity > 0.5 ['listOfTexts[0]', 'listOfTexts[1]']

"""
