# https://docs.python.org/3/library/unittest.html

import sys
import unittest

sys.path.append("..")
import textprocessing as tp



class TestText(unittest.TestCase):

    def test_text_length(self):
        result_1 = tp.Text(['test', 'test'], 'test')
        self.assertEqual(len(result_1), 2)
        result_2 = tp.Text([], 'empty')
        self.assertEqual(len(result_2), 0)

    def test_edit_distance(self):
        result_1 = tp.Text.edit_distance('test1','test2')
        self.assertEqual(result_1, 1)
        result_2 = tp.Text.edit_distance('tes','test')
        self.assertEqual(result_2, 1)
        result_3 = tp.Text.edit_distance('testt','test')
        self.assertEqual(result_3, 1)
        result_4 = tp.Text.edit_distance('','')
        self.assertEqual(result_4, 0)



class TestTextProcessing(unittest.TestCase):

    def test_frequency(self):
        result_1 = tp.TextProcessing(['test', 'test'], 'test')
        self.assertRaises(ValueError, result_1.frequency, 0)
        result_2 = tp.TextProcessing(['test', 'Test', ''], 'test')
        self.assertEqual(result_2.frequency('test'), 2)



class TextSimilarity(unittest.TestCase):

    def test_set_jaccard_similarity(self):
        one = tp.TextProcessing(['test', 'test'], 'test')
        result = tp.TextSimilarity.set_jaccard_similarity(one.text_as_set,one.text_as_set)
        self.assertEqual(result, 1)
        two = tp.TextProcessing([], 'test')
        self.assertRaises(ValueError, tp.TextSimilarity.set_jaccard_similarity, two.text_as_set, two.text_as_set)

    def test_list_jaccard_similarity(self):
        one = tp.TextProcessing(['test', 'test'], 'test')
        result = tp.TextSimilarity.list_jaccard_similarity(one.text_as_list,one.text_as_list)
        self.assertEqual(result, 0.5)
        two = tp.TextProcessing([], 'test')
        self.assertRaises(ValueError, tp.TextSimilarity.list_jaccard_similarity, two.text_as_list, two.text_as_list)



if __name__ == "__main__":
    unittest.main()
