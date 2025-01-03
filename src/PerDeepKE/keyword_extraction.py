#!/usr/bin/python3
# -*- coding: utf-8 -*-
from parsivar import Normalizer, Tokenizer
from .embedding import Embedding
from .text_segmentation import segmentor
import numpy as np
import Levenshtein
class Keyword_Extraction():
    def __init__(self, text, segment_num= 3):

        print('='*30 + '\nOur models are loading...\n')
        self.normalizer = Normalizer()
        self.tokenizer = Tokenizer()
        self.embeder = Embedding()

        self.text = text
        self.segment_num = segment_num

        self.tokens, self.text = self._preprocess()

        self.word_type = list(np.unique(self.tokens))

        self.segments_text, self.segments_token, _ = segmentor(self.tokens, self.segment_num)

    def _preprocess(self):

        print('='*30 + '\nYour text is processing now!\n')
        normal_text = self.normalizer.normalize(self.text)
        tokens = self.tokenizer.tokenize_words(self.normalizer.normalize(self.text))

        return tokens, normal_text

    def _embedding(self, texts):
        return self.embeder.sentence_embedding(texts)
    
    def get_tokens(self):
        return self.tokens

    def get_word_type(self):
        return self.word_type

    def _sim_matrix(self):

        text_em = self._embedding([self.text])
        segments_text_em = self._embedding(self.segments_text)
        word_type_em = self._embedding(self.word_type)

        word_text_sim = np.matmul(text_em, word_type_em.T)
        word_segments_sim = np.matmul(segments_text_em, word_type_em.T)

        return np.concatenate((word_text_sim, word_segments_sim), axis = 0)


    def semantic_score(self):

        sims = self._sim_matrix()

        self.weights = np.ones(sims.shape[0])
        self.weights[0] *= 3 

        word_score = zip(self.word_type, np.apply_along_axis(self._semantic_score_up, 0, sims))

        return dict(word_score)

    def _semantic_score_up(self, scores):
        return np.dot(scores, self.weights)

    def count_score(self):

        score = []
        for word in self.word_type:
            tf = self.word_type.count(word)
            idf = 0
            for segment in self.segments_token:
                if word in segment:
                    idf += 1
            
            score.append(tf/idf)
        
        return dict(zip(self.word_type, score))
    def filter_similar_words(self, words, threshold=5):
          filtered_words = []
          for word in words:
            if all(Levenshtein.distance(word, w) > threshold for w in filtered_words):
              filtered_words.append(word)
          return filtered_words
    def top_words(self, num=5):
        
        print('='*30 + '\nTOP-words are finding...\n')

        semantic_word_score = self.semantic_score()
        count_word_score = self.count_score()

        final_word_score = {}

        for word in semantic_word_score.keys():
            final_word_score[word] = semantic_word_score [word] * count_word_score[word]
        
        sorted_word_score = sorted(final_word_score.items(), key=lambda y: y[1], reverse=True)
        top_words = [word for word, _ in sorted_word_score[:num+1]]

        # Filter similar words
        top_words = self.filter_similar_words(top_words)

        return top_words


if __name__=='__main__':

    text = "بر اساس تحلیل نقشه‌های همدیدی و آینده‌نگری سازمان هواشناسی امروز در استان‌های ساحلی دریای خزر، اردبیل، شمال آذربایجان شرقی و ارتفاعات البرز مرکزی بارش باران، همراه با وزش باد شدید موقتی و کاهش نسبی دما پیش‌بینی شده است. فردا از میزان بارش‌های این مناطق کاسته شده و فقط در سواحل شمالی بارش پراکنده روی می‌دهد."
    ke = Keyword_Extraction(text, segment_num=2)

    for word, _ in ke.top_words(num=10):
        print(word)
