#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
from parsivar import Normalizer, Tokenizer
from text_segmentation import segmentor
from embedding import Embedding
import numpy as np

class Keyword_Extraction():
    def __init__(self, text, segment_num= 3, remove_stop_words= False):

        print('Our models are loading...')
        self.normalizer = Normalizer()
        self.tokenizer = Tokenizer()
        self.embeder = Embedding()

        self.text = text
        self.remove_sw = remove_stop_words
        self.segment_num = segment_num

        self.tokens, self.text = self._preprocess()

        self.word_type = list(np.unique(self.tokens))

        self.segments_text, self.segments_token, _ = segmentor(self.tokens, self.segment_num)

    def _preprocess(self):

        print('Your text is processing now!')
        normal_text = self.normalizer.normalize(self.text)
        tokens = self.tokenizer.tokenize_words(self.normalizer.normalize(self.text))
        if self.remove_sw:
            print('Stop-words are removing...')
            with open('stop_words.txt', mode="r", encoding="utf-8") as f:
                stop_words = [ sw.strip() for sw in f.readlines()]
            removed_sw_tokens = [] 
            for word in tokens: 
                if word not in stop_words:
                    removed_sw_tokens.append(word)

            return removed_sw_tokens, ' '.join(removed_sw_tokens)            
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

    def top_words(self, num=5):
        
        print('TOP-words are finding...')

        semantic_word_score = self.semantic_score()
        count_word_score = self.count_score()

        final_word_score = {}

        for word in semantic_word_score.keys():
            final_word_score[word] = semantic_word_score [word] * count_word_score[word]
        
        sorted_word_score = sorted(final_word_score.items(), key=lambda y: y[1], reverse=True)

        return sorted_word_score[:num+1]

    