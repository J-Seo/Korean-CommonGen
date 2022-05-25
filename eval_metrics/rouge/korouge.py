#!/usr/bin/env python
# 
# File Name : rouge.py
#
# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>

import numpy as np
import pdb
from konlpy.tag import Mecab
mecab = Mecab()

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

class Rouge():
    '''
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set

    '''
    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score_L(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
#        assert(len(candidate)==1)	
        assert(len(refs)>0)         
        prec = []
        rec = []

        # split into tokens
#        token_c = candidate[0].split(" ")
        token_k = mecab.morphs(candidate[0])
    	
        for reference in refs:
            # split into tokens
#            token_r = reference.split(" ")
            token_r = mecab.morphs(reference)
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_k)
            if len(token_k) == 0:
                prec.append(0)
            else:
                prec.append(lcs/float(len(token_k)))
            rec.append(lcs/float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)
        #print(prec_max)
        #print(rec_max)

        if(prec_max!=0 and rec_max !=0):
            score = ((1 + self.beta**2)*prec_max*rec_max)/float(rec_max + self.beta**2*prec_max)
        else:
            score = 0.0
        return score


    def calc_score_2(self, candidate, refs):

        prec = []
        rec = []

        tmp_score = 0

        token_k_bigram = []
        token_r_bigram = []

        token_k = mecab.morphs(candidate[0])
        for idx ,token_tmp in enumerate(token_k) :
            if token_tmp == token_k[-1]:
                break
            token_k_bigram.append("".join([token_tmp,token_k[idx+1]]))


        for reference in refs:
            token_r = mecab.morphs(reference)

            for idx ,token_tmp in enumerate(token_r) :
                if token_tmp == token_r[-1]:
                    break
                token_r_bigram.append("".join([token_tmp,token_r[idx+1]]))
            tmp_cnt = 0
            for bigram in token_r_bigram:
                if bigram in token_k_bigram:
                    tmp_cnt += 1
            if tmp_cnt == 0:
                prec.append(0)
                rec.append(0)
            else:
                prec.append(tmp_cnt/float(len(token_k_bigram)))
                rec.append(tmp_cnt/float(len(token_r_bigram)))
        prec_max = max(prec)
        rec_max = max(rec)
        #print(prec)
        #print(rec)

        if(prec_max!=0 and rec_max !=0):
            tmp_score = ((1 + self.beta**2)*prec_max*rec_max)/float(rec_max + self.beta**2*prec_max)
        else:
            tmp_score = 0.0
        return tmp_score
		

 
	    


    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py 
        :param hypo_for_image: dict : candidate / test sentences with "image name" key and "tokenized sentences" as values 
        :param ref_for_image: dict : reference MS-COCO sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        score = []
        for id in imgIds:
            hypo = res[id]
            ref  = gts[id]

            score.append(self.calc_score(hypo, ref))

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "Rouge"

# if __name__ == "__main__":
#
#     cand_1 = "A boy picks an apple tree and places it into bags."
#     cand_2 = "Two girls pick many red apples from trees and place them in a large bag."
#     ref = "A boy picks an apple from a tree and places it into bags."
#     concepts = ["pick", "apple", "tree", "place", "bag"]
#
#
#     rouge = Rouge()
#     print rouge.calc_score([cand_1], ref)
