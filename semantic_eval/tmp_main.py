
from KoBERTScore import BERTScore
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='bertscore 출력해주는 프로그램')

parser.add_argument('-c', default="./data/commongen/kobart_commongen_test_9600.txt")
parser.add_argument('-r', default="./data/commongen/kobart_commongen_test_ver2_total_real_morph.txt")
parser.add_argument('-m', default="monologg/kobert",choices=["bert-base-multilingual-cased", "monologg/kobert"])

args = parser.parse_args()

#model_name = "beomi/kcbert-base"
#model_name = "bert-base-multilingual-cased"
#model_name = "monologg/kobert"
model_name = args.m

#각 모델에 맞는 best_layer가 다름. 순서에 맞게 사용하면 됨.
#bertscore = BERTScore(model_name, best_layer=4)
bertscore = BERTScore(model_name, best_layer=2)
#bertscore = BERTScore(model_name, best_layer=2)

#candidate_example
#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/commongen_test_result/kobart_commongen_test_9600.txt'
#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/commongen_test_result/kogpt2_commongen_test_9600.txt'
#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/commongen_test_result/mbart_commongen_test_9600.txt'
#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/commongen_test_result/mt5_commongen_test_9600.txt'
#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/kommongen_test_result/kobart_commongen_test_9600.txt'
#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/kommongen_test_result/kogpt2_commongen_test_9600.txt'
#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/kommongen_test_result/mbart_commongen_test_9600.txt'
path_to_predict = args.c

#reference_example
#path_to_refer = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/kobart_commongen_test_ver2_total_real_morph.txt'
path_to_refer = args.r

print(model_name, path_to_predict, path_to_refer)

references = []

candidates = []

with open(path_to_predict, 'r', encoding="utf-8-sig") as prefile:

	prec_lines = prefile.readlines()

with open(path_to_refer, 'r', encoding="utf-8-sig") as refile:

	ref_lines = refile.readlines()

	total_BERTscore = 0
	cnt = 0

	for prec_line,ref_line in zip(prec_lines,ref_lines):

		tmp_dict = eval(ref_line)
		real_ref_line = tmp_dict['scene']

		candidates.append(prec_line.strip())
		references.append(real_ref_line.strip())

		cnt += 1


	cur_BS = bertscore(references, candidates, batch_size=128)
#	print(cur_BS)


	print(cnt)
	print(np.mean(cur_BS))



#print(bertscore(references, candidates, batch_size=128))
