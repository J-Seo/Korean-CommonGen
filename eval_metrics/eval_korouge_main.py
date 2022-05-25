
from rouge.korouge import *
import argparse

parser = argparse.ArgumentParser(description='rouge 출력해주는 프로그램')

parser.add_argument('-c', default="./data/prediction/kommongen_test_result/mt5_commongen_test_9600.txt")
parser.add_argument('-r', default="./data/prediction/tmp_kommongen_total_1k.txt")

args = parser.parse_args()

#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/commongen_test_result/kobart_commongen_test_9600.txt'
#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/commongen_test_result/kogpt2_commongen_test_9600.txt'
#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/commongen_test_result/mbart_commongen_test_9600.txt'
#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/commongen_test_result/mt5_commongen_test_9600.txt'
#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/kommongen_test_result/kobart_commongen_test_9600.txt'
#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/kommongen_test_result/kogpt2_commongen_test_9600.txt'
#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/kommongen_test_result/mbart_commongen_test_9600.txt'
#path_to_predict = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/kommongen_test_result/mt5_commongen_test_9600.txt'
path_to_predict = args.c


#path_to_refer = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/kobart_commongen_test_ver2_total_real_morph.txt'
#path_to_refer = '/home/max9985053/SPICE/CommonGen/evaluation/Traditional/eval_metrics/data/prediction/tmp_kommongen_total_1k.txt'
path_to_refer = args.r

rouge = Rouge()

with open(path_to_predict, 'r', encoding="utf-8-sig") as prefile:

	prec_lines = prefile.readlines()


with open(path_to_refer, 'r', encoding="utf-8-sig") as refile:

	ref_lines = refile.readlines()

	total_rouge_2_score = 0
	total_rouge_L_score = 0
	cnt = 0

	for prec_line,ref_line in zip(prec_lines,ref_lines):
		print(ref_line)
		print(type(ref_line))
		tmp_dict = eval(ref_line)
		real_ref_line = tmp_dict['scene']
		cur_r2 = rouge.calc_score_2([prec_line], [real_ref_line])
		cur_rl = rouge.calc_score_L([prec_line], [real_ref_line])
		
		total_rouge_2_score += cur_r2
		total_rouge_L_score += cur_rl

		cnt += 1

	avg_rouge_2_score = total_rouge_2_score/cnt 
	avg_rouge_L_score = total_rouge_L_score/cnt
	print(cnt)

	print("R2", avg_rouge_2_score)
	print("RL", avg_rouge_L_score)

