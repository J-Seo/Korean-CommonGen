# Korean CommonGen

A Dog Is Passing Over The Jet? A Text-Generation Dataset for Korean Commonsense Reasoning and Evaluation

## 1. Introduction

[NLP & AI Lab](http://blp.korea.ac.kr/), Korea University. 

*Jaehyung Seo, Seounghoon Lee, Chanjun Park, Yoonna Jang, Hyeonseok Moon, Sugyeong Eo, Seonmin Koo, and Heuiseok Lim*. **A Dog Is Passing Over The Jet? A Text-Generation Dataset for Korean Commonsense Reasoning and Evaluation. Our paper is will be published in findings of NAACL 2022**. This code is designed to evaluate the results of model generation. 

The whole dataset is publicly available at [Korean CommonGen](https://aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=459) in **AI-HUB**.

This dataset is an extension of the previous study [KommonGen](https://www.koreascience.or.kr/article/CFKO202130060697830.page) and [Dataset](https://github.com/J-Seo/KommonGen).

## 2. Overview

```python
/NAACL2022_Korean_CommonGen	
	/dataset # Dataset for Korean CommonGen train/dev/test
		- korean_commongen_official_train.txt/json 
		- korean_commongen_official_dev.txt/json (revising...)
		- korean_commongen_official_test.txt/json 
	
		/ablation_1 # Dataset for Concept Ablation Study
			- korean_commongen_free_morpheme_train.txt/json 
			- korean_commongen_free_morpheme_test.txt/json 
			- korean_commongen_only_noun_verb_train.txt/json 
			- korean_commongen_only_noun_verb_test.txt/json 
		
		/ablation_2 # Dataset for Data Source Ablation Study
			- korean_commongen_image_only_train.txt/json 
			- korean_commongen_dialogue_summary_only_train.txt/json
	
		/high_level_commonsense_reasoning # Dataset for High-level commonsense reasoning
			- high_level_korean_commongen_train_seed_42.txt/json
			- high_level_korean_commongen_train_seed_52.txt/json
			- high_level_korean_commongen_train_seed_62.txt/json
			- high_level_korean_commongen_train_seed_72.txt/json
			- high_level_korean_commongen_train_seed_82.txt/json

		/reformulated_commongen # Dataset for reformulated commongen
			- korean_commongen_reformulated_test.txt/json 

	/baseline_results # Model-generated results in quantitative evaluation and ablation study
		/quantitative_eval # Main experimental results on Korean CommonGen
			- KoGPT2_quantitative.txt
			- KoBART_quantitative.txt
			- mBART_quantitative.txt
			- mBART_50_quantitative.txt
			- mT5_small_qunatitative.txt
			- mT5_base_quantitative.txt
			- mT5_large_quantitative.txt

		/ablation_free_morph # Ablation results on Korean CommonGen (concept configuration)
			- KoGPT2_free_morph.txt
			- KoBART_free_morph.txt
			- mBART_50_free_morph.txt
			- mT5_large_free_morph.txt

		/ablation_noun_and_verb # Ablation results on Korean CommonGen (concept configuration)
			- KoGPT2_noun_and_verb.txt
			- KoBART_noun_and_verb.txt
			- mBART_50_noun_and_verb.txt
			- mT5_large_noun_and_verb.txt
		
		/ablation_only_image_captions # Ablation results on Korean CommonGen (data source)
			- KoGPT2_only_image_captions.txt
			- KoBART_only_image_captions.txt
			- mBART_50_only_image_captions.txt
			- mT5_large_only_image_captions.txt

		/ablation_only_dialogue_summary # Ablation results on Korean CommonGen (data source)
			- KoGPT2_only_dialogue_summary.txt
			- KoBART_only_dialogue_summary.txt
			- mBART_50_only_dialogue_summary.txt
			- mT5_large_only_dialogue_summary.txt

		/high_level_commonsense_reasoning # High-level commonsense reasoning results on Korean CommonGen
			- KoGPT2_high_level_commonsense_reasoning_seed42.txt
			- KoBART_high_level_commonsense_reasoning_seed42.txt
			- mBART_{25,50}_high_level_commonsense_reasoning_seed42.txt
			- mT5_{small,base,large}_high_level_commonsense_reasoning_seed42.txt

			- KoGPT2_high_level_commonsense_reasoning_seed52.txt
			- KoBART_high_level_commonsense_reasoning_seed52.txt
			- mBART_{25,50}_high_level_commonsense_reasoning_seed52.txt
			- mT5_{small,base,large}_high_level_commonsense_reasoning_seed52.txt
			
			- KoGPT2_high_level_commonsense_reasoning_seed62.txt
			- KoBART_high_level_commonsense_reasoning_seed62.txt
			- mBART_{25,50}_high_level_commonsense_reasoning_seed62.txt
			- mT5_{small,base,large}_high_level_commonsense_reasoning_seed62.txt

			- KoGPT2_high_level_commonsense_reasoning_seed72.txt
			- KoBART_high_level_commonsense_reasoning_seed72.txt
			- mBART_{25,50}_high_level_commonsense_reasoning_seed72.txt
			- mT5_{small,base,large}_high_level_commonsense_reasoning_seed72.txt

			- KoGPT2_high_level_commonsense_reasoning_seed82.txt
			- KoBART_high_level_commonsense_reasoning_seed82.txt
			- mBART_{25,50}_high_level_commonsense_reasoning_seed82.txt
			- mT5_{small,base,large}_high_level_commonsense_reasoning_seed82.txt

		/reformulated_commongen # Reformulated commongen test results on Korean CommonGen
			- KoGPT2_reformulated_commongen.txt
			- KoBART_reformulated_commongen.txt
			- mBART_50_reformulated_commongen.txt
			- mT5_large_reformulated_commongen.txt

	/human_evaluations # Evaluation results of humans with respect to the four criteria (.txt's index is same as model-generated sentence index)
		/kogpt2 # Korean commongen/reformulated commongen + (commonsense, factuality, fluency, grammar correction)
			- gpt2s_commonsense_korean_commongen.txt
			- gpt2s_commonsense_reform.txt
			- gpt2s_factuality_korean_commongen.txt
			- gpt2s_factuality_reform.txt
			- gpt2s_fluency_korean_commongen.txt
			- gpt2s_fluency_reform.txt
			- gpt2s_grammar_korean_commongen.txt
			- gpt2s_grammar_reform.txt

		/kobart # Korean commongen/reformulated commongen + (commonsense, factuality, fluency, grammar correction)
			- kobart_commonsense_korean_commongen.txt
			- kobart_commonsense_reform.txt
			- kobart_factuality_korean_commongen.txt
			- kobart_factuality_reform.txt
			- kobart_fluency_korean_commongen.txt
			- kobart_fluency_reform.txt
			- kobart_grammar_korean_commongen.txt
			- kobart_grammar_reform.txt

		/mbart-50 # Korean commongen/reformulated commongen + (commonsense, factuality, fluency, grammar correction)
			- mbart_commonsense_korean_commongen.txt
			- mbart_commonsense_reform.txt
			- mbart_factuality_korean_commongen.txt
			- mbart_factuality_reform.txt
			- mbart_fluency_korean_commongen.txt
			- mbart_fluency_reform.txt
			- mbart_grammar_korean_commongen.txt
			- mbart_grammar_reform.txt
	
		/mt5-large # Korean commongen/reformulated commongen + (commonsense, factuality, fluency, grammar correction)
			- mt5_commonsense_korean_commongen.txt
			- mt5_commonsense_reform.txt
			- mt5_factuality_korean_commongen.txt
			- mt5_factuality_reform.txt
			- mt5_fluency_korean_commongen.txt
			- mt5_fluency_reform.txt
			- mt5_grammar_korean_commongen.txt
			- mt5_grammar_reform.txt

	# Evaluation metrics
	- korean_commongen_evaluation_multi_ref.py
		/semantic_eval # For BERTscore
		/eval_metrics # For Rouge for Korean

	# Requirements for implementations
	- requirements.txt

```

## 3. Installation

- To implement evaluation metrics, you should install 1.1 KoNLPy and 1.2 Ko-mecab

**1.1** KoNLPy 0.5.2

[https://konlpy.org/ko/latest/index.html](https://konlpy.org/ko/latest/index.html)

(1) Ubuntu 16.04 ~

```bash
$ sudo apt-get install g++ openjdk-7-jdk # Install Java 1.7+
$ sudo apt-get install python-dev; pip install konlpy     # Python 2.x
$ sudo apt-get install python3-dev; pip3 install konlpy   # Python 3.x
```

you should check the optimal version of openjdk considering the ubuntu version (if you use ubuntu 20.04, then you should install openjdk-11-jdk)


(2) MAC

```bash
$ pip install konlpy     # Python 2.x
$ pip3 install konlpy    # Python 3.x
```

**1.2** Ko-mecab

(1) Ubuntu 16.04 ~ 20.04

```bash
$ sudo apt-get install curl git
$ bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

if you have some problem in loading mecab, it is one of the options using *bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)* in your conda environment.

(2) MAC

```bash
$ bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

## 4. Evaluation

```python
- $TASK_NAME$: Enter the folder name of the sub-folder.
- $MODEL_GENERATE.TXT$: Enter a file name of the results generated by the model.
- $MODEL_NAME$: Enter the model name.
```

```bash
conda create -n evaluation python=3.7
conda activate evaluation
pip install -r requirements_eval.txt
```

```python
# 2. Quantitative Experiment / 5. High-level
python korean_commongen_evaluation_multi_ref.py --reference_file dataset/korean_commongen_official_test.txt --prediction_file baseline_results/$TASK_NAME$/$MODEL_GENERATE.TXT$ --model $MODEL_NAME$

# 3 & 4. Ablation Study
python korean_commongen_evaluation_multi_ref.py --reference_file dataset/ablation_${1,2}$/korean_commongen_${free_morpheme_test,only_noun_verb_test}$.txt --prediction_file baseline_results/$TASK_NAME$/$MODEL_GENERATE.TXT$ --model $MODEL_NAME$

# Reformulated
python korean_commongen_evaluation_multi_ref.py --reference_file dataset/reformulated_commongen/korean_commongen_reformulated_test.txt --prediction_file baseline_results/$TASK_NAME$/$MODEL_GENERATE.TXT$ --model $MODEL_NAME$
```

😳 Warning! 

If the version of the cuda driver is 11.1 or higher, the following error may occur.

```bash
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Then, installing conda pytorch can be one of the solutions.

```bash
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**We basically modified the code to enable evaluation even in the CPU environment.**

## 5. Training & Decoding

To train and genetate the sentences, we use pytorch framework and [HuggingFace](https://github.com/huggingface/transformers) language modeling.
Using the pytorch pipeline disclosed by huggingface, it will be possible to implement training and generation code without difficulty.

## 6. Citation

```
Will be published
```

## 7. Thanks

Our dataset is based on [CommonGen](https://github.com/INK-USC/CommonGen "CommonGen"). We thank the authors for their academic contribution to us the opportunity to do this research.





