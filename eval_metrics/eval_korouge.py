
from rouge.korouge import *

str1 = "나는 밥을 먹었는데 너는 어떠니?"
str2 = "너는 밥을 먹었니?"


str1_token = mecab.morphs(str1)
print(str1_token)
str2_token = mecab.morphs(str2)
print(str2_token)
rouge = Rouge()
#str3 = rouge.calc_score([str1], [str2])
str3 = rouge.calc_score_2([str1], [str2])
print(str3)
