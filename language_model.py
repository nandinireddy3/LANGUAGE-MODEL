from tokenization import tokenizer as tok
from collections import Counter
import random
import re
import math
import numpy as np
import sys
"""
step 1: Vocabulary
step 2: every 4 grams and their counts
step 3: every 3 grams and corresponding counts of all vocabulary with those 3-grams
"""

n_unigrams = {}
n_bigrams = {}
n_trigrams = {}
n_NGRAMS = [n_unigrams,n_bigrams,n_trigrams]

sum_unigrams = {}
sum_bigrams = {}
sum_trigrams = {}
sum_fourgrams = {}
sum_NGRAMS = [sum_unigrams,sum_bigrams,sum_trigrams,sum_fourgrams]

def n_grams(n,text):
    ngrams = {}
    string = re.split('\s',text)
    if n == 1:
        for i in range(n, len(string)+1):
            # Excluding the n gram containg . thinking of each sentence
            if(string[i-1] == '.'):
              continue
            if " ".join(string[i-n:i]) in ngrams:
                ngrams[" ".join(string[i-n:i])] += 1
            else:
                ngrams[" ".join(string[i-n:i])] = 1
    else:
        for i in range(n, len(string)+1):
            # Excluding the n gram containg . thinking of each sentence
            if( (4 <= i and string[i-4] == '.') or (3 <= i and string[i-3] == '.') or (2 <= i and string[i-2] == '.') or (1 <= i and string[i-1] == '.')):
                continue
            if " ".join(string[i-n:i]) in ngrams:
                ngrams[" ".join(string[i-n:i])] += 1
            else:
                if " ".join(string[i-n:i-1]) in n_NGRAMS[n-2]:
                    n_NGRAMS[n-2][" ".join(string[i-n:i-1])] += 1
                else:
                    n_NGRAMS[n-2][" ".join(string[i-n:i-1])] = 1
                ngrams[" ".join(string[i-n:i])] = 1
            if(n == 2):
                if string[i-n] in sum_NGRAMS[n-2]:
                    sum_NGRAMS[n-2][string[i-n]] += 1
                else:
                    sum_NGRAMS[n-2][string[i-n]] = 1
            else:

                if " ".join(string[i-n:i-1]) in sum_NGRAMS[n-2]:
                    sum_NGRAMS[n-2][" ".join(string[i-n:i-1])] += 1
                else:
                    sum_NGRAMS[n-2][" ".join(string[i-n:i-1])] = 1
    return ngrams
def AllGramsFreq(n,para):
    res = {}
    for i in range(n,1,-1):
        freq = n_grams(i,para)
        res[i] = freq
    freq = n_grams(1,para)
    ug = {}
    for items,count in freq.items():
        if count < 5:
            if "<UNK>" not in ug:
                ug["<UNK>"] = 1
            else:
                ug["<UNK>"] = ug["<UNK>"]+1
        else:
            ug[items] = count
    res[1] = ug
    return res   

def sum_counts(history,cnt_allgrams):
    n = len(history.split())
    if n != 1:
        try:
            return sum_NGRAMS[n-1][history]
        except:
            return 0
    else:
        try: 
            return sum_NGRAMS[n-1][history]
        except:
            return cnt_allgrams[1]["<UNK>"]
     
def positive_counts(history,cnt_allgrams):
    n = len(history.split())
    if n == 1 and history not in cnt_allgrams[1]:
        return cnt_allgrams[1]["<UNK>"]
    try:
        return n_NGRAMS[n-1][history]
    except:
        return 0

def kneyser_smoothing(n,history,word,cnt_allgrams):
    d = 0.75
    if word not in cnt_allgrams[1]:
        return d/cnt_allgrams[1]["<UNK>"]
    if (n-1) == 0:
        return (1-d)/len(cnt_allgrams[1]) + d/cnt_allgrams[1]["<UNK>"]
        # return d/cnt_allgrams[1]["<UNK>"]
    text = " ".join([history,word])
    new_history = history.split()
    new_history = " ".join(new_history[1:])
    try:
        prob_firstterm = max(sum_counts(text,cnt_allgrams)-d,0)/sum_counts(history,cnt_allgrams)
    except:
        prob_firstterm = 0
    try:
        lamda = (d/sum_counts(history,cnt_allgrams))*positive_counts(history,cnt_allgrams)
    except:
        lamda = 0.000001
    prob_secondterm = lamda*kneyser_smoothing(n-1,new_history,word,cnt_allgrams)
    #print(n,lamda,prob_firstterm,prob_secondterm,sum_counts(history,cnt_allgrams),positive_counts(history,cnt_allgrams))
    return prob_firstterm+prob_secondterm

def kneyser_sentence2ngrams(n,text,cnt_allgrams):
    prob = []
    for gg,val in n_grams(n,text).items():
        p = gg.split()
        history = " ".join(p[:-1])
        word = p[-1]
        prob.append(kneyser_smoothing(n,history,word,cnt_allgrams))
    return prob
def Witten_Bell_Smoothing(n,history,word,cnt_allgrams):

    text = " ".join([history,word])
    new_history = history.split()
    new_history = " ".join(new_history[1:])
    if(n == 1):
        return sum_counts(text,cnt_allgrams)/cnt_allgrams[1]["<UNK>"]
    else:
        try:
            pML = sum_counts(text,cnt_allgrams)/sum_counts(history,cnt_allgrams)
            lamda = positive_counts(history,cnt_allgrams)/max(1,sum_counts(history,cnt_allgrams) + positive_counts(history,cnt_allgrams))
        except:
            pML = 0
            lamda = 1e-6
    return (1-lamda)*pML+lamda*Witten_Bell_Smoothing(n-1,new_history,word,cnt_allgrams)

def Witten_Bell_Sentence2ngrams(n,text,cnt_allgrams):
    prob = []
    # print(n_grams(n,text))
    for gg,val in n_grams(n,text).items():
        p = gg.split()
        history = " ".join(p[:-1])
        word = p[-1]
        prob.append(Witten_Bell_Smoothing(n,history,word,cnt_allgrams))
    return prob

def perplexity_calculation(prob):
    perplexity = 0
    for p in prob:
        perplexity += math.log(p,2)
    perplexity = -1*perplexity/len(prob)
    return math.pow(2,perplexity)
    # return np.power(1/np.prod(prob), 1/len(prob))


smooth_type = sys.argv[1]
fil = sys.argv[2]
with open(fil, "r", encoding="utf8") as file:
     text = file.read()
tk = tok()
t = tk.substitute(text)
t = tk.Punctuations(t)

sents = t.split(" . ")
train_data = ""
train_list = []
test_data = []
idx = np.random.choice(len(sents), 1000, replace=False)
for i in range(0,len(sents)):
    if i in idx:
        test_data.append(sents[i])
    else:
        train_data += sents[i]
        train_data += "."
        train_list.append(sents[i])


cnt_allgrams = AllGramsFreq(4,train_data)

sentence = str(input())
sentence = tk.substitute(sentence)
sentence = tk.Punctuations(sentence)
if(smooth_type == "k"):
    p = 1
    for prob in kneyser_sentence2ngrams(4,sentence,cnt_allgrams):
        p *= prob
    print(p)
elif(smooth_type == "w"):
    p = 1
    for prob in Witten_Bell_Sentence2ngrams(4,sentence,cnt_allgrams):
        p *= prob
    print(p)

# scores = []
# i = 0
# avg_perplexity = 0
# for samp in test_data:
#     if len(samp.split()) >= 4:
#         i += 1
#         print(i)
#         try:
#             prob = kneyser_sentence2ngrams(4,samp,cnt_allgrams)
#             curr_perplexity = perplexity_calculation(prob)
#             if(curr_perplexity < 100000):
#                 scores.append("<s>"+samp.rstrip("\n") + "</s>"+ "\t" + str(curr_perplexity)+ "\n")
#                 avg_perplexity = avg_perplexity+curr_perplexity
#         except:
#             continue
# with open ("2020101038_LM3_test-perplexity.txt",'w',encoding="utf8") as f:
#     f.write("Average perplexity: " + str(avg_perplexity/len(test_data) ) + "\n")
#     f.writelines(scores)
