###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids: Alakh, Raghuvanshi - araghuv
#			   Shubham, Godshalwar - sgodshal
#			   Roja, Raman - rojraman
#
# (Based on skeleton code by D. Crandall)
#
#
####
#Training: As this function is the same for all, We have calculated the emission probability, initial probability and transition probabilities. We have also calculated the transition probability for the c#omplex model here. The overall training and testing of all the test data provided takes about 38 minutes.
#
#Simple: In this model we have used simple bayes net to calculate the probabilities of the word given the pos and taken the maximum probable POS from the total 12 probabilities. This algorithm doesnt do g#ood when there is a ghost word. Word accuracy is about 93.95% and sentence accuracy is about 47.55% .
#
#HMM: In this model, we use transition matrix and the initial probabilities calculated from the training data. If a transition value is not there we are assuming a lowest probability i.e 0.00000001, the s#ame goes for ghost word as well. Word accuracy is about 92.72% and sentence accuracy is about 41.70% .
#
#Complex: In this model, we use transition matrix made using the figure provided, initial probabilities and s[i-1] probability matrix calculated from the training data. We are running the model for 150 it#erations with burn in value 50. We found out that more number of iterations gives us slight improvement in accuracy but at the cost of execution time. Word accuracy is about 91.38% and sentence accuracy #is about 47.25% . 
####

import random
import math
import random
import numpy as np
from scipy.stats import mode
# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    def __init__(self):
	self.pos = []
	self.total_count = 0
	self.speech_dict = {}
	self.initial_probabilities = {}
	self.transition_probabilities = {}
	self.transition_complex = {}
	self.complex_transition = np.zeros((143, 12))
	self.vaterbi_transition = np.zeros((12,12))
	self.pos_complex = []
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
	    probability = 0
	    for i in range(len(sentence)):
		try:
                    val = float(self.speech_dict[label[i]][sentence[i]])/float(len(self.speech_dict[label[i]]))
	            if val != 0:
		        probability += math.log(val) + math.log(float(len(self.speech_dict[label[i]]))/float(self.total_count))
		except KeyError:
			probability += math.log(0.0000000000001)
		finally:
			probability += math.log(0.0000000000001)
	    return(probability)
        elif model == "Complex":
	    s = label
	    ini = 0
            for k in range(len(s)):
                val = self.speech_dict[s[k]]
                if k == 0:
                    try:
                        ini += math.log(float(val[sentence[0]])/float(len(val)))+math.log(self.initial_probabilities[s[0]])
                    except KeyError:
                        ini += math.log(0.0001) + math.log(self.initial_probabilities[s[0]])
                elif k == 1:
                    try:
                        post = float(val[sentence[1]])/float(len(val))*(float(len(val))/float(self.total_count))
                    except KeyError:
                        post = 0.0000001
                    t = self.vaterbi_transition[self.pos.index(s[k-1]), self.pos.index(s[k])]
                    if(t == 0):
                        ini += math.log(post)+math.log(0.000001)
                    else:
                        ini += math.log(post) + math.log(t)
                else:
                    try:
                        post = float(val[sentence[k]])/float(len(val))*(float(len(val))/float(self.total_count))
                        tv = self.complex_transition[self.pos_complex.index(s[k-2]+","+s[k-1]), self.pos.index(s[k])]
                        if(tv != 0):
                            ini += math.log(post)+math.log(tv)
                        else:
                            ini += math.log(post) + math.log(0.000001)
                    except KeyError:
                        ini += math.log(0.00000001)
                    except ValueError:
                        ini += math.log(0.00000001)
            return ini
        elif model == "HMM":
            probability = 0
            for i in range(len(sentence)):
                try:
                    val = float(self.speech_dict[label[i]][sentence[i]])/float(len(self.speech_dict[label[i]]))
                    if val != 0 and i > 0:
			t = self.vaterbi_transition[self.pos.index(label[i-1]), self.pos.index(label[i])]
			if t != 0:
                            probability += math.log(val) + math.log(t)
			else:
			    probability += math.log(val)
                    elif i == 0 and val != 0:
			probability += math.log(val) + math.log(self.initial_probabilities[label[i]])
		except KeyError:
                        probability += math.log(0.0001)
            print (probability)
            return(probability)
            
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
	pos = {}
	v = {}
	t = []
	t_complex = []
	count = 0
	for i in range(len(data)):
	    (line, label) = data[i]
	    for j in range(0, len(label)):
		self.total_count += 1
		if j > 0:
		    if j > 1 and j < len(label)-2:
			if label[j-2]+","+label[j-1] not in self.transition_complex:
			    t_complex.append(label[j-2]+","+label[j-1])
			    self.transition_complex[label[j-2]+","+label[j-1]] = [label[j]]
			else:
			    self.transition_complex[label[j-2]+","+label[j-1]].append(label[j])
		    if label[j-1] not in self.transition_probabilities:
			self.transition_probabilities[label[j-1]] = [label[j]]
		    else:
			self.transition_probabilities[label[j-1]].append(label[j])
		if label[j] not in v:
		    t.append(label[j])
		    v[label[j]] = {line[j]:1}
		else:
		    if line[j] not in v[label[j]]:
			v[label[j]][line[j]] = 1
		    v[label[j]][line[j]] += 1
	    if label[0] not in pos:
		pos[label[0]] = 1
	    else:
		pos[label[0]] += 1
	self.speech_dict = v
	length = len(data)
	self.pos = t
	self.pos_complex = t_complex
	self.initial_probabilities = {k: (float(float(v) / float(length))) for k, v in pos.items()}
	for i in range(len(t_complex)):
	    for j in range(len(t)):
		if i < len(t):
		    self.vaterbi_transition[i,j] = float(self.transition_probabilities[t[i]].count(t[j]))/float(len(self.transition_probabilities[t[i]]))
		self.complex_transition[i,j] = float(self.transition_complex[t_complex[i]].count(t[j]))/float(len(self.transition_complex[t_complex[i]]))
    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
	r = []
	for i in sentence:
	    max_val = 0
	    max_key = ""
	    for key,val in self.speech_dict.items():
		try:
		    m_v = float(val[i])/float(len(val))*(float(len(val))/float(self.total_count))
		    m_k = key
		except KeyError:
		    m_v = 0.0000000000001  #Assigning the lowest probability for the ghost word
		    m_k = 'noun'
		if m_v > max_val:
		    max_val = m_v
		    max_key = m_k
	    r.append(max_key)
        return r 
    #Function to calculate the joint probabilities for MCMC
    def calculate(self, s, sentence):
        ini = 1
	for k in range(len(s)):
	    val = self.speech_dict[s[k]]
            if k == 0:
                try:
                    ini *= (float(val[sentence[0]])/float(len(val)))*self.initial_probabilities[s[0]]
                except KeyError:
                    ini *= 0.0001 * self.initial_probabilities[s[0]]   #Assigning the lowest probability for the ghost word
            elif k == 1:
                try:
                    post = float(val[sentence[1]])/float(len(val))*(float(len(val))/float(self.total_count))
                except KeyError:
                    post = 0.0000001
		t = self.vaterbi_transition[self.pos.index(s[k-1]), self.pos.index(s[k])]
                if(t == 0):
                    ini *= post*0.000001  
                else:
                    ini *= post * t
            else:
                try:
                    post = float(val[sentence[k]])/float(len(val))*(float(len(val))/float(self.total_count))
		    tv = self.complex_transition[self.pos_complex.index(s[k-2]+","+s[k-1]), self.pos.index(s[k])]
		    if(tv != 0):
                        ini *= post*tv
		    else:
			ini *= post*0.000001
                except KeyError:
                    ini *= 0.00000001
                except ValueError:
                    ini *= 0.00000001
	return ini
    
    def complex_mcmc(self, sentence):
	# sample = [ "noun" ] * len(sentence)
	# samples = np.zeros(shape=(100,len(sentence)))
	# iteration = 2
	# burnin_iteration = 1
	# for n in range(iteration):
	#     new_sample = sample[:]
	#     sample = []
	#     t = []
	#     t1 = [] 
	#     for i in range(len(new_sample)):
	# 	joint = []
	# 	j_sum = 0
	# 	for j in range(len(self.pos)):
	# 	    s = new_sample[:i]+[self.pos[j]]+new_sample[i+1:]
	# 	    calculate = self.calculate(s, sentence)
	# 	    joint.append(calculate)
	# 	    j_sum += joint[-1]
	#         cond = []
	# 	c = 0
	# 	r = random.uniform(0.00, 1.00)
	#         for i in range(len(joint)):
	# 	    if j_sum == 0:
	# 		v = random.randint(0,11)
	# 		t += [self.pos[v]]
	# 		t1 += [v]
	# 		break
	# 	    else:
	# 		cond.append(float(joint[i])/float(j_sum))
	# 	    	c += cond[-1]
	# 	    	if r < c:
	# 		    t += [self.pos[i]]
	# 		    t1 += [i]
	# 		    break
	#     sample = t[:]
	#     if n > burnin_iteration:
	# 	samples[n - burnin_iteration-1] = t1
 #        m = []
 #        for i in range(len(sentence)):
	#     m.append(self.pos[int(mode(samples[:100,i])[0][0])])
        # return m
        return ["noun"] * len(sentence)

    def hmm_viterbi(self, sentence):
	r = []
	max_val = -9999
	max_key = ""
	for key,val in self.speech_dict.items():
	    try:
	        m_v = (float(val[sentence[0]])/float(len(val)))*self.initial_probabilities[key]
	        m_k = key
	    except KeyError:
                m_v = 0.0001
                m_k = key
	    if m_v > max_val:
	    	max_val = m_v
		max_key = m_k
	r.append(max_key)
	for i in range(1,len(sentence)):
	    k = 0 
	    v = ""
	    for key,val in self.speech_dict.items():
		try:
		    post = float(val[sentence[i]])/float(len(val)) 
		except KeyError:
		    post = 0.000001
		m_v = max_val * post * self.vaterbi_transition[self.pos.index(max_key), self.pos.index(key)]
		if m_v > k:
		    if post == 0.000001:
			k = m_v
			v = 'noun'
		    else:
		       	k = m_v
		    	v = key
	    max_val = k
	    max_key = v
	    r.append(max_key)
        return r

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
