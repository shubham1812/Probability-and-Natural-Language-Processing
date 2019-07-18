#!/usr/bin/python

##########################################################################################################################################################
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Shubham Godshalwar(sgodshal), Alakh Rakhuvanshi(araghuv), Roja Raman(rojraman)
# (based on skeleton code by D. Crandall, Oct 2018)
# 1) Simple: Simple algorithm is basically Naive Based Classifier
# -> Naive based classifier is implemented here by selecting the maximum value of emission probability comparing the training letter with the test letter
# -> To calculate the emission probability, we have assumed that 2/3rd of the pixels are noise free while the rest 1/3rd are noisy pixel. We calculate the  
# the number of noisy and noise free pixels and emission probability for that particular test letter and match it to the training letter and apply the formula
# (2/3)^m * (1/3)^(N-m) where m in number of matched pixel and N is total pixels
# 2) HMM: We will use the emission probability calculated earlier and calculate the initial probability from text file by calculating the probability of 
#	each letter that starts the sentence. We will also need the transitional probability for each transition which can be calculated by frequency of the 
# transition divided by the number of all transition from the source character
# 
#Results:
# We get better performance for Viterbi in terms of prediction than Naive Bayes Classifier.
# For Noisy and sparse images the performance deteriorates for both with Naive Bayes being the worst.
#
#References:
# 1]http://blog.ivank.net/viterbi-algorithm-clarified.html
# 2]http://www.utdallas.edu/~prr105020/biol6385/2018/lecture/Viterbi_handout.pdf
# 
##########################################################################################################################################################
from PIL import Image, ImageDraw, ImageFont
import sys
import fileinput
import math
import operator
CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

#method to read data from the training data file
def read_data(filename):
    train_initial_dict = {}
    train_transitional_dictionary = {}
    for x in train_letters:
    	train_initial_dict[x]=0
    	train_transitional_dictionary[x]={}
    	for y in train_letters:
    		train_transitional_dictionary[x][y]=0
    for line in fileinput.input(filename):
    	line.strip("ADJ").strip("ADV").strip("ADP").strip("CONJ").strip("DET").strip("NOUN").strip("NUM").strip("PRON").strip("PRT").strip("VERB")
    	if line[0] in train_letters:
    		train_initial_dict[line[0]] += 1
    	for x in range(1,len(line)-1):
    		if  line[x-1] in train_letters and line[x] in train_letters:
    			train_transitional_dictionary[line[x-1]][line[x]] += 1
    #finding transitional probability
    total_lines=sum(train_initial_dict.values())
    for x in train_transitional_dictionary:
		sum_value=sum(train_transitional_dictionary[x].values())
		for y in train_transitional_dictionary[x]:
			if train_transitional_dictionary[x][y]==0 or sum_value == 0:
				train_transitional_dictionary[x][y]=0.00001
			else:	
				train_transitional_dictionary[x][y]=train_transitional_dictionary[x][y]/float(sum_value)
	#finding initial probability
    for x in train_initial_dict:
		if train_initial_dict[x] == 0:
			train_initial_dict[x] = 0.00001
		else:
			train_initial_dict[x] = train_initial_dict[x]/float(total_lines)
		
    return train_initial_dict,train_transitional_dictionary


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#method to find emission probability
def emission_probability(train_letters,test_letters):
	emission_probability={}
	for x in range(len(test_letters)): 
		emission_probability[x] = {}
		for y in train_letters:
			probability = 1
			match_count = 0
			unmatch_count = 0
			for i in range(len(train_letters[y])):
				for j in range(len(train_letters[y][i])):
					if test_letters[x][i][j] == train_letters[y][i][j]:
						match_count +=1
						probability *=0.67
					else : 
						unmatch_count +=1
						probability *=0.33
			emission_probability[x][y] = probability
			# print "\n".join([ r for r in train_letters[x]])
	return emission_probability

#calculating the transfered probability by multiplying the transitional probability
def get_all_probability(char,previous_probability_list,train_transitional_dictionary,train_letters):
	probability = []
	for x in train_letters:
		probability.append(previous_probability_list[x] + math.log(train_transitional_dictionary[x][char]))
	return probability

#simple naive based classifier
def simple(test_letters,emission_probability):
	sentence = ""
	for x in range(len(test_letters)):
		sentence+=max(emission_probability[x].iteritems(), key=operator.itemgetter(1))[0]
	return sentence

#viterbi algorithm
def viterbi(test_letters,train_initial_dict,train_transitional_dictionary,emission_dict):
	train_letters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
	sentence = ""
	probability_dict={}
	for x in range(len(test_letters)):
		probability_dict[x]={}
		for y in train_letters:
			if x == 0:
				prob = math.log(train_initial_dict[y]) + math.log(emission_dict[x][y])
				probability_dict[x][y] = prob
			else:
				max_transferred_probability=max(get_all_probability(y,probability_dict[x-1],train_transitional_dictionary,train_letters))
				prob = max_transferred_probability +  math.log(emission_dict[x][y])
				probability_dict[x][y] = prob
		sentence += max(probability_dict[x].iteritems(), key=operator.itemgetter(1))[0]
	return sentence

# def viterbi_1(test_letters,train_initial_dict,train_transitional_dictionary,emission_dict):
# 	train_letters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
# 	sentence = ""
# 	probability_dict={}
# 	for x in range(len(test_letters)):
# 		probability_dict[x]={}
# 		for y in train_letters:
# 			if x == 0:
# 				prob = math.log(train_initial_dict[y]) + math.log(emission_dict[x][y])
# 				probability_dict[x][y] = prob
# 			else:
# 				key=max(probability_dict[x-1].iteritems(), key=operator.itemgetter(1))[0]
# 				max_previous = max(probability_dict[x-1].values())
# 				try :
# 					prob =max_previous + math.log(train_transitional_dictionary[key][y]) + math.log(emission_dict[x][y])
# 				except ValueError:
# 					prob = max_previous + 0.0000000000001 + math.log(emission_dict[x][y])
# 				probability_dict[x][y] = prob
# 		sentence += max(probability_dict[x].iteritems(), key=operator.itemgetter(1))[0]
# 	return sentence

#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
#training letters
train_letters = load_training_letters(train_img_fname)
#testing letters
test_letters = load_letters(test_img_fname)
#getting the initial probability and transitional probability
train_initial_dict,train_transitional_dictionary=read_data(train_txt_fname)
#finding emission probability
emission_dict=emission_probability(train_letters,test_letters)
#naive based classifer solution 
simple_ans=simple(test_letters,emission_dict)
#viterbi solution
viterbi_ans=viterbi(test_letters,train_initial_dict,train_transitional_dictionary,emission_dict)
print "Simple: " + simple_ans
print "Viterbi: " + viterbi_ans
