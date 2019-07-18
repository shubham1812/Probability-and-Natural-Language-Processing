import math
def emission_probability(train_letters,test_letters):
	emission_probability={}
	for x in range(len(test_letters)): 
		emission_probability[x] = {}
		for y in train_letters:
			print y
			probability = 1
			for i in range(len(train_letters[y])):
				for j in range(len(train_letters[y][i])):
					if test_letters[x][i][j] == train_letters[y][i][j]:
						print test_letters[x][i][j] + " " + train_letters[y][i][j]
						probability *=0.8
					else :
						print test_letters[x][i][j] + " no " + train_letters[y][i][j] 
						probability *=0.2
			emission_probability[x][y] = math.log(probability)
	print emission_probability

a = {'a': ['absc'],'b':['sasw']}
b = [['bbse']]
emission_probability(a,b)