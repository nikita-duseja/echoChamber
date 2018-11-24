import sys
import os
import math
import operator
import random
import matplotlib as mp
import networkx as nx
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

eCon = "EXTREME_CONSERVATIVE" # 2.5 - 1.5
mCon = "MOD_CONSERVATIVE" #1.5 - 0.5
neut = "NEUTRAL" # 0.5 - -0.5
mLib = "MOD_LIBERAL" #-0.5 - -1.5
eLib = "EXTREME_LIBERAL" #-1.5 - -2.5
ftot = "TOTAL_FRIENDS"

def computeFraction(userId):
	polarityUser = userPolarityTrain[userId]

	if abs(polarityUser) < 0.5:
		graph[userId]["sameClass"] = len(graph[userId][neut])
		graph[userId]["diffClass"] = len(graph[userId][eCon]) + len(graph[userId][eLib]) + len(graph[userId][mCon]) +len(graph[userId][mLib])
	elif polarityUser > 0.5:
		graph[userId]["sameClass"] = len(graph[userId][mCon]) + len(graph[userId][eCon])
		graph[userId]["diffClass"] = len(graph[userId][eLib]) + len(graph[userId][mLib])
	elif polarityUser < -0.5:
		graph[userId]["diffClass"] = len(graph[userId][mCon]) + len(graph[userId][eCon])
		graph[userId]["sameClass"] = len(graph[userId][eLib]) + len(graph[userId][mLib])

	graph[userId]["diffFrac"] = 	(graph[userId]["sameClass"] - graph[userId]["diffClass"])/float(len(graph[userId][ftot]))


	
def getProcessList():
	file1 = open("out.txt","r")
	for line in file1:
		processList.append(line[:-1])
	file1.close()

def userKeys():
	file = open("polarity.txt","r")
	for line in file:
		fields = line.split()
		u_id = fields[0]
		polarity = float(fields[1])
		if polarity >= -2.5 and polarity < -1.5 :
			polarized_classes[eLib].append(u_id)
		elif polarity >= -1.5 and polarity < -0.5 :
			polarized_classes[mLib].append(u_id)
		elif polarity >= -0.5 and polarity < 0.5 :
			polarized_classes[neut].append(u_id)
		elif polarity >= 0.5 and polarity < 1.5:
			polarized_classes[mCon].append(u_id)
		elif polarity >= 1.5 and polarity < 2.5:
			polarized_classes[eCon].append(u_id)

		userPolarityTrain[u_id] = polarity
	file.close()


def checkForIntersection(filename, userId):

	if os.path.exists(fileName):
		file1 = open(filename,"r")
		userFriends = []
		for line in file1:
			userFriends.append(line[:-1])
		file1.close()
		for id in userFriends:
			if id in userPolarityTrain	:
				if not userId in graph:
					graph[userId] = {"class" : "null", eCon : [], mCon : [], neut :[],mLib: [], eLib :[], ftot :[], "sameClass" : 0, "diffClass" : 0, "diffFrac":0}
					userPolarity = userPolarityTrain[userId]
					if  userPolarity < -1.5 :
						graph[userId]["class"] = eLib
					elif userPolarity >= -1.5 and userPolarity < -0.5 :
						graph[userId]["class"] = mLib
					elif userPolarity >= -0.5 and userPolarity < 0.5 :
						graph[userId]["class"] = neut
					elif userPolarity >= 0.5 and userPolarity < 1.5:
						graph[userId]["class"] = mCon
					elif userPolarity >= 1.5 :
						graph[userId]["class"] = eCon

				fpol = userPolarityTrain[id]
				if fpol >= -2.5 and fpol < -1.5 :
					graph[userId][eLib].append(id)
				elif fpol >= -1.5 and fpol < -0.5 :
					graph[userId][mLib].append(id)
				elif fpol >= -0.5 and fpol < 0.5 :
					graph[userId][neut].append(id)
				elif fpol >= 0.5 and fpol < 1.5:
					graph[userId][mCon].append(id)
				elif fpol >= 1.5 and fpol < 2.5:
					graph[userId][eCon].append(id)
				graph[userId][ftot].append(id)


graph = {}
polarized_classes = {eCon : [], mCon : [], neut :[],mLib: [], eLib :[]}
userPolarityTrain = {}
processList = []
getProcessList()
userKeys()


print len(processList)
print len(userPolarityTrain)
for friend in processList:
	fileName = friend+"_friends"
	checkForIntersection(fileName, friend)
print len(graph)

# for id in random.sample(graph, 50):
# 	print "for id ", id, "  econ: ",len(graph[id][eCon])
# 	print "for id ", id, "  mcon: ",len(graph[id][mCon])
# 	print "for id ", id, "  eLib: ",len(graph[id][eLib])
# 	print "for id ", id, "  mLib: ",len(graph[id][mLib])
# 	print "for id ", id, "  neut: ",len(graph[id][neut])
# 	print "for id ", id, " total ", len(graph[id][ftot])

polarized_classes1 = {eCon : [], mCon : [], neut :[],mLib: [], eLib :[]}
for id in graph:
		computeFraction(id)
		#print graph[id]["diffFrac"]
		polarized_classes1[graph[id]["class"]].append(graph[id]["diffFrac"])

## histogram for each class

for class1 in polarized_classes1:
	print "polarized_classes1[class1] = ", polarized_classes1[class1]
	print "class1 = ", class1
	n, bins, patches = plt.hist(polarized_classes1[class1], normed=1)
	plt.xlabel('Number of users')
	plt.ylabel('Fraction value '+ class1)
	plt.title(r'$\mathrm{Histogram\ of\ fraction}\ \mu=100,\ \sigma=15$')
	#plt.axis([40, 160, 0, 0.03])
	plt.grid(True)
	plt.show()


# G = nx.Digraph()
# for id in random.sample(graph, 50):
# 	if id not in G:
# 		G.add_node(id)
# 	for friend in graph[id]:
# 		if friend not in G:
# 			G.add_node(friend)
# 		G.add_edge(friend, id)

# print G.number_of_edges()
# print G.number_of_nodes()
print "extreme liberal " + str(len(polarized_classes[eLib]))
print "extreme con " + str(len(polarized_classes[eCon]))
print "mod lib " + str(len(polarized_classes[mLib]))
print "mod con ", len(polarized_classes[mCon])


## plot the distribution of classes
# objects = []
# values = []
# for id in polarized_classes:
# 	print id
# 	objects.append(id)
# 	values.append(len(polarized_classes[id]))
#
# y_pos = np.arange(len(objects))
#
# plt.bar(y_pos, values, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Usage')
# plt.title('distribution classes')



plt.show()