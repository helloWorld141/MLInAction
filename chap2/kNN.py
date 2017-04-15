from numpy import * #matrix class
import operator

def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize,1)) - dataSet #numpy operation return array
													# for i in range (dataSetSize): (inX - x[i])
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1) #numpy array operation
	distances = sqDistances**0.5
	sortedDistanceIndicies = distances.argsort() #save the old index of the currently sorted array
	classCount = {} # there are 2 classes in this example, so there are 2 pair in this dict
	for i in range(k):
		voteIlabel = labels[sortedDistanceIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	sortedClassCount = sorted(classCount.items(),	#iteritems in python2
								key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr = open(filename)
	numberOfLines = len(fr.readlines())
	returnMat = zeros((numberOfLines, 3)) #numpy operation
	classLabelVector = []
	fr = open(filename)
	index = 0
	for line in fr.readlines():
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index+=1
	return returnMat, classLabelVector