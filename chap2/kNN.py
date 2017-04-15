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

def autoNorm(dataset): #class(dataset) = numpy matrix
	minVals = dataset.min(0)
	maxVals = dataset.max(0)
	ranges = maxVals - minVals
	normDataset = zeros(shape(dataset))
	m = dataset.shape[0] #number of rows in dataset, i.e number of samplesd
	normDataset = dataset - tile(minVals, (m,1))
	normDataset = normDataset/tile(ranges, (m,1)) # / is element wise division, NOT matrix division
	return normDataset, ranges, minVals

def datingClassTest():
	hoRatio = 0.10
	data, labels = file2matrix('datingTestSet2.txt')
	normdata, ranges, mins = autoNorm(data)
	m = normdata.shape[0]
	numTestCases = int(m*hoRatio)
	errCnt = 0.0
	for i in range(numTestCases):
		classifierResult = classify0(normdata[i,:], normdata[numTestCases:m, :], \
										labels[numTestCases:m] ,3)
		print("The classifier came back with: %d, the real answer is: %d" % (classifierResult, labels[i]))
		if (classifierResult != labels[i]):
			errCnt+=1.0
	print("\nThe total error rate is: %f" % (errCnt/float(numTestCases)))

def classifyPerson():
	resultList = ['not at all','in small doses', 'in large doses']
	percentTats = float(input(\
						"percentage of time spent playing video games?"))
	ffMiles = float(input("frequent flier miles earned per year?"))
	iceCream = float(input("liters of ice cream consumed per year?"))
	datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr-\
						minVals)/ranges,normMat,datingLabels,3)
	print ("You will probably like this person: ",\
				resultList[classifierResult - 1])