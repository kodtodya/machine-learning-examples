import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import graphviz
import pydot
iris = load_iris()

#Data is available @ https://en.wikipedia.org/wiki/Iris_flower_data_set

# just printing the names of features from loaded data
#print(iris.feature_names)

# just printing the names of targets from loaded data.. i.e. flower names
#print(iris.target_names)

# just printing the first example data from loaded data
#print(iris.data[0])

# just printing the first target from loaded data
#print(iris.target[0])

#for i in range(len(iris.target)):
#    print("Example %d: label %s , features %s" % (i, iris.target[i], iris.data[i]))

test_idx = [0, 50, 100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

classifier = tree.DecisionTreeClassifier()
classifier.fit(train_data, train_target)

#just printing testing target values
#print(test_target)

#just printing classifier prdictions
#print(classifier.predict(test_data))

dot_data = StringIO()

tree.export_graphviz(classifier, out_file=dot_data,
                      feature_names=iris.feature_names,
                      class_names=iris.target_names,
                      filled=True, rounded=True,
                      special_characters=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())

graph[0].write_pdf("iris.pdf") 
