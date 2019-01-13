from sklearn import tree

#Weight (grams)		Texture		Label
#-------------------------------------------------
#	155		Rough		Orange
#	180		Rough		Orange
#	135		Smooth		Apple
#	110		Smooth		Apple

#### ------- Creating sample example data and adding it to collection
## converting data into machine understanding format, as Scikit-learn requires numerical features ---------

#features = [[155, "rough"], [180, "rough"], [135, "smooth"], [110, "smooth"]]
# -------Considering rough=0 and smooth=1
features = [[155, 0], [180, 0], [135, 1], [110, 1]]


# ------- considering orange=0 and apple=1
#labels = ["orange", "orange", "apple", "apple"]
labels = [1, 1, 0, 0]

classifier = tree.DecisionTreeClassifier()

classifier = classifier.fit(features, labels)

print (classifier.predict([[120, 0]]))
