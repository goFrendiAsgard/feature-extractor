from Feature_Extractor import *
import pylab as pl
import numpy as np
from sklearn.svm import SVC

# parameters
records                     = extract_csv('synthesis_03.csv')
ommited_class_for_plotting  = ['D','E']
new_features                = [
                               '(f2) / (f1)',
                               '(f1) / (f3)'
                              ]


projection_feature_indexes = [0,1]
records = [item for item in records if item[-1] not in ommited_class_for_plotting]
groups = []
for record in records[1:]:
    if record[-1] not in groups:
        groups.append(record[-1])
group_count = len(groups)
clf = SVC()
#clf = DecisionTreeClassifier(max_depth=group_count-1, random_state=0)
# calculate new_data, label_targets and numeric_targets
new_data = []
old_data = []
label_targets = []
numeric_targets = []
old_features = records[0][:-1]
target_dict = {}
index = 1
for record in records[1:]:
    old_data.append(record[:-1])
    label_target = record[-1]
    label_targets.append(label_target)
    if not label_target in target_dict:
        target_dict[label_target] = index
        index += 1
    numeric_targets.append(target_dict[label_target])
for i in xrange(len(old_data)):
    new_data.append([])
for new_feature in new_features:
    projection = get_projection(new_feature, old_features, old_data)
    for i in xrange(len(old_data)):
        new_data[i].append(projection[i])
# train classifier
clf.fit(new_data, numeric_targets)
# get minimum and maximum x & y
new_data = np.array(new_data)

x_min, x_max = new_data[:, projection_feature_indexes[0]].min() - 0.1, new_data[:, projection_feature_indexes[0]].max() + 0.1
y_min, y_max = new_data[:, projection_feature_indexes[1]].min() - 0.1, new_data[:, projection_feature_indexes[1]].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001),
                     np.arange(y_min, y_max, 0.001))
    
# plotting
pl.subplot(1,1,1)
if len(new_features)>2:
    tup = xx.ravel(), yy.ravel()
    for i in xrange(len(new_features)-2):
        tup = tup + ( [0.5]*len(xx.ravel()) , )
    Z = clf.predict(np.c_[tup])
else:
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
pl.xlabel(new_features[projection_feature_indexes[0]])
pl.ylabel(new_features[projection_feature_indexes[1]])
pl.contourf(xx, yy, Z)
#pl.axis('off')

# Plot also the training points
pl.scatter(new_data[:, projection_feature_indexes[0]], new_data[:, projection_feature_indexes[1]], c=numeric_targets, cmap=plt.cm.gist_rainbow)


prediction = clf.predict(new_data)
correct = 0.0
for i in xrange(len(prediction)):
    if prediction[i] == numeric_targets[i]:
        correct += 1
print correct, len(prediction), correct/len(prediction)

pl.title('plot')
pl.show()

