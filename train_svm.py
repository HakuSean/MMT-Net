'''
2019-06-02: Need to add bootstrap

'''

from sklearn import svm
from sklearn.metrics import confusion_matrix

import os
import h5py
import ipdb

from opts import parse_opts

args = parse_opts()
assert args.model == 'svm', 'The model should be svm.'
# the hdf5 files are in the result_path + tag.
train = os.path.join(args.result_path, args.tag, 'training.hdf5')
test = os.path.join(args.result_path, args.tag, 'validation.hdf5')

# each h5py file contains images and labels two datasets.
# For both datasets, each row is an instance. 

f_train = h5py.File(train, 'r')
train_data = f_train['images']
train_label = f_train['labels']

f_test = h5py.File(test, 'r') 
test_data = f_test['images']
test_label = f_test['labels']

ipdb.set_trace()

clf = svm.SVC(kernel='linear', class_weight='balanced')
clf.fit(train_data, train_label)

test_pred = clf.predict(test_data)

cf = confusion_matrix(test_label, test_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

print(np.mean(cls_hit/cls_cnt))
precision, recall, F1, specificity = f1_score(outputs_label, targets_label, compute=args.concern_label)
print('Precision (tp/ tp+fp): {:.3f}\t'
        'Recall (tp / tp+fn): {:.3f}\t'
        'F1 (2pr/p+r): {:.3f}\t'
        'Sensitivity (tp / tp+fn):{:.3f}\t'
        'Specificity (tn / tn+fp):{:.3f}'.format(
        precision, recall, F1, recall, specificity))