import matplotlib
import numpy as np
from classifier.dataset import ProcessedDataset
from input.regressors import filter_single_class
from input.utils import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

matplotlib.use('Agg')
import matplotlib.pyplot as plt

source_dir = '/data/user/teodoro/uniprot/dataset/no_large/processed/no_tag'
train_set = ProcessedDataset(source_dir, type='train').get_content()
dev_set = ProcessedDataset(source_dir, type='dev').get_content()
test_set = ProcessedDataset(source_dir, type='test').get_content()

_, train_targets, train_regressors = filter_single_class(train_set[0], train_set[1], train_set[2])
_, dev_targets, dev_regressors = filter_single_class(dev_set[0], dev_set[1], dev_set[2])
_, test_targets, test_regressors = filter_single_class(test_set[0], test_set[1], test_set[2])

train_targets, train_regressors = resample(train_regressors, train_targets)
train_regressors = MinMaxScaler().fit_transform(train_regressors)
dev_regressors = MinMaxScaler().fit_transform(dev_regressors)
test_regressors = MinMaxScaler().fit_transform(test_regressors)

full_col = np.vstack((train_regressors, dev_regressors))
full_col = np.vstack((full_col, test_regressors))

pca = PCA(n_components=2)
X_col = pca.fit_transform(full_col)

colors = []
markers = []
for i in range(len(train_regressors)):
    colors.append('red')
    markers.append('o')
for i in range(len(dev_regressors)):
    colors.append('green')
    markers.append('*')
for i in range(len(test_regressors)):
    colors.append('blue')
    markers.append('+')

fig = plt.figure(figsize=(20, 18))
plt.scatter(X_col[:, 0], X_col[:, 1], c=colors, marker='+', alpha=0.3)
plt.xlabel('component 1')
plt.ylabel('component 2')
fig.savefig('pca_no_tag_cols.png', bbox_inches='tight')
plt.close()
