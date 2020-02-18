import numpy as np

from classifier.dataset import ProcessedDataset
from input.regressors import filter_single_class
from input.utils import resample

print('loading features')
source_dir = '/data/user/teodoro/uniprot/dataset/no_large/processed/no_tag'
notag_set = ProcessedDataset(source_dir, type='train').get_content()

source_dir = '/data/user/teodoro/uniprot/dataset/no_large/processed/tag'
tag_set = ProcessedDataset(source_dir, type='train').get_content()

source_dir = '/data/user/teodoro/uniprot/dataset/no_large/processed/both'
both_set = ProcessedDataset(source_dir, type='train').get_content()

print('filtering features')
_, notag_targets, notag_regressors = filter_single_class(notag_set[0], notag_set[1], notag_set[2])
_, tag_targets, tag_regressors = filter_single_class(tag_set[0], tag_set[1], tag_set[2])
_, both_targets, both_regressors = filter_single_class(both_set[0], both_set[1], both_set[2])

print('resamling features')
notag_targets, notag_regressors = resample(notag_regressors, notag_targets)
tag_targets, tag_regressors = resample(tag_regressors, tag_targets)
both_targets, both_regressors = resample(both_regressors, both_targets)

print('saving features')
np.save('/data/user/teodoro/uniprot/dataset/no_large/processed/no_tag/train_features_rs.npy', notag_regressors)
np.savetxt('/data/user/teodoro/uniprot/dataset/no_large/processed/no_tag/train_labels_rs.csv', notag_targets,
           delimiter=',', fmt='%i')

np.save('/data/user/teodoro/uniprot/dataset/no_large/processed/tag/train_features_rs.npy', tag_regressors)
np.savetxt('/data/user/teodoro/uniprot/dataset/no_large/processed/tag/train_labels_rs.csv', tag_targets, delimiter=',',
           fmt='%i')

np.save('/data/user/teodoro/uniprot/dataset/no_large/processed/both/train_features_rs.npy', both_regressors)
np.savetxt('/data/user/teodoro/uniprot/dataset/no_large/processed/both/train_labels_rs.csv', both_targets,
           delimiter=',', fmt='%i')
