import numpy as np
from scipy.stats import ttest_ind

from sklearn.metrics import precision_score, recall_score

from input.utils import get_class_map

query_id_cnn = "/data/user/teodoro/uniprot/results/pub_res/tag/cnn_0.03162277660168379.qid"
query_res_cnn = "/data/user/teodoro/uniprot/results/pub_res/tag/cnn_0.03162277660168379.res"

query_id_svm = "/data/user/teodoro/uniprot/dataset/pub_data/processed/tag/test_docs.csv"
query_res_svm = "/data/user/teodoro/uniprot/results/pub_res/tag/svm_4.641588833612782e-05.res"

query_id_logistic = "/data/user/teodoro/uniprot/dataset/pub_data/processed/tag/test_docs.csv"
query_res_logistic = "/data/user/teodoro/uniprot/results/pub_res/tag/logistic_1e-05.res"


query_id_cnn_notag = "/data/user/teodoro/uniprot/results/pub_res/no_tag/cnn_4.641588833612782e-05.qid"
query_res_cnn_notag = "/data/user/teodoro/uniprot/results/pub_res/no_tag/cnn_4.641588833612782e-05.res"

query_id_svm_notag = "/data/user/teodoro/uniprot/dataset/pub_data/processed/no_tag/test_docs.csv"
query_res_svm_notag = "/data/user/teodoro/uniprot/results/pub_res/no_tag/svm_1e-05.res"

query_id_logistic_notag = "/data/user/teodoro/uniprot/dataset/pub_data/processed/no_tag/test_docs.csv"
query_res_logistic_notag = "/data/user/teodoro/uniprot/results/pub_res/no_tag/logistic_4.641588833612782e-05.res"


def compute_score(cmap, query_id, query_res):
    qid = []
    res = {}

    with open(query_id) as f:
        for line in f.readlines():
            line = line.strip()
            qid.append(line)

    with open(query_res) as f:
        for i, line in enumerate(f):
            query = qid[i]
            prediction = np.array([[int(float(i)) for i in line.strip().split(",")]])
            labels = np.array([cmap[qid[i]]])

            res[query] = {}
            prec_micro = precision_score(labels, prediction, average='micro')
            rec_micro = recall_score(labels, prediction, average='micro')

            prec_macro = precision_score(labels, prediction, average='macro')
            rec_macro = recall_score(labels, prediction, average='macro')

            res[query]["prec_micro"] = prec_micro
            res[query]["prec_macro"] = prec_macro
            res[query]["rec_micro"] = rec_micro
            res[query]["rec_macro"] = rec_macro
            res[query]["f1_micro"] = 2 * (prec_micro * rec_micro) / (prec_micro + rec_micro) if (prec_micro + rec_micro) != 0 else 0
            res[query]["f1_macro"] = 2 * (prec_macro * rec_macro) / (prec_macro + rec_macro) if (prec_macro + rec_macro) != 0 else 0

            if i % 5000 == 0:
                print(i)

    return res


def compute_p(res1: dict, res2: dict):
    for metric in ["prec_micro", "prec_macro", "rec_micro", "rec_macro", "f1_micro", "f1_macro"]:
        vals1 = []
        vals2 = []
        for key in res1.keys():
            vals1.append(res1[key][metric])
            vals2.append(res2[key][metric])

        ttest, pval = ttest_ind(vals1, vals2)
        print("%s p-value: %.5f" %(metric, pval))

cmap = get_class_map()

score_svm = compute_score(cmap, query_id_svm, query_res_svm)
score_svm_notag = compute_score(cmap, query_id_svm_notag, query_res_svm_notag)

print("pval svm tag no_tag")
compute_p(score_svm, score_svm_notag)

score_cnn = compute_score(cmap, query_id_cnn, query_res_cnn)
score_cnn_notag = compute_score(cmap, query_id_cnn_notag, query_res_cnn_notag)

print("pval cnn tag no_tag")
compute_p(score_cnn, score_cnn_notag)

print("pval cnn svm tag")
compute_p(score_cnn_notag, score_svm)

score_logistic = compute_score(cmap, query_id_logistic, query_res_logistic)

print("pval cnn logistig tag")
compute_p(score_cnn, score_logistic)

# score_logistic_notag = compute_score(cmap, query_id_logistic_notag, query_res_logistic_notag)
# print("pval cnn svm tag", compute_p(score_cnn_notag, score_svm_notag))
# print("pval cnn logistic tag", compute_p(score_cnn_notag, score_logistic_notag))
# print("pval cnn logistic tag", compute_p(score_cnn, score_logistic))