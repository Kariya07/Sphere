import argparse
import numpy as np
import matplotlib.pyplot as pl
from tqdm import tqdm
import matplotlib.gridspec as gridspec


# Import the random forest package
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
__author__ = 'salikhovakariya'

def main():
    args = parse_args()

    train_data = np.loadtxt(args.train)
    test_data  = np.loadtxt(args.test)
    

    total_data = np.concatenate(([train_data, test_data]), axis=0)

    #visualize_data(total_data[0::, 1::], len(train_data[:,0]), len(test_data[:,0]))
    number_of_features = len(train_data[0, :])
    use_features_in_tree = (int)(args.features_percent * number_of_features)

    forest = RandomForestClassifier(n_estimators = args.trees, max_features=use_features_in_tree)
    knn = KNeighborsClassifier()
    log_reg = LogisticRegression(random_state=0, max_iter=1000)
    gnb = GaussianNB()
    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    for name, model in zip(['Random Forest:', 'KNN:', 'LogReg:', 'Naive Bayes:', 'SVM:'], [forest, knn, log_reg, gnb, svm]):
        prediction = model.fit(train_data[0::, 1::], train_data[0::, 0]).predict(test_data[0::, 1::])
        print (classification_report(name, test_data[0::, 0], prediction))
        
    gridSearch_rf(train_data, test_data)


def visualize_data(total_data, train_size, test_size):
    ''' Visualization of total spam data
    :param total_data: Train and test data
    :param train_size: Size of train set
    :param test_size: Size of test set
    :return:
    '''
    pca = PCA(n_components=2)
    projection = pca.fit_transform(total_data)

    fig = pl.figure(figsize=(8, 8))

    pl.rcParams['legend.fontsize'] = 10
    pl.plot(projection[0:train_size, 0], projection[0:train_size, 1],
            'o', markersize=7, color='blue', alpha=0.5, label='Train')
    pl.plot(projection[train_size:train_size+test_size, 0], projection[train_size:train_size+test_size, 1],
            'o', markersize=7, color='red', alpha=0.5, label='Test')
    pl.title('Spam data')
    pl.show()


def classification_report(classificator_type, y_true, y_pred):
    ''' Computes clasification metrics

    :param y_true - original class label
    :param y_pred - predicted class label
    :return presicion, recall for each class; micro_f1 measure, macro_f1 measure
    '''
    report = classificator_type + '\n'
    last_line_heading = 'avg / total'
    final_line_heading = 'final score'

    labels = unique_labels(y_true, y_pred)

    width = len(last_line_heading)
    target_names = ['{0}'.format(l) for l in labels]

    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report += fmt % tuple(headers)
    report += '\n'

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None)

    f1_macro = 0
    precision_macro = 0
    recall_macro = 0

    for i, label in enumerate(labels):
        values = [target_names[i]]
        f1_macro += f1[i]
        precision_macro += p[i]
        recall_macro += r[i]
        for v in (p[i], r[i], f1[i]):
            values += ["{0:0.5f}".format(v)]
        values += ["{0}".format(s[i])]
        report += fmt % tuple(values)

    report += '\n'

    # compute averages
    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["{0:0.5f}".format(v)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)

    values = [final_line_heading]
    for v in (precision_macro, recall_macro, f1_macro):
        values += ["{0:0.5f}".format(v / labels.size)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)

    return report

def parse_args():
    parser = argparse.ArgumentParser(description='Random Forest Tutorial')
    parser.add_argument("-tr", "--train", action="store", type=str, help="Train file name")
    parser.add_argument("-te", "--test", action="store", type=str, help="Test file name")
    parser.add_argument("-t", "--trees", action="store", type=int, help="Number of trees in random forest", default=10)
    parser.add_argument("-fp", "--features_percent", action="store", type=float, help="Percent of features in each tree", default=0.9)
    return parser.parse_args()

def gridSearch_rf(train_data, test_data):
    number_of_features = len(train_data[0, :])
    num_of_trees_list = range(10, 160, 10)
    features_percent_list = np.linspace(0.1, 0.99, 10)
    precision = []
    recall = []
    F1 = []
    x = []
    
    for num_of_trees in tqdm(num_of_trees_list):
        for features_percent in features_percent_list:
            x += ["({0}, {1:0.2f})".format(num_of_trees, features_percent)]
            use_features_in_tree = int(features_percent * number_of_features)
            forest = RandomForestClassifier(n_estimators = num_of_trees, max_features=use_features_in_tree)
            prediction = forest.fit(train_data[0::, 1::], train_data[0::, 0]).predict(test_data[0::, 1::])
            labels = unique_labels(test_data[0::, 0], prediction)
            p, r, f1, _ = precision_recall_fscore_support(test_data[0::, 0], prediction,
                                                  labels=labels,
                                                  average=None)
            f1_macro = 0
            precision_macro = 0
            recall_macro = 0
            for i, _ in enumerate(labels):
                f1_macro += f1[i]
                precision_macro += p[i]
                recall_macro += r[i]
            for list_name, v in zip([precision, recall, F1],[precision_macro, recall_macro, f1_macro]):
                list_name += [v / labels.size]
                
    max_f1_idx = np.argmax(F1)
    print('Best f1 in', x[max_f1_idx], ':', "{0:0.2f}".format(F1[max_f1_idx]))
    max_recall_idx = np.argmax(recall)
    print('Best recall in', x[max_recall_idx], ':', "{0:0.2f}".format(recall[max_recall_idx]))
    max_precision_idx = np.argmax(precision)
    print('Best precision in', x[max_precision_idx], ':', "{0:0.2f}".format(precision[max_precision_idx]))
    print('Best parameters (number of trees, features percent):', x[max_f1_idx])
    
    AX = gridspec.GridSpec(3,1)
    AX.update(hspace = 0.8)
    ax_1  = pl.subplot(AX[0,:])
    ax_2 = pl.subplot(AX[1,:])
    ax_3 = pl.subplot(AX[2,:])
    i = 0
    for ax in [ax_1, ax_2, ax_3]:
        ax.plot(x[i:i+50], precision[i:i+50])
        ax.plot(x[i:i+50], recall[i:i+50])
        ax.plot(x[i:i+50], F1[i:i+50])
        if max_f1_idx in range(i, i+50, 1):
            ax.plot(x[max_f1_idx], F1[max_f1_idx], 'o', markersize=7, color='red')
        ax.legend(['precision', 'recall', 'F1'])
        ax.grid(axis = 'x')
        ax.tick_params(labelrotation=90, axis='x')
        i += 50
    pl.show()

if __name__ == "__main__":
    main()