import pandas as pd
import numpy as np
import scipy as sp

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz, ExtraTreeClassifier


def get_feature_relevance(X, y):
    vect = DictVectorizer()
    X = vect.fit_transform(X)
    
    lr = LogisticRegression()
    lr.fit(X, y)

    return zip(lr.classes_, vect.inverse_transform(lr.coef_))


def get_feature_relevance_tree(X, y):
    vect = DictVectorizer()
    X = vect.fit_transform(X)
    
    tree = ExtraTreeClassifier(criterion='entropy')
    tree.fit(X, y)

    return zip(['general'], vect.inverse_transform(tree.feature_importances_.reshape(-1,1)))


def get_decision_tree(X, y, depth=None):
    vect = DictVectorizer()
    X = vect.fit_transform(X)
    
    tree = ExtraTreeClassifier(max_depth=depth)
    tree.fit(X, y)

    return export_graphviz(tree, feature_names=vect.feature_names_, class_names=tree.classes_, filled=True)


def get_correlation(X):
    vect = DictVectorizer(sparse=False)
    X = vect.fit_transform(X)
    
    return pd.DataFrame(vect.inverse_transform(X)).fillna(0).corr()
