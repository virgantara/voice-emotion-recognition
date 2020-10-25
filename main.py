import numpy as np
import pandas as pd

import os

from tqdm import tqdm

tqdm.pandas()

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC

from mfcc_extraction import get_mfcc, convert_to_labels

audio_train_files = os.listdir('dataset/audio_train')
audio_test_files = os.listdir('dataset/audio_test')

train = pd.read_csv('dataset/train.csv')
submission = pd.read_csv('dataset/test.csv')



train_data = pd.DataFrame()
train_data['fname'] = train['fname']
test_data = pd.DataFrame()
test_data['fname'] = audio_test_files

train_data = train_data['fname'].progress_apply(get_mfcc, path='dataset/audio_train/')
print('done loading train mfcc')
test_data = test_data['fname'].progress_apply(get_mfcc, path='dataset/audio_test/')
print('done loading test mfcc')

train_data['fname'] = train['fname']
test_data['fname'] = audio_test_files

train_data['label'] = train['label']
test_data['label'] = np.zeros((len(audio_test_files)))

# print(train_data.head())

X = train_data.drop(['label', 'fname'], axis=1)
feature_names = list(X.columns)
X = X.values
labels = np.sort(np.unique(train_data.label.values))
num_class = len(labels)
c2i = {}
i2c = {}
for i, c in enumerate(labels):
    c2i[c] = i
    i2c[i] = c
y = np.array([c2i[x] for x in train_data.label.values])
X_test = test_data.drop(['label', 'fname'], axis=1)
X_test = X_test.values

# print(X.shape)
# print(X_test.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=15).fit(X_scaled)
X_pca = pca.transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)

# print(sum(pca.explained_variance_ratio_))

X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size = 0.2, random_state = 42, shuffle = True)

clf = SVC(kernel = 'rbf', probability=True, C=10, gamma=0.01)

clf.fit(X_train, y_train)
hasil_prediksi = clf.predict(X_val)


print(confusion_matrix(y_val, hasil_prediksi))
print(classification_report(y_val, hasil_prediksi))
print(accuracy_score(y_val, hasil_prediksi))

