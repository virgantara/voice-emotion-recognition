import numpy as np
import pandas as pd

import os
import librosa

import scipy
from scipy.stats import skew
from tqdm import tqdm, tqdm_pandas

tqdm.pandas()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC

audio_train_files = os.listdir('dataset/audio_train')
audio_test_files = os.listdir('dataset/audio_test')

train = pd.read_csv('dataset/train.csv')
submission = pd.read_csv('dataset/test.csv')

SAMPLE_RATE = 44100
def get_mfcc(name, path):
    data, _ = librosa.core.load(path + name, sr = SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        ft1 = librosa.feature.mfcc(data, sr = SAMPLE_RATE, n_mfcc=30)
        ft2 = librosa.feature.zero_crossing_rate(data)[0]
        ft3 = librosa.feature.spectral_rolloff(data)[0]
        ft4 = librosa.feature.spectral_centroid(data)[0]
        ft5 = librosa.feature.spectral_contrast(data)[0]
        ft6 = librosa.feature.spectral_bandwidth(data)[0]
        ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.median(ft1, axis = 1), np.min(ft1, axis = 1)))
        ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
        ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
        ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
        ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
        ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
        return pd.Series(np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc)))
    except:
        print('bad file')
        return pd.Series([0]*210)

def convert_to_labels(preds, i2c, k=3):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans, ids

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
# print(accuracy_score(clf.predict(X_val), y_val))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_val, hasil_prediksi))
print(classification_report(y_val, hasil_prediksi))
print(accuracy_score(y_val, hasil_prediksi))

