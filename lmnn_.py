from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from pylmnn import LargeMarginNearestNeighbor as LMNN
import time
import numpy as np

print('Start!!!')

x = np.load('data/feature_2048.npy')
y = np.load('data/label.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, stratify=y, random_state=10)

acc1 = []
acc2 = []
acc3 = []
acc4 = []
T = []
T1 = []
T2 = []
T3 = []
T4 = []

for k in [9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29]:
    print('Running K={} ... ... '.format(k))

    t0 = time.time()
    lmnn = LMNN(n_neighbors=k, max_iter=200, n_components=x.shape[1])
    lmnn.fit(x_train, y_train)
    x_train_ = lmnn.transform(x_train)
    x_test_ = lmnn.transform(x_test)
    t1 = time.time()
    T.append(t1 - t0)
    print('LMNN Cost:', t1 - t0)

    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='cosine', algorithm='brute')
    knn.fit(x_train_, y_train)
    lmnn_acc = knn.score(x_test_, y_test)
    acc1.append(lmnn_acc)
    t2 = time.time()
    T1.append(t2 - t1)
    print('cosine Cost:', t2 - t1, '|accuracy:', lmnn_acc)

    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean', algorithm='brute')
    knn.fit(x_train_, y_train)
    lmnn_acc = knn.score(x_test_, y_test)
    acc2.append(lmnn_acc)
    t3 = time.time()
    T2.append(t3 - t2)
    print('euclidean Cost:', t3 - t2, '|accuracy:', lmnn_acc)

    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='manhattan', algorithm='brute')
    knn.fit(x_train_, y_train)
    lmnn_acc = knn.score(x_test_, y_test)
    acc3.append(lmnn_acc)
    t4 = time.time()
    T3.append(t4 - t3)
    print('manhattan Cost:', t4 - t3, '|accuracy:', lmnn_acc)

    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='chebyshev', algorithm='brute')
    knn.fit(x_train_, y_train)
    lmnn_acc = knn.score(x_test_, y_test)
    acc4.append(lmnn_acc)
    t5 = time.time()
    T4.append(t5 - t4)
    print('chebyshev Cost:', t5 - t4, '|accuracy:', lmnn_acc)

acc1 = np.array(acc1)
acc2 = np.array(acc2)
acc3 = np.array(acc3)
acc4 = np.array(acc4)
T = np.array(T)
T1 = np.array(T1)
T2 = np.array(T2)
T3 = np.array(T3)
T4 = np.array(T4)
print(acc1)
print(acc2)
print(acc3)
print(acc4)
print(T)
print(T1)
print(T2)
print(T3)
print(T4)
np.savetxt('result/Lmnn_acc1.csv', acc1)
np.savetxt('result/Lmnn_acc2.csv', acc2)
np.savetxt('result/Lmnn_acc3.csv', acc3)
np.savetxt('result/Lmnn_acc4.csv', acc4)
np.savetxt('result/Lmnn_T0.csv', T)
np.savetxt('result/Lmnn_T1.csv', T1)
np.savetxt('result/Lmnn_T2.csv', T2)
np.savetxt('result/Lmnn_T3.csv', T3)
np.savetxt('result/Lmnn_T4.csv', T4)
