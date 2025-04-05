import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Вхідний файл, який містить дані
input_file = "census_income/adult.data"

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape)
for i in range(X.shape[1]):
    if X[0, i].isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoder.append(encoder)

X_features = X_encoded[:, :-1].astype(int)
y_labels = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=5)

# Поліноміальне ядро
classifier_poly = SVC(kernel='poly', degree=8)
classifier_poly.fit(X_train, y_train)
y_test_pred_poly = classifier_poly.predict(X_test)

# Оцінка
accuracy_poly = accuracy_score(y_test, y_test_pred_poly)
precision_poly = precision_score(y_test, y_test_pred_poly, average='weighted')
recall_poly = recall_score(y_test, y_test_pred_poly, average='weighted')
f1_poly = cross_val_score(classifier_poly, X_features, y_labels, scoring='f1_weighted', cv=3)

# Вивід результатів
print("Poly Kernel Results")
print("Accuracy:", round(100 * accuracy_poly, 2), "%")
print("Precision:", round(100 * precision_poly, 2), "%")
print("Recall:", round(100 * recall_poly, 2), "%")
print("F1 score:", round(100 * f1_poly.mean(), 2), "%")