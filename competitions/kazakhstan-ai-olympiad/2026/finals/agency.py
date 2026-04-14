import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

FEATURE_PREFIX = "f_"

def get_feature_columns(df):
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]

train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")
subm = pd.read_csv("sample_output.csv")

train.shape, test.shape, subm.shape

train.head()

train['author_id'].value_counts()

train['author_id'].nunique()

features = get_feature_columns(train)
target_col = 'author_id'

print(f"Using {len(features)} features")

avg_per_author = len(train) / train["author_id"].nunique()
n_clusters = max(2, int(round(len(test) / avg_per_author)))
print(
    f"Estimated {n_clusters} clusters (avg {avg_per_author:.1f} submissions/author)"
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

from sklearn.model_selection import train_test_split 

X, y = train[features].values, train[target_col].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

from sklearn.metrics import adjusted_rand_score

score = adjusted_rand_score(y_valid, model.predict(X_valid))

print(f'Score: {score:.6f}')

plt.bar(range(len(features)), model.feature_importances_)
plt.show()

num_to_leave = 29
features = [features[i] for i in np.argsort(model.feature_importances_)[::-1][:num_to_leave]]
len(features)

from sklearn.cluster import KMeans

model = KMeans(n_clusters=int(n_clusters*0.9), n_init=10, random_state=42, algorithm='elkan')
# model.fit(train[features].values)
# labels = model.fit_predict(np.vstack([train[features].values, test[features].values]))[len(train):]
labels = model.fit_predict(test[features])
labels.shape

submission = pd.DataFrame(
    {
        "subtaskID": test["subtaskID"].values,
        "datapointID": test["datapointID"].values,
        "answer": labels,
    }
)

submission.to_csv('submission.csv', index=False)
print(f"Wrote {len(submission)} predictions")
print(f"Clusters: {len(set(labels))}, submissions: {len(labels)}")