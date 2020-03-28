import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

mode_dict = {0: "standing", 1: "walking", 2: "running"}

data_s = pd.read_csv("./dataset_s.csv")  # standing 0
data_w = pd.read_csv("./dataset_w.csv")  # walking 1
data_r = pd.read_csv("./dataset_r.csv")  # running 2

# drop the data if gyro_x/y/z is zero
data_s_1 = data_s[data_s['gyro_x'] != 0]
data_w_1 = data_w[data_w['gyro_x'] != 0]
data_r_1 = data_r[data_r['gyro_x'] != 0]

frames = [data_s_1, data_w_1, data_r_1]
data = pd.concat(frames)
pd.set_option('display.max_rows', None)
data.to_csv("./data_all.csv")
data.head()

labels = data.activity
data_dropped = data.drop(["time", "activity"], axis=1)
features = data_dropped.values
data_dropped.head()

FEATURES = features
LABELS = labels.values

# import ipdb
# ipdb.set_trace()

X_embedded = TSNE(n_components=2).fit_transform(FEATURES)

# x_train,x_test,y_train,y_test=train_test_split(FEATURES,LABELS,test_size=0.3,random_state=1)
# RanFor=RandomForestClassifier(n_estimators=100,random_state=1)

# RanFor.fit(x_train,y_train)
# print("Train score is: {}".format(accuracy_score(y_train,RanFor.predict(x_train))))
# print("Test score {}".format(accuracy_score(y_test,RanFor.predict(x_test))))
STR_LABELS = list(map(mode_dict.get, list(LABELS)))
import seaborn as sns
sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 3)
sns.scatterplot(X_embedded[:, 0],
                X_embedded[:, 1],
                hue=STR_LABELS,
                legend='full',
                palette=palette)
plt.show()