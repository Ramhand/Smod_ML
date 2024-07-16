import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import re
import pickle
import warnings


class Housing_Model:
    def __init__(self):
        self.model = KMeans(n_clusters=6, random_state=42)  # algorithm='elkan')
        self.vault = []
        self.submodels = [LogisticRegression, KNeighborsClassifier, RandomForestClassifier]
        self.data_init()
        self.model.fit(self.train)
        self.vault.append(self.model)
        self.test['cluster'] = self.model.predict(self.test)
        self.train['cluster'] = self.model.labels_
        self.data['cluster'] = self.model.predict(
            pd.DataFrame(StandardScaler().fit_transform(self.data), index=self.data.index, columns=self.data.columns))
        fig, axs = plt.subplots(3, 3, figsize=(5, 7))
        tables = [self.data, self.train, self.test]
        columns = ['Latitude', 'Longitude', 'MedInc']
        for i in range(len(tables)):
            for j in range(len(columns)):
                j2 = j + 1
                if j2 == len(columns):
                    j2 = 0
                sns.scatterplot(data=tables[i], x=columns[j], y=columns[j2], hue='cluster', ax=axs[j, i])
        plt.show()
        self.after()
        with open('vault.dat', 'wb') as file:
            pickle.dump(self.vault, file)

    def after(self):
        self.train_x = self.train.drop(columns='cluster')
        self.train_y = self.train['cluster']
        self.test_x = self.test.drop(columns='cluster')
        self.test_y = self.test['cluster']
        self.submodel_names = [i.replace('.', '').replace("'", '') for j in str(self.submodels).split('>') for i in
                               re.findall(r"[.]\w+'", j)]
        for i in range(len(self.submodels)):
            smod = self.submodels[i]()
            pred = smod.fit(self.train_x, self.train_y).predict(self.test_x)
            acc = accuracy_score(self.test_y, pred)
            self.vault.append(smod)
            print(f'{self.submodel_names[i]} Accuracy: {acc}')

    def data_init(self):
        try:
            with open('chd.dat', 'rb') as file:
                data = pickle.load(file)
        except FileNotFoundError:
            data = pd.read_csv(
                'https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv')
            data.drop_duplicates(inplace=True)
            with open('chd.dat', 'wb') as file:
                pickle.dump(data, file)
        finally:
            self.trim(data)

    def trim(self, data):
        if len(data.columns) == 3 or len(data.columns) == 4:
            return
        else:
            data = data[['Latitude', 'Longitude', 'MedInc']]
        self.data = data
        self.train, self.test = train_test_split(self.data, test_size=.2, random_state=42)
        scaler = StandardScaler()
        self.train_s = scaler.fit_transform(self.train)
        self.test_s = scaler.transform(self.test)
        self.train = pd.DataFrame(self.train_s, index=self.train.index, columns=self.train.columns)
        self.test = pd.DataFrame(self.test_s, index=self.test.index, columns=self.test.columns)


if __name__ == '__main__':
    Housing_Model()
