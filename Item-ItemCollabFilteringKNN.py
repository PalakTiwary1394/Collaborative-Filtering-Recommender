import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
path = 'C:/Users/palak/Desktop/ShopHopper/'

df_meta=pd.read_csv(path + 'products.csv', low_memory=False, encoding='cp1252')
pd.set_option('display.max_colwidth', 20)

df = pd.DataFrame()
df['id'] = df_meta['id']
df['title'] = df_meta['title']
df['index'] = df_meta['Index']
df['Gender'] = df_meta['Gender']
df['Product_Type'] = df_meta['Product_Type']

sparse_matrix_products = df[["Gender","Product_Type"]]


def get_recommendations(sparse_matrix_products):
    sparse_matrix_products = pd.get_dummies(sparse_matrix_products)

    model = NearestNeighbors(n_neighbors=10, metric='cosine',algorithm='brute', n_jobs=-1)
    model.fit(sparse_matrix_products)

    filename = 'knnpickle_file.pkl'
    pickle.dump(model, open(filename, 'wb'))

    # load the model from disk
    loaded_model = pickle.load(open('knnpickle_file.pkl', 'rb'))
    query_index=10
    distances,indices=loaded_model.kneighbors(sparse_matrix_products.iloc[query_index,:].values.reshape(1,-1))


    list=[query_index]
    for i in range(0,10):
        if i==0:
            print("")
            #print("Recommendation for {0}:".format(sparse_matrix_products.index[query_index]))
        else:
            #print("{0}: {1}".format(i,sparse_matrix_products.index[indices.flatten()[i]]))
            list.append(sparse_matrix_products.index[indices.flatten()[i]])

    return list


list = get_recommendations(sparse_matrix_products)
print("Recommended Products List:",list)
print("\n")

for i in range(0,len(list)):
    print("\n",df.loc[list[i]:list[i]])