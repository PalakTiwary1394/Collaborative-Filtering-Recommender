import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
from sqlalchemy import create_engine

# path = 'C:/Users/palak/Desktop/ShopHopper/'
#
# df_meta=pd.read_csv(path + 'products.csv', low_memory=False, encoding='cp1252')
# pd.set_option('display.max_colwidth', 20)

def create_vector_matrix():
    dbConnection = connect_to_db()
    df_meta = pd.read_sql("select id, title, index, gender, product_type from products",
                          dbConnection)
    pd.set_option('display.expand_frame_repr', False);
    disconnect_from_db(dbConnection)

    df = pd.DataFrame()
    df['id'] = df_meta['id']
    df['title'] = df_meta['title']
    df['index'] = df_meta['index']
    df['Gender'] = df_meta['gender']
    df['Product_Type'] = df_meta['product_type']

    sparse_matrix_products = df[["Gender","Product_Type"]]

    return sparse_matrix_products, df

def connect_to_db():
    # establishing the connection
    alchemyEngine = create_engine(
        'postgresql://otjoiayz:WDcK1I9f9hhsx51XD_pAahhE5G5KN7kg@peanut.db.elephantsql.com/otjoiayz', pool_recycle=3600);
    dbConnection = alchemyEngine.connect()
    return dbConnection


def disconnect_from_db(dbConnection):
    dbConnection.close()

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

sparse_matrix_products, df = create_vector_matrix()
list = get_recommendations(sparse_matrix_products)
print("Recommended Products List:",list)
print("\n")

for i in range(0,len(list)):
    print("\n",df.loc[list[i]:list[i]])