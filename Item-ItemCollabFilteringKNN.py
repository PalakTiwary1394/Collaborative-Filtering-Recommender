import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
from sqlalchemy import create_engine

def create_vector_matrix():
    dbConnection = connect_to_db()
    df_meta = pd.read_sql("select id, title, index, gender,buckets_manual_1, buckets_manual_2,product_type_manual from products",
                          dbConnection)
    pd.set_option('display.expand_frame_repr', False);
    disconnect_from_db(dbConnection)

    df = pd.DataFrame()
    df['id'] = df_meta['id']
    df['title'] = df_meta['title']
    df['index'] = df_meta['index']
    df['Gender'] = df_meta['gender']
    df['Product_Type'] = df_meta['product_type_manual']
    df['buckets_1'] = df_meta['buckets_manual_1']
    #df['buckets_2'] = df_meta['buckets_manual_2']
    df.set_index('index', inplace=True)

    sparse_matrix_products = df[["Gender","Product_Type", "buckets_1"]]

    return sparse_matrix_products, df

def connect_to_db():
    # establishing the connection
    alchemyEngine = create_engine(
        'postgresql://otjoiayz:WDcK1I9f9hhsx51XD_pAahhE5G5KN7kg@peanut.db.elephantsql.com/otjoiayz', pool_recycle=3600);
    dbConnection = alchemyEngine.connect()
    return dbConnection


def disconnect_from_db(dbConnection):
    dbConnection.close()

def getOrderedProduct():
    dbConnection = connect_to_db()
    ordered_product = pd.read_sql("SELECT index	FROM public.products where id in (select \"Product_id\" from public.\"ShopOrders\" where \"User_id\" = 'cl84pz9ki003509kxn2r2fgfi' order by \"Created_At\" desc limit 1);",
                                  dbConnection)
    disconnect_from_db(dbConnection)
    print(f'ordered_product: {ordered_product}')
    return ordered_product.values[0][0]

def get_recommendations(sparse_matrix_products, df):
    sparse_matrix_products = pd.get_dummies(sparse_matrix_products)
    ordered_product = getOrderedProduct()

    model = NearestNeighbors(n_neighbors=10, metric='cosine',algorithm='brute', n_jobs=-1)
    model.fit(sparse_matrix_products)

    filename = 'knnpickle_file.pkl'
    pickle.dump(model, open(filename, 'wb'))

    # load the model from disk
    loaded_model = pickle.load(open('knnpickle_file.pkl', 'rb'))


    query_index=ordered_product
    distances,indices=loaded_model.kneighbors(sparse_matrix_products.loc[query_index,:].values.reshape(1,-1))


    list=[query_index]
    for i in range(0,10):
        if i!=0:
            list.append(sparse_matrix_products.index[indices.flatten()[i]])

    return list

sparse_matrix_products, df = create_vector_matrix()
list = get_recommendations(sparse_matrix_products, df)
print("Recommended Products List:",list)
print("\n")

for i in range(0,len(list)):
    print("\n",df.loc[list[i]:list[i]])