import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import Utility as util

def create_vector_matrix():
    dbConnection = util.connect_to_db()
    df_meta = pd.read_sql(util.selectAllProductsKNN,dbConnection)
    pd.set_option('display.expand_frame_repr', False);
    util.disconnect_from_db(dbConnection)

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


def getOrderedProduct():
    dbConnection = util.connect_to_db()
    ordered_product = pd.read_sql(util.getOrderedProduct,
                                  dbConnection)
    util.disconnect_from_db(dbConnection)
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

    distances,indices=loaded_model.kneighbors(sparse_matrix_products.loc[ordered_product,:].values.reshape(1,-1))


    list=[ordered_product]
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