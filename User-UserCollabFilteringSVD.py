import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.decomposition import TruncatedSVD

def connect_to_db():
    # establishing the connection
    alchemyEngine = create_engine(
        'postgresql://otjoiayz:WDcK1I9f9hhsx51XD_pAahhE5G5KN7kg@peanut.db.elephantsql.com/otjoiayz', pool_recycle=3600);
    dbConnection = alchemyEngine.connect()
    return dbConnection

def disconnect_from_db(dbConnection):
    dbConnection.close()

#Loading the dataset
dbConnection = connect_to_db()
product_ratings = pd.read_sql("SELECT \"Index\", \"Order_id\", \"User_id\", \"Product_id\", \"Rating\", \"Created_At\" FROM public.\"ShopOrders\";",
                                  dbConnection)
disconnect_from_db(dbConnection)

product_ratings = product_ratings.dropna()
product_ratings.head()

print("Amazon ratings shape -->>",product_ratings.shape)

popular_products = pd.DataFrame(product_ratings.groupby('Product_id')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
print("Top 3 popular products",most_popular.head(3))

#create a subset
product_ratings1=product_ratings.head(10000)

ratings_utility_matrix = product_ratings1.pivot_table(values='Rating', index='User_id', columns='Product_id', fill_value=0)
ratings_utility_matrix.head()
#print(ratings_utility_matrix.head())

print("Ratings utility matrix shape -->>",ratings_utility_matrix.shape)

#Transposing utility matrix

X = ratings_utility_matrix.T
print("Transpose of ratings utility matrix",X.head())

print("Transpose shape-->>",X.shape)

#Unique products in subset of data
X1 = X

#Decomposing the matrix
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
print("Decomposed matrix shape --->>",decomposed_matrix.shape)

#Correlation Matrix
correlation_matrix = np.corrcoef(decomposed_matrix)
print("Correlation matrix shape -->>",correlation_matrix.shape)

#Choosing random product
print("Randomly chosen product -->> ",X.index[99])

# chosen product
i = "7748554490070"

product_names = list(X.index)
# print(product_names)
product_ID = product_names.index(i)
print("Randomly chosen product ID-->> ",product_ID)

#Correlation for all items with the item purchased by
# this customer based on items rated by other customers people who bought the same product
correlation_product_ID = correlation_matrix[product_ID]
print("Correlation product ID shape -->> ",correlation_product_ID.shape)


#Top 10 highly correlated products
Recommend = list(X.index[correlation_product_ID > 0.90])

# Remove the item already bought by the customer
Recommend.remove(i)

print("Top 10 recommendations",Recommend[0:9])




