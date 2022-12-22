import pandas as pd
import seaborn as sns
sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from sqlalchemy import create_engine

df_products = pd.DataFrame()
similarity = ''


def connect_to_db():
    # establishing the connection
    alchemyEngine = create_engine(
        'postgresql://otjoiayz:WDcK1I9f9hhsx51XD_pAahhE5G5KN7kg@peanut.db.elephantsql.com/otjoiayz', pool_recycle=3600);
    dbConnection = alchemyEngine.connect()
    return dbConnection


def disconnect_from_db(dbConnection):
    dbConnection.close()


def create_vector_matrix(df_products):
    dbConnection = connect_to_db()
    df_meta = pd.read_sql("select id, title, index, gender, product_type, buckets, size_new from products",
                          dbConnection)
    pd.set_option('display.expand_frame_repr', False);
    disconnect_from_db(dbConnection)

    df_products['id'] = df_meta['id']
    df_products['title'] = df_meta['title']
    df_products['index'] = df_meta['index']
    df_products['Gender'] = df_meta['gender']
    df_products['Product_Type'] = df_meta['product_type']
    df_products['buckets'] = df_meta['buckets']
    df_products['Sizes'] = df_meta['size_new']

    df_products.dropna(axis=0, how='any', inplace=True)

    df_products['tags'] = df_products[df_products.columns[3:]].apply(
        lambda x: '|'.join(x.dropna().astype(str)),
        axis=1
    )

    df_products['new_id'] = range(0, len(df_products))
    df_products = df_products[['id', 'new_id', 'title', 'index', 'tags']]
    pd.set_option('display.max_colwidth', 500)
    pd.set_option('display.expand_frame_repr', False)
    df_products['tag_len'] = df_products['tags'].apply(lambda x: len(x))
    stop = list(stopwords.words('english'))

    tfidf = TfidfVectorizer(max_features=5000, analyzer='word', stop_words=set(stop))
    vectorized_data = tfidf.fit_transform(df_products['tags'])
    count_matrix = pd.DataFrame(vectorized_data.toarray(), index=df_products['tags'].index.tolist())

    # reduce dimensionality for improved performance
    svd = TruncatedSVD(count_matrix.shape[1])
    reduced_data = svd.fit_transform(count_matrix)

    # compute the cosine similarity matrix
    similarity = cosine_similarity(reduced_data)
    return similarity


# create a function that takes in product title as input and returns a list of the most similar products
def get_recommendations(title, n, cosine_sim=similarity):
    product_index = df_products[df_products.title == title].new_id.values[0]

    # get the pairwsie similarity scores of all products with that product and sort the products based on the
    # similarity scores
    sim_scores_all = sorted(list(enumerate(cosine_sim[product_index])), key=lambda x: x[1], reverse=True)

    # checks if recommendations are limited
    if n > 0:
        sim_scores_all = sim_scores_all[1:n + 1]

    # get the product indices of the top similar products
    product_indices = [i[0] for i in sim_scores_all]
    scores = [i[1] for i in sim_scores_all]

    # return the top n most similar products from the products df
    top_titles_df = pd.DataFrame(df_products.iloc[product_indices]['title'])
    top_titles_df['sim_scores'] = scores
    top_titles_df['ranking'] = range(1, len(top_titles_df) + 1)

    return top_titles_df, sim_scores_all


def Fetch_User_Preference_Product():
    # variables
    gender = 'women'
    product_type = 'accessories'
    buckets = ['Bohemian']
    size_new = ['M']
    print('Initial User Preferences set in the app')
    print(f'Gender: {gender}')
    print(f'Product Type: {product_type}')
    print(f'Style: {buckets}')
    print(f'Size: {size_new}')
    size_input = (', '.join("'{}'".format(x) for x in size_new))

    # cursor = connect_to_db()
    dbConnection = connect_to_db()

    # Executing an MYSQL function using execute() method
    single_row = pd.read_sql(
        f"select title from products where gender ='{gender}' and lower(product_type) like '%%{product_type}%%' and ( buckets[1] "
        f"= '{buckets[0]}') and size_new @>(ARRAY[{size_input}]::TEXT[]) limit 1;", dbConnection)

    single_row = single_row.to_numpy()

    # Fetch a single row using fetchone() method.
    return single_row


product_name = str(Fetch_User_Preference_Product())
product_name = product_name.replace('[', '').replace(']', '')
product_name = product_name[1:len(product_name) - 1]
print(f"Fetch_User_Preference_Product: {product_name}")

similarity = create_vector_matrix(df_products)

number_of_recommendations = 5
top_titles_df, _ = get_recommendations(product_name, number_of_recommendations, similarity)

print(f'Top 5 Recommended_Products: {top_titles_df}')