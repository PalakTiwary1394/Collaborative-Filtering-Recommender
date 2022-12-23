from sqlalchemy import create_engine

# SQL Queries for Item-Item Collaborative Filtering
getOrderedProduct = "SELECT index	FROM public.products where id in (select \"Product_id\" from public.\"ShopOrders\" where \"User_id\" = 'cl84pz9ki003509kxn2r2fgfi' order by \"Created_At\" desc limit 1);"
selectAllProductsKNN = "select id, title, index, gender,buckets_manual_1, buckets_manual_2,product_type_manual from products"

# SQL Queries for User-User Collaborative Filtering
selectAllProductsSVD = "SELECT \"Index\", \"Order_id\", \"User_id\", \"Product_id\", \"Rating\", \"Created_At\" FROM public.\"ShopOrders\";"


# Establishing DB connection
def connect_to_db():
    # establishing the connection
    alchemyEngine = create_engine(
        'postgresql://otjoiayz:WDcK1I9f9hhsx51XD_pAahhE5G5KN7kg@peanut.db.elephantsql.com/otjoiayz', pool_recycle=3600);
    dbConnection = alchemyEngine.connect()
    return dbConnection


def disconnect_from_db(dbConnection):
    dbConnection.close()
