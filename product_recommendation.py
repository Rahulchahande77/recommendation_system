# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import time
import turicreate as tc
from sklearn.model_selection import train_test_split

import sys
# sys.path.append("..")
retail_data = pd.read_csv("dataset/catalog_product.csv",encoding = "ISO-8859-1",dtype='unicode',low_memory=False)
retail_data = retail_data[['sku','name','weight','product_online','price','qty','is_in_stock']]
train_data_obtain = None
test_data_obtain = None

def create_data_dummy(data):
    data_dummy = data.copy()
    return data_dummy


def normalize_data(data):
    data = data.dropna(how='any')
    data['price_element'] = (data['price'].astype(float)/50).astype(int)
    # d = df_matrix_ele.reset_index()
    # d.index.names = ['price_element']
    df_matrix = pd.pivot_table(data, values='price_element', index=['sku'], columns=['weight'],aggfunc='mean', fill_value=0)
    df_matrix = df_matrix[(df_matrix.T!=0).any()]
    df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())
    d = df_matrix_norm.reset_index()
    d.index.names = ['scaled_avalable_freq']       
    return pd.melt(d, id_vars=['sku'], value_name='scaled_avalable_freq').dropna(how='any')


def split_data(data):
    # global train_data_obtain,test_data_obtain
    '''
    Splits dataset into training and test set.
    
    Args:
        data (pandas.DataFrame)
        
    Returns
        train_data (tc.SFrame)
        test_data (tc.SFrame)

    '''
    train, test = train_test_split(data, test_size = .2,random_state=1)
    # test_data_obtain = tc.SFrame(test)
    # train_data_obtain = tc.SFrame(train)
    test_data_obtain = test
    train_data_obtain = train
    return train_data_obtain,test_data_obtain
    
def model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display):
    try:
        train_data['sku'] = train_data['sku'].astype(str)

        if name == 'popularity':
            model = tc.popularity_recommender.create(train_data,user_id=user_id, item_id=item_id, target=target)
        elif name == 'cosine':
            model = tc.item_similarity_recommender.create(train_data, 
                                                        user_id=user_id, 
                                                        item_id=item_id, 
                                                        target=target, 
                                                        similarity_type='cosine')
        elif name == 'pearson':
            model = tc.item_similarity_recommender.create(train_data, 
                                                        user_id=user_id, 
                                                        item_id=item_id, 
                                                        target=target, 
                                                        similarity_type='pearson')
            
        # recom = model.recommend(users=users_to_recommend, k=n_rec)
        # recom.print_rows(n_display)
        return model
    except Exception as e:
        print("Error ==> ",str(e))
nor_retail_data = normalize_data(retail_data)
print("***********************************************",nor_retail_data.head)
dumy_retail_data = create_data_dummy(nor_retail_data)
user_id = 'sku'
item_id = 'weight' 
n_rec = 10 # number of items to recommend
n_display = 30 # to display the first few rows in an output dataset
train_data_obtain,test_data_obtain = split_data(dumy_retail_data)
users_to_recommend = list(test_data_obtain[user_id])
name = 'cosine'
target = 'scaled_avalable_freq'
train_data_obtain.dropna(how='any')
train_data_obtain[target] = train_data_obtain[target].astype(float)
print("::::::::",train_data_obtain)
popularity = model(tc.SFrame(train_data_obtain), name, user_id, item_id, target, users_to_recommend, n_rec, n_display)
recom = popularity.recommend(users=['100-C30D01'], k=n_rec)
recom.print_rows(n_display)
print("========================================================================")
# print(popularity)