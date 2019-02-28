# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import time
import turicreate as tc
from sklearn.model_selection import train_test_split

import sys
# sys.path.append("..")
retail_data = pd.read_csv("dataset/online.csv",encoding = "ISO-8859-1")
train_data_obtain = None
test_data_obtain = None

def create_data_dummy(data):
    data_dummy = tc.SFrame(data.copy())
    data_dummy['purchase_dummy'] = 1
    return data_dummy


def normalize_data(data):
    df_matrix = pd.pivot_table(data, values='Quantity', index='CustomerID', columns='StockCode')
    df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())
    d = df_matrix_norm.reset_index()
    d.index.names = ['scaled_purchase_freq']
    return pd.melt(d, id_vars=['CustomerID'], value_name='scaled_purchase_freq').dropna()


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
        train_data['CustomerID'] = train_data['CustomerID'].astype(str)
        print("type => ",train_data)

        if name == 'popularity':
            model = tc.popularity_recommender.create(train_data,user_id=user_id, item_id=item_id, target=target)
            print("model ==> ",model)
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
        print("==> ",str(e))
nor_retail_data = normalize_data(retail_data)
dumy_retail_data = create_data_dummy(nor_retail_data)
user_id = 'CustomerID'
item_id = 'StockCode'
n_rec = 10 # number of items to recommend
n_display = 30 # to display the first few rows in an output dataset
train_data_obtain,test_data_obtain = split_data(dumy_retail_data.to_dataframe())
users_to_recommend = list(test_data_obtain[user_id])
name = 'popularity'
target = 'scaled_purchase_freq'
popularity = model(tc.SFrame(train_data_obtain), name, user_id, item_id, target, users_to_recommend, n_rec, n_display)
recom = popularity.recommend(users=['54323'], k=n_rec)
recom.print_rows(n_display)
print("================================")
# print(popularity)
