# coding:utf-8
import pandas as pd
import numpy as  np
import time
import datetime
import math


train = pd.read_csv('datagrand_0517/train.csv')
candidate = pd.read_csv('datagrand_0517/candidate.txt')
news_info = pd.read_csv('datagrand_0517/news_info.csv')
all_news_info = pd.read_csv('datagrand_0517/all_news_info.csv')


def test_txtTocsv(test):
    test_csv = pd.DataFrame()
    n = 0
    k = 0
    for user_id, group in test.groupby(['user_id'], as_index=False, sort=False):
        item_id = group['item_id'][k].split()
        for i in item_id:
            test_csv.loc[n, 0] = user_id
            test_csv.loc[n, 1] = i
            n += 1
        k += 1

    test_csv.columns = ['user_id', 'item_id']
    test_csv.sort_index(axis=1)
    return test_csv
test = pd.read_csv('datagrand_0517/test.txt', header=None)
test.columns = ['user_id', 'item_id']
test = test_txtTocsv(test)
test['user_id'] = test['user_id'].astype(str)   
test['item_id'] = test['item_id'].astype(int)
test.to_csv('datagrand_0517/test0729.csv',index=False)
print (u'test转化完成')


def timestamp_transfer(x):
    x = time.localtime(x)
    x = time.strftime("%Y-%m-%d %H:%M:%S", x)
    return x


train['date'] = train['action_time'].apply(timestamp_transfer)
all_news_info['date'] = all_news_info['timestamp'].apply(timestamp_transfer)
news_info['date'] = news_info['timestamp'].apply(timestamp_transfer)




def action_type_transfer(x):
    if x == 'view':
        return 0.8
    elif x == 'deep_view':
        return 1
    elif x == 'comment':
        return 1.2
    elif x == 'collect':
        return 1.2
    elif x == 'share':
        return 1.5
    else:
        return 1


train['actiontype_weight'] = train['action_type'].apply(action_type_transfer)

train['date'] = pd.to_datetime(train['date']).dt.date
all_news_info['date'] = pd.to_datetime(all_news_info['date']).dt.date
news_info['date'] = pd.to_datetime(news_info['date']).dt.date
news_info['sub_days'] = (datetime.date(2017, 2, 19) - news_info['date']).dt.days
all_news_info['sub_days'] = (datetime.date(2017, 2, 19) - all_news_info['date']).dt.days
train['sub_days'] = (datetime.date(2017, 2, 19) - train['date']).dt.days


def user_cate_like_rate_feat(train, test):
    b = 0.15
    train['count'] = 1.0 / (1 + b * train['sub_days']) * train['actiontype_weight']
    user_cate = train.drop_duplicates(['user_id', 'cate_id'])[['user_id', 'cate_id']]
    for i, date in enumerate(train['date'].unique()):
        print(i, date)
        sub_train = train[train['date'] == date]
        user_cate_count = sub_train.groupby(['user_id', 'cate_id'], as_index=False).sum()[['user_id', 'cate_id', 'count']]
        user_cate_count.columns = ['user_id', 'cate_id', 'user_cate_count']
        user_count = sub_train.groupby(['user_id'], as_index=False).sum()[['user_id', 'count']]
        user_count.columns = ['user_id', 'user_count']
        cate_count = sub_train.groupby(['cate_id'], as_index=False).sum()[['cate_id', 'count']]
        cate_count.columns = ['cate_id', 'cate_count']
        cate_count['cate_clicked_rate'] = cate_count['cate_count'] / cate_count['cate_count'].sum()
        user_cate_count = pd.merge(user_cate_count, user_count, on='user_id', how='left')
        user_cate_count = pd.merge(user_cate_count, cate_count, on='cate_id', how='left')
        user_cate_count['user_cate_click_rate'] = user_cate_count['user_cate_count'] / user_cate_count['user_count']
        user_cate_count['user_cate_prefer'] = user_cate_count['user_cate_click_rate'] / user_cate_count['cate_clicked_rate']
        user_cate_count = user_cate_count[['user_id', 'cate_id', 'user_cate_prefer']]
        cate_count = cate_count[['cate_id', 'cate_count']]
        user_cate_count.columns = ['user_id', 'cate_id', 'user_cate_prefer_' + str(i)]
        cate_count.columns = ['cate_id', 'cate_count_' + str(i)]
        user_count.columns = ['user_id', 'user_count_' + str(i)]
        user_cate = pd.merge(user_cate, user_count, on='user_id', how='left')
        user_cate = pd.merge(user_cate, cate_count, on='cate_id', how='left')
        user_cate = pd.merge(user_cate, user_cate_count, on=['user_id', 'cate_id'], how='left')

        user_cate = user_cate.fillna(0)

    user_cate_prefer = 10
    user_cate_sum = 10
    for i in range(3):
        user_cate_prefer += user_cate['user_count_' + str(i)] * user_cate['user_cate_prefer_' + str(i)]
        user_cate_sum += user_cate['user_count_' + str(i)]
    user_cate['user_cate_prefer'] = user_cate_prefer / user_cate_sum

    test_all_news_info = pd.merge(test, all_news_info, on='item_id', how='inner')
    test_all_news_info['count'] = 1

    test_item_count = test_all_news_info.groupby('item_id', as_index=False).count()[['item_id', 'count']]
    test_item_count = pd.merge(test_item_count, all_news_info, on='item_id', how='inner')
    today_cate = test_item_count.groupby('cate_id', as_index=False).sum()[['cate_id', 'count']]
    today_cate['cate_clicked_rate'] = today_cate['count'] / today_cate['count'].sum()
    user_cate = pd.merge(user_cate, today_cate, on='cate_id', how='left')
    user_cate.fillna(0, inplace=True)
    user_cate['user_cate_prefer'] = user_cate['user_cate_prefer'] * user_cate['cate_clicked_rate']

    print (u'用户品类偏好计算完成')
    return user_cate


def test_popularity_feat(test):
    test_all_news_info = pd.merge(test, all_news_info, on='item_id', how='inner')
    test_all_news_info['count'] = 1

    test_item_count = test_all_news_info.groupby('item_id', as_index=False).count()[['item_id', 'count']]
    test_item_count = pd.merge(test_item_count, all_news_info, on='item_id', how='inner')
    a = 0.4
    test_item_count['count'] = test_item_count['count'] / (1 + a * test_item_count['sub_days'])
    test_item_count.sort_values(by='count', ascending=False, inplace=True)

    test_item_cate_count = test_item_count[['item_id', 'cate_id', 'count']].groupby(['item_id', 'cate_id'],
                                                                                    as_index=False).sum()
    return test_item_cate_count


def make_test_set(train, test):
    test_all_news_info = pd.merge(test, all_news_info, on='item_id', how='inner')
    test_all_news_info['count'] = 1

    test_item_count = test_all_news_info.groupby('item_id', as_index=False).count()[['item_id', 'count']]
    test_item_count = pd.merge(test_item_count, all_news_info, on='item_id', how='inner')
    a = 0.4
    test_item_count['count'] = test_item_count['count'] / (1 + a * test_item_count['sub_days'])
    test_item_count.sort_values(by='count', ascending=False, inplace=True)
    test_item_count = test_item_count[test_item_count['item_id'].isin(news_info['item_id'].unique())]
    cate_set = test_item_count['cate_id'].unique()
    candidate_item = pd.DataFrame(columns=test_item_count.columns)
    for cate in cate_set:
        candidate_item = pd.concat([candidate_item, test_item_count[test_item_count['cate_id'] == cate].head(50)])

    user_set = train['user_id'].unique()
    item_set = candidate_item['item_id'].unique()

    import itertools
    user_item_set = []
    for x in itertools.product(user_set, item_set):
        user_item_set.append(x)
    user_item_set = pd.DataFrame(user_item_set, columns=['user_id', 'item_id'])
    test_item_cate_count = test_popularity_feat(test)
    user_item_set = pd.merge(user_item_set, test_item_cate_count, on='item_id', how='left')

    user_cate = user_cate_like_rate_feat(train, test)
    user_cate = user_cate[['user_id', 'cate_id', 'user_cate_prefer']].groupby(['user_id', 'cate_id'], as_index=False).sum()
    print(user_cate)
    user_item_set = pd.merge(user_item_set, user_cate, on=['user_id', 'cate_id'], how='left')
    user_item_set['item_id_prefer'] = user_item_set['count'] * user_item_set['user_cate_prefer']


    print(u'测试集构建完成') 

    return user_item_set


user_item_set = make_test_set(train, test)

user_item_set = user_item_set.sort_values(['item_id_prefer'], ascending=False)
user_item_set['item_id'] = user_item_set['item_id'].astype(int)
user_item_set['user_id'] = user_item_set['user_id'].astype(str)



user_item_seen = {}
for row in train.values:
    user_id = row[0]
    item_id = row[1]
    if user_item_seen.__contains__(user_id) == False:
        user_item_seen[user_id] = {}
    user_item_seen[user_id][item_id] = 1

user_list = []
recommend_result_items_list = []
user_item = user_item_set.groupby('user_id')
for user, item in user_item:
    user_list.append(user)
    items = list(item['item_id'].values)
    item_result = []
    for i in items:
        if len(item_result) == 5:
            break
        if user_item_seen[user].__contains__(i) == False:
            item_result.append(i)
    item_result = " ".join(map(str, item_result)) 
    recommend_result_items_list.append(item_result)

recommend_result = pd.DataFrame()
recommend_result['user_id'] = user_list
recommend_result['item_id'] = recommend_result_items_list  
recommend_result = recommend_result[recommend_result.item_id != str(0)]   
recommend_result = recommend_result[recommend_result.item_id != str(0)]   


users = pd.read_csv('datagrand_0517/candidate.txt', header=None)
users.columns = ['user_id']
test = pd.read_csv('datagrand_0517/test0729.csv')
item_topHot = test.groupby(['item_id'], as_index=False).count().sort_values(['user_id'], ascending=False).head(5)['item_id'].values  # 热度排名前5的item_id
recommend_test = users
recommend_test['recommend_test_item'] = [" ".join(map(str, list(item_topHot)))] * len(recommend_test)
recommend_test = pd.merge(recommend_test, recommend_result, how='left', on='user_id', ).fillna(0)
recommend_test = recommend_test[recommend_test.item_id == 0][['user_id', 'recommend_test_item']]
recommend_test.columns = ['user_id', 'item_id']
recommend_result = recommend_result.append(recommend_test)

recommend_result = recommend_result.drop_duplicates('user_id')
recommend_result.to_csv('sub/sub0723_1.txt', index=None, header=None)





