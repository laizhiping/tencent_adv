import os
import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def deal_train_csv(ad_csv = 'train_preliminary/ad.csv', click_log_csv = 'train_preliminary/click_log.csv', 
                   user_csv = 'train_preliminary/user.csv', new_csv = 'train_preliminary/user_ad_info.csv'):
    if os.path.exists(new_csv):
        return
    ad = pd.read_csv(ad_csv, na_values = '\\N')
    click_log = pd.read_csv(click_log_csv, na_values = '\\N')
    user = pd.read_csv(user_csv, na_values = '\\N')
    print(ad.info())
    print(click_log.info())
    print(user.info())
    # print(ad.head(2))
    # print(click_log.head(2))
    # print(user.head(2))
    # print(ad.isnull().sum())
    # print(click_log.isnull().sum())
    # print(user.isnull().sum())
    #ad表的product_id缺失929524，industry缺失101048，暂时先用-1填充； click_log表无缺失
    ad.fillna(-1)
    user_ad_info = pd.merge(ad, click_log, on = ('creative_id'), how = 'left') #连接ad和click_log两张表
    user_ad_info = pd.merge(user_ad_info, user, on = ('user_id'), how = 'left') #连接ad、click_log和user三张表
    print('the new scv user_ad_info.scv after merge ad.scv, click_log.csv and user.csv')
    print(user_ad_info.info()) 
    user_ad_info.to_csv(new_csv, index = None)
    print('write user_ad_info.csv finished!')

def deal_test_csv(ad_csv = 'test/ad.csv', click_log_csv = 'test/click_log.csv', 
                  new_csv = 'test/ad_info.csv'):
    if os.path.exists(new_csv):
        return
    ad = pd.read_csv(ad_csv, na_values = '\\N')
    click_log = pd.read_csv(click_log_csv, na_values = '\\N')
    # print(ad.head(2))
    # print(click_log.head(2))
    # print(ad.isnull().sum())
    # print(click_log.isnull().sum())
    #ad表的product_id缺失929524，industry缺失101048，暂时先用-1填充； click_log表无缺失
    ad.fillna(-1)
    ad_info = pd.merge(ad, click_log, on = ('creative_id'), how = 'left') #连接ad和click_log两张表
    print('the new scv ad_info.scv after merge ad.scv and click_log.csv')
    print(ad_info.info())
    #新表的product_id缺失12593112，industry缺失1276874
    ad_info.to_csv(new_csv, index = None)
    print('write ad_info.csv finished!')

#将数据按某列取值分表
def split_by_age(csv_path = 'train_preliminary/user_ad_info.csv', column = 'age'):
    path = re.match('.*/', csv_path).group()
    # print(path.group())
    df = pd.read_csv(csv_path)
    values = df[column].unique()
    for value in values:
        new_df = df[df[column] == value]
        new_csv_name = path + column + '_' + str(value) + '.csv' 
        # print(new_csv_name)
        new_df.to_csv(new_csv_name, index = None)


    #数字过大会溢出
    # def trans2binary(n):
    #     if n < 2:
    #         return n
    #     else:
    #         return trans2binary(n // 2) + n % 2

def trans2binary(n):
    result = []
    while True:
        n, remainder = divmod(n, 2)
        result.append(remainder)
        if n == 0:
            return result[::-1]

class ReadDataset(Dataset):
    def __init__(self, csv_path = 'train_preliminary/user_ad_info.csv', is_train_data = True): 
        self.__user_ad_info =  pd.read_csv(csv_path)
        self.__is_train_data = is_train_data
        # print(self.__user_ad_info.info())

        # 把user_id 放在最后一列
        cols = list(self.__user_ad_info)
        cols.append(cols.pop(cols.index('user_id')))
        self.__user_ad_info = self.__user_ad_info.loc[:, cols]
        # print(self.__user_ad_info.info())
               
        # user_id = self.__user_ad_info['user_id']
        # self.__user_ad_info = self.__user_ad_info.drop('user_id', axis = 1) 
        # print(self.__user_ad_info.info())
        # self.__user_ad_info = self.__user_ad_info.insert(self.__user_ad_info.shape[1], 'user_id', user_id)
        # print(self.__user_ad_info.info())        

        self.__user_ad_info = self.__user_ad_info.drop('product_id', axis = 1) # 去掉有缺失值的列product_id
        self.__user_ad_info = self.__user_ad_info.drop('industry', axis = 1) # 去掉有缺失值的列industry

        # print('the infos in ReadDataset')
        # print(self.__user_ad_info.info())
        # print(self.__user_ad_info.max())


        self.__user_ad_info = torch.from_numpy(np.array(self.__user_ad_info)) #数据格式转成torch tensor
        self.__length = self.__user_ad_info.shape[0]
        
        self.__len_creative_id = len(trans2binary(4445720))
        self.__len_ad_id = len(trans2binary(3812202))
        self.__len_product_category= len(trans2binary(18))
        self.__len_advertiser_id = len(trans2binary(62965))
        self.__len_time = len(trans2binary(91))
        self.__len_click_times = len(trans2binary(185))


    def __getitem__(self, index):
        item = self.__user_ad_info[index]
        if self.__is_train_data: # 训练数据，0~n-3, n-2, n-1: feature, age, gender, user_id
            n_feature = item.shape[0] - 3
            x = item[ :n_feature].numpy()

            creative_id = trans2binary(x[0])
            ad_id = trans2binary(x[1])
            product_category = trans2binary(x[2])
            advertiser_id = trans2binary(x[3])
            time = trans2binary(x[4])
            click_times = trans2binary(x[5])

            creative_id_feature = [0 for i in range(0, self.__len_creative_id - len(creative_id))]
            creative_id_feature.extend(creative_id)
            ad_id_feature = [0 for i in range(0, self.__len_ad_id - len(ad_id))]
            ad_id_feature.extend(ad_id)
            product_category_feature = [0 for i in range(0, self.__len_product_category - len(product_category))]
            product_category_feature.extend(product_category)
            advertiser_id_feature = [0 for i in range(0, self.__len_advertiser_id - len(advertiser_id))]
            advertiser_id_feature.extend(advertiser_id)                        
            time_feature = [0 for i in range(0, self.__len_time - len(time))]
            time_feature.extend(time)
            click_times_feature = [0 for i in range(0, self.__len_click_times - len(click_times))]
            click_times_feature.extend(click_times)
            x = creative_id_feature + ad_id_feature + product_category_feature + \
                      advertiser_id_feature + time_feature
            # print(x)
            x = torch.from_numpy(np.array(x))

            

            age = item[n_feature]
            gender = item[n_feature+1]
            #构造one_hot
            train_age = torch.zeros(10) 
            train_gender = torch.zeros(2)
            train_age[age-1] = 1
            train_gender[gender-1] = 1
            return x, train_age, train_gender

        else: # 测试数据，0~n-2, n-1: feature, user_id
            n_feature = item.shape[0] - 1
            test_x = item[ :n_feature]
            x = item[ :n_feature].numpy()

            creative_id = trans2binary(x[0])
            ad_id = trans2binary(x[1])
            product_category = trans2binary(x[2])
            advertiser_id = trans2binary(x[3])
            time = trans2binary(x[4])
            click_times = trans2binary(x[5])

            creative_id_feature = [0 for i in range(0, self.__len_creative_id - len(creative_id))]
            creative_id_feature.extend(creative_id)
            ad_id_feature = [0 for i in range(0, self.__len_ad_id - len(ad_id))]
            ad_id_feature.extend(ad_id)
            product_category_feature = [0 for i in range(0, self.__len_product_category - len(product_category))]
            product_category_feature.extend(product_category)
            advertiser_id_feature = [0 for i in range(0, self.__len_advertiser_id - len(advertiser_id))]
            advertiser_id_feature.extend(advertiser_id)                        
            time_feature = [0 for i in range(0, self.__len_time - len(time))]
            time_feature.extend(time)
            click_times_feature = [0 for i in range(0, self.__len_click_times - len(click_times))]
            click_times_feature.extend(click_times)
            x = creative_id_feature + ad_id_feature + product_category_feature + \
                      advertiser_id_feature + time_feature
            x = torch.from_numpy(np.array(x))

            user_id = item[n_feature]
            return x, user_id

    def __len__(self):
        return self.__length



if __name__ == '__main__':
    
    # deal_train_csv()
    # split_by_age(csv_path = 'train_preliminary/user_ad_info.csv', column = 'age')

    is_train_data = False

    if is_train_data: # train data
        if not os.path.exists('train_preliminary/user_ad_info.csv'):
            deal_train_csv(ad_csv = 'train_preliminary/ad.csv', click_log_csv = 'train_preliminary/click_log.csv', 
                           user_csv = 'train_preliminary/user.csv', new_csv = 'train_preliminary/user_ad_info.csv')
        batch_size = 1
        #读入新表，返回新表数据(torch tensor格式)
        train_data = ReadDataset(csv_path = 'train_preliminary/user_ad_info.csv', is_train_data = is_train_data)
        # print(len(train_data))
        train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = False)
        # print(train_loader.size())
        for batch, item in enumerate(train_loader):
            train_x , age, gender = item[0], item[1], item[2]
            print(train_x , age, gender)
     #       tensor([[  1,   1,   5, 381,  81,   1]]) tensor([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]]) tensor([[1., 0.]])
            break

    else: # test data
        if not os.path.exists('test/ad_info.csv'):
            deal_test_csv(ad_csv = 'test/ad.csv', click_log_csv = 'test/click_log.csv', 
                          new_csv = 'test/ad_info.csv')

        batch_size = 1
        #读入新表，返回新表数据(torch tensor格式)
        test_data = ReadDataset(csv_path = 'test/ad_info.csv', is_train_data = is_train_data)
        # print(len(test_data))
        test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)
        # print(train_loader.size())
        for batch, item in enumerate(test_loader):
            test_x , user_id = item[0], item[1]
            print(test_x , user_id)
            # tensor([[  1,   1,   5, 381,  81,   1]]) tensor([3153317])
            break


