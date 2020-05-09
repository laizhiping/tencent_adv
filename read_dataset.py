import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def deal_train_csv(ad_csv = 'train_preliminary/ad.csv', click_log_csv = 'train_preliminary/click_log.csv', 
                   user_csv = 'train_preliminary/user.csv', new_csv = 'train_preliminary/user_ad_info.csv'):
    ad = pd.read_csv(ad_csv, na_values = '\\N')
    click_log = pd.read_csv(click_log_csv, na_values = '\\N')
    user = pd.read_csv(user_csv, na_values = '\\N')
    # print(ad.head(2))
    # print(click_log.head(2))
    # print(user.head(2))
    # print(ad.isnull().sum())
    # print(click_log.isnull().sum())
    # print(user.isnull().sum())
    #ad表的product_id缺失929524，industry缺失101048，暂时先用-1填充
    ad.fillna(-1)
    user_ad_info = pd.merge(ad, click_log, on = ('creative_id'), how = 'left') #连接ad和click_log两张表
    user_ad_info = pd.merge(user_ad_info, user, on = ('user_id'), how = 'left') #连接ad、click_log和user三张表
    user_ad_info.to_csv(new_csv, index = None)


class ReadDataset(Dataset):
    def __init__(self, csv_path = 'train_preliminary/user_ad_info.csv'): 
        self.__user_ad_info =  pd.read_csv(csv_path)
        # print(user_ad_info.info())
        self.__user_ad_info = self.__user_ad_info.drop('user_id', axis = 1) # 去掉连接键user_id
        self.__user_ad_info = self.__user_ad_info.drop('product_id', axis = 1) # 去掉有缺失值的列product_id
        self.__user_ad_info = self.__user_ad_info.drop('industry', axis = 1) # 去掉有缺失值的列industry
        self.__user_ad_info = torch.from_numpy(np.array(self.__user_ad_info)) #数据格式转成torch tensor
        self.__length = self.__user_ad_info.shape[0]
    def __getitem__(self, index):
        item = self.__user_ad_info[index]
        n_feature = item.shape[0] - 2
        train_x = item[ :n_feature]
        age = item[n_feature]
        gender = item[n_feature+1]

        #构造one_hot
        train_age = torch.zeros(10) 
        train_gender = torch.zeros(2)
        train_age[age-1] = 1
        train_gender[gender-1] = 1
        return train_x, train_age, train_gender
    def __len__(self):
        return self.__length



if __name__ == '__main__':
    #处理三张表
    deal_train_csv(ad_csv = 'train_preliminary/ad.csv', click_log_csv = 'train_preliminary/click_log.csv', 
                   user_csv = 'train_preliminary/user.csv', new_csv = 'train_preliminary/user_ad_info.csv')

    batch_size = 1
    #读入新表，返回新表数据(torch tensor格式)
    train_data = ReadDataset('train_preliminary/user_ad_info.csv')
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = False)
    # print(train_loader.size())
    for batch, item in enumerate(train_loader):
        train_x , age, gender = item[0], item[1], item[2]
        print(train_x , age, gender)
 #       tensor([[  1,   1,   5, 381,  81,   1]]) tensor([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]]) tensor([[1., 0.]])
        break
