import os
import sys
import numpy
import torch
import pandas as pd
from torch.utils.data import DataLoader
from train_baseline import BPbaseline
from read_dataset import ReadDataset
from pandas.core.frame import DataFrame

if __name__ == '__main__':
    batch_size = 512

    network = BPbaseline(input_dim=6)
    if os.path.exists("baseline.pkl"):
        network.load_state_dict(torch.load("baseline.pkl",map_location='cpu'))
        print("[+] Load model Sucessfully")
    else:
        print("[-] Load model failed")
        sys.exit()
    network.eval()

    # 导入测试数据
    test_data = ReadDataset(csv_path = 'test/ad_info.csv', is_train_data = False)
    test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)
    print("[+] test begin ........................")
    
    user_age = {}
    user_gender = {}
    for data in test_loader:
        age,gender = network(data[0].float())
        user_ids = data[1].numpy() 
        ages = (torch.argmax(age,dim=1) + 1).detach().numpy() 
        genders = (torch.argmax(gender,dim=1)).detach().numpy()
        # print(user_ids.shape, ages.shape, genders.shape) # (512,)
        for i in range(0, user_ids.shape[0]):
            if user_ids[i] in user_age.keys():
                user_age[user_ids[i]].append(ages[i])
            else:
                user_age[user_ids[i]] = [ages[i]]

            if user_ids[i] in user_gender.keys():
                user_gender[user_ids[i]].append(genders[i])
            else:
                user_gender[user_ids[i]] = [genders[i]]
            # print(user_gender[user_ids[i]], ',')

    new_user_age = {}
    for user_id, ages in user_age.items():
        if len(ages) == 0:
            continue
        age_set = set(ages)
        age = ages[0]
        for item in age_set:
            if ages.count(item) > ages.count(age): # 投票
                age = item

        new_user_age[user_id] = age
    # print(new_user_age)

    new_user_gender = {}
    for user_id, genders in user_gender.items():
        if len(genders) == 0:
            continue

        gender_set = set(genders)
        gender = genders[0]
        for item in age_set:
            if genders.count(item) > genders.count(gender): # 投票
                gender = item

        new_user_gender[user_id] = gender
    # print(new_user_age)


    user_age = {'user_id': list(new_user_age.keys()),
                'predicted_age': list(new_user_age.values())}
    user_age_csv = DataFrame(user_age)
    # print(user_age_csv.info())
    # print(user_age_csv.head(2))

    user_gender = {'user_id': list(new_user_gender.keys()),
                'predicted_gender': list(new_user_gender.values())}
    user_gender_csv = DataFrame(user_gender)
    # print(user_gender_csv.info())
    # print(user_gender_csv.head(2))
    user_info_csv = pd.merge(user_age_csv, user_gender_csv, on = ('user_id'), how = 'left')
    print(user_info_csv.info())
    print(user_info_csv.head(2))
    user_info_csv.to_csv('test/submission.csv', index = None)  


        #扩充维度
        # data[1] = data[1].unsqueeze(1)
        # age = age.unsqueeze(1)
        # gender = gender.unsqueeze(1)
        # print(data[1].shape, age.shape, gender.shape)

        # item = torch.cat((data[1], age, gender), dim = 1)
        # print(item.shape)
        # user_info.append(item)

    # user_info = torch.cat(user_info, dim = 0)
    



    