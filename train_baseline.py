import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from read_dataset import ReadDataset

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BPbaseline(nn.Module):
    def __init__(self,input_dim = 8):
        super(BPbaseline, self).__init__()
        self.common_layer = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.Hardtanh(),
            nn.Linear(4096,4096),
            nn.Hardtanh(),
            nn.Linear(4096,2048),
            nn.Hardtanh(),
            nn.Linear(2048,1024),
            nn.Hardtanh()
            )
        self.age_layer = nn.Sequential(
            nn.Linear(1024,512),
            nn.Hardtanh(),
            nn.Linear(512,10),
            nn.Softmax(dim=-1)
            )

        self.gender_layer = nn.Sequential(
            nn.Linear(1024,512),
            nn.Hardtanh(),
            nn.Linear(512,2),
            nn.Softmax(dim=-1)
            )
    def forward(self,input, age_input=None, gender_input=None):
        common_output = self.common_layer(input)
        age_output = self.age_layer(common_output)
        gender_output = self.gender_layer(common_output)

        if age_input is None:
            return age_output, gender_output
        else:
            age_loss = torch.sqrt((age_output-age_input)**2)
            gender_loss = torch.sqrt((gender_output-gender_input)**2)
            loss = age_loss.mean()+gender_loss.mean()
            return loss

if __name__ == '__main__':
    batch_size = 512
    learning_rate = 0.00001
    epoch_num = 100
    # 导入数据集
    train_data = ReadDataset('train_preliminary/user_ad_info.csv')
    train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)

    # 构造网络
    network = BPbaseline(input_dim=6)
    if os.path.exists("baseline.pkl"):
        network.load_state_dict(torch.load("baseline.pkl"))
        print("[+] Load model Sucessfully")

    # 设置优化器
    adam = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(adam, step_size=1, gamma=0.98)

    print('train begin .................................')
    network.train()
    for epoch in range(epoch_num):
        scheduler.step()
        epoch_loss = 0
        for batch,data in enumerate(train_loader):

            loss = network(data[0].float(),data[1].float(),data[2].float())
            adam.zero_grad()
            loss.backward()
            adam.step()
            epoch_loss += loss
            print("epoch:{},batch:{},batchloss:{:.4f},".format(epoch,batch,loss))

        torch.save(network.state_dict(),"baseline.pkl")
        print("epoch:{},epoch loss:{:.4f}".format(epoch,epoch_loss))
    print("training finished")