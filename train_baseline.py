import torch
import torch.nn as nn

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BPbaseline(nn.Module):
    def __init__(self,input_dim = 7):
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
    network = BPbaseline()
    rand_input = (torch.rand(1,7),torch.rand(1,10),torch.rand(1,2))
    adam = torch.optim.Adam(network.parameters(), lr=0.00001)
    loss = network(rand_input[0],rand_input[1],rand_input[2])
    print('loss:{:.4f}'.format(loss))
