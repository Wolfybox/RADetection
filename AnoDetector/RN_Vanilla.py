import torch
import torch.nn as nn


class RNVanilla(nn.Module):
    """
    The Ranking Regression network.
    """

    def __init__(self, pretrained=False, model_dir='', input_dim=4096):
        super(RNVanilla, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout1 = nn.Dropout(p=0.6)
        self.dropout2 = nn.Dropout(p=0.6)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.__init_weight()
        if pretrained:
            self.__load_pretrained_weights(model_dir=model_dir)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        score = self.sigmoid(x)
        return score

    def get_layer_output(self, x, layer):
        if layer == 0: return x
        x = self.dropout1(self.relu1(self.fc1(x)))
        if layer == 1: return x
        x = self.dropout2(self.fc2(x))
        if layer == 2: return x
        x = self.fc3(x)
        if layer == 3: return x
        return self.sigmoid(x)

    def predict(self, x):
        return self.forward(x)

    def __load_pretrained_weights(self, model_dir):
        """Initialize network with pre-trained weights"""
        p_dict = torch.load(model_dir, map_location=torch.device('cpu'))
        s_dict = self.state_dict()
        for name in s_dict.keys():
            s_dict[name] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)


if __name__ == "__main__":
    device = torch.device('cuda:5')
    pass
