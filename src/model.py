import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

HID_DIM = 2048
OUT_DIM = 128

# Resnet backbone
class resnet50_fext(nn.Module):
    def __init__(self, pretarin=True):
        super(resnet50, self).__init__()
        if pretarin:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        embedding = self.encoder(x)
        embedding = embedding.view(embedding.size()[0], -1)
        return (embedding)


# Linear model
class MLP(nn.Module):
    def __init__(self, in_dim, mlp_hid_size, proj_size):
        super(MLP, self).__init__()
        self.head = nn.Sequential(nn.Linear(in_dim, mlp_hid_size),
                                  nn.BatchNorm1d(mlp_hid_size),
                                  nn.ReLU(),
                                  nn.Linear(mlp_hid_size, proj_size))

    def forward(self, x):
        x = self.head(x)
        return (x)

# Byol model
class BYOL(nn.Module):
    def __init__(self, net, backbone, hid_dim, out_dim):
        super(BYOL, self).__init__()
        self.net = net
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.projection = MLP(in_dim=backbone.fc.in_features,
                              mlp_hid_size=hid_dim, proj_size=out_dim)
        self.prediction = MLP(
            in_dim=out_dim, mlp_hid_size=hid_dim, proj_size=out_dim)

    def forward(self, x):
        embedding = self.encoder(x)
        embedding = embedding.view(embedding.size()[0], -1)
        project = self.projection(embedding)

        if self.net == 'target':
            return (project)
        predict = self.prediction(project)
        return (predict)

# SimCLR model
class SimCLR(nn.Module):
    def __init__(self, backbone, hid_dim, out_dim):
        super(SimCLR, self).__init__()
        # we get representations from avg_pooling layer
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.projection = MLP(in_dim=backbone.fc.in_features,
                              mlp_hid_size=hid_dim, proj_size=out_dim)

    def forward(self, x):
        embedding = self.encoder(x)
        embedding = embedding.view(embedding.size()[0], -1)
        project = self.projection(embedding)
        return (project)

# Supervised model that I took from wei-cheng 
class ResNet50Predictor(nn.Module):
    def __init__(self, embed_dim, dropout=0.5):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        in_feats = backbone.fc.in_features
        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_feats, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            ## regression output (prediction  output such as age)
            nn.Linear(embed_dim, 1) 
        )
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.predictor(x)
        return x    

#### Building finetune model for simCLR pre-trained model
class finetune_net(nn.Module):
    def __init__(self,model,num_classes=0):
        # in_dim: dimension of the input feature to the linear layer
        super(finetune_net,self).__init__()
        # model is the backbone that we want to add linear layer on top of that for fine tuning
        self.model = model
        #print('#################################')
        #print(f'model is as follows: {self.model}')
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        #print(f'feature_extactor: {self.feature_extractor}')
        # print feature extractor architecture
        #print(f'feature extractor architecture: {self.feature_extractor}')
        # optional: to stabilize training
        # self.dropout = nn.Dropout(p_drop)
        in_feats = 2048  # For resnet50
        self.linear = nn.Linear(in_feats,num_classes)
        
    def forward(self,x):
        embeding = self.feature_extractor(x)
        #print(f'size of embedding: {embeding.size()}')
        embeding = embeding.view(embeding.size()[0],-1)
        #print(f'size of flatted embeddings:{embeding.size()}')
        logits = self.linear(embeding) 
        return(logits)
    
if __name__=="__main__":
    # Example usage
    HID_DIM = 2048
    OUT_DIM = 128
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    simclr_model = SimCLR(backbone=backbone, hid_dim=HID_DIM, out_dim=OUT_DIM)
    #print(f'simclr_model: {simclr_model}')
    print('#################################')
    #load supervised model as feature extractor
    net = ResNet50Predictor(embed_dim=2048, dropout=0.5)
    #finetune_net = finetune_net(model=simclr_model, num_classes=2)
    finetune_net = finetune_net(model=net, num_classes=2)
    #print(f'finetune_net: {finetune_net}')
    print('#################################')
    dummy_input = torch.randn(8, 3, 224, 224)  # Batch of 8 RGB images of size 224x224
    #simclr_output = simclr_model(dummy_input)
    finetune_output = finetune_net(dummy_input)
    
    #print("SimCLR output shape:", simclr_output.shape)
    print("finetune_net feature extractor output shape:", finetune_output.shape)
    
