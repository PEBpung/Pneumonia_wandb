import torchvision.models as models
import torch.nn as nn

class PneumoniaNet(nn.Module):
    def __init__(self, pretrained=True):
        super(PneumoniaNet, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        self.num_features = self.backbone.fc.in_features
        self.fc = nn.Linear(in_features=512, out_features=1) 
    
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)        
        x = self.backbone.relu(x)       
        x = self.backbone.maxpool(x)  
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)        
        x = self.backbone.layer3(x)        
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), 512)
        x = self.fc(x)
        
        return x