def build_model():
    import torchvision.models as models
    import torch.nn as nn

    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5) 
    return model

