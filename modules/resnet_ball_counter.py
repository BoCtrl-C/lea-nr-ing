import torch
from torch.nn import functional as F, Conv2d
from torchvision.models.resnet import BasicBlock, ResNet


def resnet18_ball_counter(max_num):
    """The function returns a ResNet-based classifier capable of counting up to
    max_num balls.
    """

    return ResNetBallCounter(BasicBlock, [2, 2, 2, 2], num_classes=max_num + 1)


class ResNetBallCounter(ResNet):
    """The class implements a convolutional ball counter based on ResNet.
    """

    def __init__(self, *args, **kwargs):
        super(ResNetBallCounter, self).__init__(*args, **kwargs)
        
        self.conv1 = Conv2d(
            1, # grayscale
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
    
    def count(self, batch):
        """The method counts how many balls are present in each frame of the
        provided batch.
        """
        
        self.eval()

        with torch.no_grad():
            logits = self(batch.unsqueeze(1))
        probabilities = F.softmax(logits, dim=- 1)
        predictions = torch.argmax(probabilities, dim=- 1)

        return predictions