import torch


NUM_LEVELS = 16


def quantize(x):
    """The function uniformly quantizes the bouncing balls dataset.
    """
    
    return ((NUM_LEVELS - 1)*x).long()

def unquantize(x):
    """The function "inverts" the action of quantize().
    """

    return x/(NUM_LEVELS - 1)


class BallCounter:
    """The class implements an area-based heuristic exploited to count balls in
    the input frame. The average ball area is estimated from the provided
    dataset (DataLoader).
    """

    def __init__(self, train_ldr):
        # compute the average ball area
        num_balls = 0
        num_pixels = 0
        for nums, frms in train_ldr:
            num_balls += torch.sum(nums).item()
            frms = quantize(frms)
            num_pixels += torch.sum(frms > 0).item()
        self.area = num_pixels/num_balls

    def count(self, frm):
        return torch.round(torch.sum(frm > 0)/self.area)\
            .int()\
            .item()