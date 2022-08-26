import torch.nn as nn


class FeedForwardAutoRegressive(nn.Module):
    def __init__(self, feed_forward, auto_regressive_head):
        super().__init__()
        self.feed_forward = feed_forward
        self.auto_regressive_head = auto_regressive_head

    def forward(self, x):
        feed_forward_output = self.feed_forward(x)
        auto_regressive_output = self.auto_regressive_head(feed_forward_output)
        return auto_regressive_output
