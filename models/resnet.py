import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog


torch.manual_seed(0)


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class RLlibResNet(TorchModelV2, nn.Module):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.hiddenSize = kwargs.get("hiddenSize", 16)
        self.numLayers = kwargs.get("numLayers", 4)
        self.mazeSize = kwargs.get("mazeSize", 13)
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, self.hiddenSize, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hiddenSize),
            nn.ReLU(),
        )
        self.backBone = nn.ModuleList([ResBlock(self.hiddenSize) for i in range(4)])
        self.policy_branch = nn.Sequential(
            nn.Conv2d(self.hiddenSize, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.mazeSize**2 * 4, num_outputs),
        )
        self.value_branch = nn.Linear(num_outputs, 1)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].permute(0, 3, 1, 2).to(torch.float32)
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policy_branch(x)

        return policy, []

    def value_function(self):
        return self.value_branch(self._last_output).squeeze(1)


ModelCatalog.register_custom_model("resnet", RLlibResNet)
