import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

from .grid import PlaceProcessor
from .simple_conv import SimpleConv


class SimpleMazeNet(TorchModelV2, nn.Module):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.hiddenSize = kwargs.get("hiddenSize", 16)
        self.numLayers = kwargs.get("numLayers", 4)
        self.inputSize = kwargs.get("inputSize", 13)
        self.linearHiddenSize = self.hiddenSize * 8
        self.primaryConvModule = SimpleConv(self.hiddenSize)
        primaryConvModuleOutSize = ((self.inputSize + 1) // 2 + 1) // 2
        self.prePredictionHead = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                primaryConvModuleOutSize**2 * self.hiddenSize * 2, self.linearHiddenSize
            ),
            nn.ReLU(),
        )
        self.policy_branch = nn.Linear(self.linearHiddenSize, num_outputs)
        self.value_branch = nn.Linear(self.linearHiddenSize, 1)

    def forward(self, input_dict, state, seq_lens):
        mapInput = input_dict["obs"]["vision"].permute(0, 3, 1, 2).to(torch.float32)
        mapOutput = self.primaryConvModule(mapInput)
        mapOutput = self.prePredictionHead(mapOutput)
        policy = self.policy_branch(mapOutput)
        self._value_out = self.value_branch(mapOutput).squeeze(1)
        return policy, []

    def value_function(self):
        assert self._value_out is not None, (
            "forward() must be called before value_function()"
        )
        return self._value_out


class MemoryMazeNet(SimpleMazeNet):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        SimpleMazeNet.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        self.trajectoryMemory = nn.LSTM(
            self.linearHiddenSize, self.linearHiddenSize, 1, batch_first=True
        )

    def forward(self, input_dict, state, seq_lens):
        memory = input_dict["obs"]["memory"]
        memoryShape = memory.shape
        memory = memory.reshape(-1, *memoryShape[2:])
        memory = memory.permute(0, 3, 1, 2).to(torch.float32)
        memoryFeatures = self.primaryConvModule(memory)
        memoryFeatures = self.prePredictionHead(memoryFeatures)
        memoryFeatures = memoryFeatures.reshape(*memoryShape[:2], self.linearHiddenSize)
        _, (currentStateFeatures, __) = self.trajectoryMemory(memoryFeatures)
        currentStateFeatures = currentStateFeatures.squeeze(0)
        policy = self.policy_branch(currentStateFeatures)
        self._value_out = self.value_branch(currentStateFeatures).squeeze(1)
        return policy, []

    def value_function(self):
        assert self._value_out is not None, (
            "forward() must be called before value_function()"
        )
        return self._value_out


class PlaceMazeNet(SimpleMazeNet):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        SimpleMazeNet.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        self.placer = PlaceProcessor(
            obs_space.shape[0] - (self.mazeSize**2) * 3, self.hiddenSize, self.numLayers
        )
        self.place_policy_branch = nn.Linear(self.hiddenSize, num_outputs)
        self.place_value_branch = nn.Linear(self.hiddenSize, 1)

    def forward(self, input_dict, state, seq_lens):
        mapInput = input_dict["obs"]["map"].permute(0, 3, 1, 2).to(torch.float32)
        mapOutput = self.resnet(mapInput)
        placeInput = input_dict["obs"]["place"].to(torch.float32)
        placeOutput = self.placer(placeInput)
        policy = self.policy_branch(mapOutput) + self.place_policy_branch(placeOutput)
        self._value_out = self.value_branch(mapOutput).squeeze(
            1
        ) + self.place_value_branch(placeOutput).squeeze(1)

        return policy, []


ModelCatalog.register_custom_model("simple_maze_net", SimpleMazeNet)
ModelCatalog.register_custom_model("place_maze_net", PlaceMazeNet)
ModelCatalog.register_custom_model("memory_maze_net", MemoryMazeNet)
