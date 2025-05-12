import torch
import torch.nn as nn
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.utils.annotations import override

from .grid import PlaceProcessor
from .simple_conv import SimpleConv


class SimpleMazeModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        self.hiddenSize = self.model_config.get("hiddenSize", 16)
        self.numLayers = self.model_config.get("numLayers", 4)
        self.inputSize = self.model_config.get("inputSize", 13)
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
        self.policy_branch = nn.Linear(self.linearHiddenSize, self.action_space.n)
        self.value_branch = nn.Linear(self.linearHiddenSize, 1)

    def _forward_intermediate(self, batch):
        mapInput = batch[Columns.OBS]["vision"].permute(0, 3, 1, 2).to(torch.float32)
        mapOutput = self.primaryConvModule(mapInput)
        return self.prePredictionHead(mapOutput)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        policy = self.policy_branch(self._forward_intermediate(batch))
        return {Columns.ACTION_DIST_INPUTS: policy}

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        return self.value_branch(self._forward_intermediate(batch))


class MemoryMazeModule(SimpleMazeModule):
    def setup(self):
        SimpleMazeModule.setup(self)
        self.trajectoryMemory = nn.GRU(
            self.linearHiddenSize, self.linearHiddenSize, batch_first=True
        )

    def _forward(self, input_dict, state, seq_lens):
        memory = input_dict["obs"]["memory"]
        memoryShape = memory.shape
        memory = memory.reshape(-1, *memoryShape[2:])
        memory = memory.permute(0, 3, 1, 2).to(torch.float32)
        memoryFeatures = self.primaryConvModule(memory)
        memoryFeatures = self.prePredictionHead(memoryFeatures)
        memoryFeatures = memoryFeatures.reshape(*memoryShape[:2], self.linearHiddenSize)
        _, currentStateFeatures = self.trajectoryMemory(memoryFeatures)
        currentStateFeatures = currentStateFeatures.squeeze(0)
        policy = self.policy_branch(currentStateFeatures)
        self._value_out = self.value_branch(currentStateFeatures).squeeze(1)
        return {Columns.ACTION_DIST_INPUTS: policy}

    def compute_values(self, batch):
        assert self._value_out is not None, (
            "forward() must be called before value_function()"
        )
        return self._value_out


class PlaceMazeModule(SimpleMazeModule):
    def setup(self):
        SimpleMazeModule.setup(self)
        self.placer = PlaceProcessor(
            self.observation_space.shape[0] - (self.inputSize**2) * 3,
            self.hiddenSize,
            self.numLayers,
        )
        self.place_policy_branch = nn.Linear(self.hiddenSize, self.action_space.n)
        self.place_value_branch = nn.Linear(self.hiddenSize, 1)

    def _forward(self, input_dict, state, seq_lens):
        mapInput = input_dict["obs"]["map"].permute(0, 3, 1, 2).to(torch.float32)
        mapOutput = self.resnet(mapInput)
        placeInput = input_dict["obs"]["place"].to(torch.float32)
        placeOutput = self.placer(placeInput)
        policy = self.policy_branch(mapOutput) + self.place_policy_branch(placeOutput)
        self._value_out = self.value_branch(mapOutput).squeeze(
            1
        ) + self.place_value_branch(placeOutput).squeeze(1)

        return {Columns.ACTION_DIST_INPUTS: policy}
