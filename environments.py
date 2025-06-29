from typing import Optional
from collections import deque
import numpy as np
import gymnasium as gym
import random

from maze import print_maze, generateMaze


class MazeEnv(gym.Env):
    def __init__(self, config=None):
        self._episode_len = 0
        self._mazeArray = config["maze"]
        self._mazeSize = config["mazeSize"]
        self._randomMaze = not self._mazeArray
        self._goalLocation = config["goal"]
        self._fixedGoal = bool(config["goal"])
        self._startLocation = config.get("start", None)
        self._maxSteps = config["maxSteps"]
        self._gateCloseRate = config.get("gateCloseRate", 0)

        self._map = None
        self._agentLocation = (
            np.array(self._startLocation, dtype=np.int32)
            if self._startLocation
            else None
        )
        self.observation_space = gym.spaces.Dict(
            {
                "vision": gym.spaces.MultiBinary((self._mazeSize, self._mazeSize, 3)),
            }
        )
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    def _get_info(self):
        return {"location": self._agentLocation}

    def _getObs(self):
        return {"vision": self._map}

    def getShortestDistance(self):
        size = len(self._mazeArray)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = deque([(1, 1, 0)])

        visited = set()
        visited.add((0, 0))
        maze = self._map[:, :, 0].squeeze().tolist()

        while queue:
            r, c, dist = queue.popleft()
            if (r, c) == (size - 2, size - 2):
                return dist

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < size
                    and 0 <= nc < size
                    and maze[nr][nc] == 1
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))

        return -1

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self._randomMaze:
            self._mazeArray = generateMaze(self._mazeSize)

        mazeSize = len(self._mazeArray)
        if not self._fixedGoal:
            goalLocation = self.np_random.integers(0, mazeSize, size=2, dtype=int)
            while (
                not self._mazeArray[goalLocation[0]][goalLocation[1]]
                or goalLocation[0] == mazeSize // 2
                and goalLocation[1] == mazeSize // 2
            ):
                goalLocation = self.np_random.integers(0, mazeSize, size=2, dtype=int)
            self._goalLocation = goalLocation
        targetChannel = np.zeros([mazeSize, mazeSize, 1], dtype=np.int32)
        targetChannel[self._goalLocation[0], self._goalLocation[1], 0] = 1
        agentChannel = np.zeros([mazeSize, mazeSize, 1], dtype=np.int32)
        if not self._startLocation:
            agentLocation = self.np_random.integers(0, mazeSize, size=2, dtype=int)
            while (
                np.array_equal(agentLocation, self._goalLocation)
                or not self._mazeArray[agentLocation[0]][agentLocation[1]]
            ):
                agentLocation = self.np_random.integers(0, mazeSize, size=2, dtype=int)
            self._agentLocation = agentLocation
            agentChannel[agentLocation[0], agentLocation[1], 0] = 1
        else:
            agentChannel[self._startLocation[0], self._startLocation[1], 0] = 1
            self._agentLocation = np.array(self._startLocation, dtype=np.int32)
        self._pastLocation = self._agentLocation
        mazeChannel = np.expand_dims(self._mazeArray, axis=2)
        probLimit = self._gateCloseRate * 2
        if probLimit < 1:
            actualGateCloseRate = random.random() * probLimit
        else:
            actualGateCloseRate = random.uniform(probLimit - 1, 1)
        gateClosed = np.random.choice(
            [0, 1],
            size=mazeChannel.shape,
            p=[actualGateCloseRate, 1 - actualGateCloseRate],
        )
        mazeChannel = np.where(mazeChannel > 1, gateClosed, mazeChannel)
        self._map = np.concat((mazeChannel, targetChannel, agentChannel), axis=2)
        self._episode_len = 0
        if not self._maxSteps:
            self._maxSteps = self.getShortestDistance()
        return self._getObs(), self._get_info()

    def step(self, action):
        direction = self._action_to_direction[action]
        newLoc = self._agentLocation + direction
        if (
            0 <= newLoc[0] < len(self._mazeArray)
            and 0 <= newLoc[1] < len(self._mazeArray)
            and self._map[newLoc[0], newLoc[1], 0] == 1
        ):
            self._map[self._agentLocation[0], self._agentLocation[1], 2] = 0
            self._agentLocation = newLoc
            self._map[self._agentLocation[0], self._agentLocation[1], 2] = 1

        terminated = np.array_equal(self._agentLocation, self._goalLocation)
        self._episode_len += 1
        truncated = self._episode_len > self._maxSteps
        reward = 1 if terminated else 0
        return self._getObs(), reward, terminated, truncated, self._get_info()

    def render(self):
        mazeClone = [[item for item in row] for row in self._mazeArray]
        mazeClone[self._agentLocation[0]][self._agentLocation[1]] = 3
        print_maze(mazeClone)


class FoggedMazeEnv(MazeEnv):
    def __init__(self, config=None):
        super().__init__(config)
        self._visualRange = config.get("visualRange", 4)
        self._memoryLen = config.get("memoryLen", False)
        visualObsSize = self._visualRange * 2 + 1
        self._memory = None
        self.observation_space = gym.spaces.MultiBinary(
            (visualObsSize, visualObsSize, 3)
        )

    def _getObs(self):
        paddedMap = np.pad(
            self._map,
            (
                (self._visualRange, self._visualRange),
                (self._visualRange, self._visualRange),
                (0, 0),
            ),
            mode="constant",
        )
        _paddedAgentLoc = self._agentLocation + np.array((4, 4))
        vision = paddedMap[
            _paddedAgentLoc[0] - 4 : _paddedAgentLoc[0] + 5,
            _paddedAgentLoc[1] - 4 : _paddedAgentLoc[1] + 5,
            :,
        ]
        """
        The following logic imposes obstructed vision that was more
        accurate when the maze path was 1 unit wide. Now that we
        switch to more of an open field style maze, it is no longer
        applicable.
        """
        # mask = np.zeros((9, 9, 3), dtype=bool)
        # mask[4, :] = True
        # mask[:, 4] = True
        # vision[~mask] = 0
        # leftVision = vision[4, :4, 0].squeeze().flatten()
        # leftZeroIdx = np.where(leftVision == 0)[0]
        # rightVision = vision[4, 5:, 0].squeeze().flatten()
        # rightZeroIdx = np.where(rightVision == 0)[0]
        # upVision = vision[:4, 4, 0].squeeze().flatten()
        # upZeroIdx = np.where(upVision == 0)[0]
        # downVision = vision[5:, 4, 0].squeeze().flatten()
        # downZeroIdx = np.where(downVision == 0)[0]
        # if len(leftZeroIdx):
        #     vision[4, : leftZeroIdx[-1], :] = 0
        # if len(rightZeroIdx):
        #     vision[4, 5 + rightZeroIdx[0] :, :] = 0
        # if len(upZeroIdx):
        #     vision[: upZeroIdx[-1], 4, :] = 0
        # if len(downZeroIdx):
        #     vision[5 + downZeroIdx[0] :, 4, :] = 0

        if self._memoryLen > 1:
            if not self._memory:
                self._memory = [vision]
            else:
                self._memory.append(vision)
            if len(self._memory) > self._memoryLen:
                self._memory.pop(0)
        return vision

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._memory = None
        return super().reset(seed=seed)
