from typing import Optional
from collections import deque
import numpy as np
import gymnasium as gym
import random

from maze import getMazeDebugString, generateMaze


class MazeEnv(gym.Env):
    def __init__(self, config=None):
        self._episode_len = 0
        self._mazeArray = config.get("maze", None)
        self._mazeSize = config.get("mazeSize", None)
        self._actualMazeSize = self._mazeSize * 2
        self._randomMaze = not self._mazeArray
        self._goalLocation = [self._mazeSize, self._mazeSize]
        self._fixedGoal = bool(self._goalLocation)
        self._startLocation = config.get("start", None)
        self._maxSteps = config["maxSteps"]
        self._actionTaken = 4
        self._debugging = config.get("debugging", None)
        self._mazeTracker = []
        self._shortestDistance = 0

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

    def getWorstCaseTry(self, location: int, goal: int):
        if location <= 4 or self._mazeSize - location <= 4:
            return abs(goal - location)
        elif location > goal:
            return self._mazeSize - 4 - location + self._mazeSize - 4 - goal
        else:
            return location + goal - 8

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self._randomMaze:
            self._mazeArray = generateMaze(self._actualMazeSize)

        mazeSize = len(self._mazeArray)

        targetChannel = np.zeros([mazeSize, mazeSize, 1], dtype=np.int32)
        targetChannel[self._goalLocation[0], self._goalLocation[1], 0] = 1
        agentChannel = np.zeros([mazeSize, mazeSize, 1], dtype=np.int32)
        if not self._startLocation:
            allLocations = []
            _range = range(self._mazeSize // 2, self._mazeSize // 2 + self._mazeSize)
            for i in _range:
                for j in _range:
                    allLocations.append((i, j))
            np.random.shuffle(allLocations)
            agentLocation = np.array(allLocations.pop())
            goalDiff = np.abs(agentLocation - self._goalLocation)
            isCloseToGoal = goalDiff[0] <= 6 and goalDiff[1] <= 6
            while len(allLocations) and (
                np.array_equal(agentLocation, self._goalLocation)
                or not self._mazeArray[agentLocation[0]][agentLocation[1]]
                or isCloseToGoal
            ):
                agentLocation = np.array(allLocations.pop())
                goalDiff = np.abs(agentLocation - self._goalLocation)
                isCloseToGoal = goalDiff[0] <= 6 and goalDiff[1] <= 6
            self._agentLocation = agentLocation
            agentChannel[agentLocation[0], agentLocation[1], 0] = 1
        else:
            agentChannel[self._startLocation[0], self._startLocation[1], 0] = 1
            self._agentLocation = np.array(self._startLocation, dtype=np.int32)

        self._mazeTracker = []
        for i in range(self._actualMazeSize):
            currentRow = []
            self._mazeTracker.append(currentRow)
            for j in range(self._actualMazeSize):
                originalValue = self._mazeArray[i][j]
                if not originalValue:
                    currentRow.append("X")
                elif originalValue == 1:
                    currentRow.append(0)
                else:
                    currentRow.append(originalValue)
        self._mazeTracker[self._agentLocation[0]][self._agentLocation[1]] = "S"
        self._mazeTracker[self._goalLocation[0]][self._goalLocation[1]] = "*"

        self._pastLocation = self._agentLocation
        mazeChannel = np.expand_dims(self._mazeArray, axis=2)
        self._map = np.concat((mazeChannel, targetChannel, agentChannel), axis=2)
        self._episode_len = 0
        if not self._maxSteps:
            self._maxSteps = self.getShortestDistance()
        return self._getObs(), self._get_info()

    def isValidLocation(self, location: np.ndarray):
        return (
            0 <= location[0] < len(self._mazeArray)
            and 0 <= location[1] < len(self._mazeArray)
            and self._map[location[0], location[1], 0] == 1
        )

    def step(self, action):
        direction = self._action_to_direction[action]
        newLoc = self._agentLocation + direction
        if self.isValidLocation(newLoc):
            self._map[self._agentLocation[0], self._agentLocation[1], 2] = 0
            self._agentLocation = newLoc
            self._map[self._agentLocation[0], self._agentLocation[1], 2] = 1
            self._actionTaken = action
        else:
            self._actionTaken = 4

        terminated = np.array_equal(self._agentLocation, self._goalLocation)
        self._episode_len += 1
        truncated = self._episode_len > self._maxSteps
        if terminated:
            if self._episode_len > 100:
                reward = 0.1
            else:
                closenessFactor = 1 - self._episode_len / 100
                reward = 0.1 + closenessFactor**2
        else:
            reward = 0

        agentLocationValue = self._mazeTracker[self._agentLocation[0]][
            self._agentLocation[1]
        ]
        if isinstance(agentLocationValue, int) and agentLocationValue < 9:
            self._mazeTracker[self._agentLocation[0]][self._agentLocation[1]] += 1
        if (terminated or truncated) and self._debugging:
            print("Steps:", self._episode_len)
            print("Shortest:", self._shortestDistance)
            print(self.render())
        return self._getObs(), reward, terminated, truncated, self._get_info()

    def render(self):
        renderOutput = [
            [i for i in range(self._actualMazeSize)]
            for j in range(self._actualMazeSize)
        ]
        for i in range(self._actualMazeSize):
            for j in range(self._actualMazeSize):
                if not self._mazeTracker[i][j]:
                    renderOutput[i][j] = " "
                else:
                    renderOutput[i][j] = self._mazeTracker[i][j]
        return getMazeDebugString(renderOutput)


class FoggedMazeEnv(MazeEnv):
    def __init__(self, config=None):
        super().__init__(config)
        self._visualRange = config.get("visualRange", 4)
        visualObsSize = self._visualRange * 2 + 1
        self.observation_space = gym.spaces.MultiBinary(
            (visualObsSize, visualObsSize, 2)
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
        return vision[:, :, :2]


class PlaceMazeEnv(FoggedMazeEnv):
    def __init__(self, config=None):
        super().__init__(config)
        visualObsSize = self._visualRange * 2 + 1
        self._lastLocation = self._agentLocation
        self.observation_space = gym.spaces.Box(
            0, self._mazeSize, (visualObsSize**2 * 2 + 4 + self.action_space.n + 1,)
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._lastLocation = np.array([1, 1])
        super().reset(seed=seed, options=options)
        self._lastLocation = self._agentLocation
        return self._getObs(), self._get_info()

    def step(self, action):
        self._lastLocation = self._agentLocation
        return super().step(action)

    def _getObs(self):
        vision = super()._getObs()
        actionOneHot = np.zeros(5)
        actionOneHot[self._actionTaken] = 1
        return np.concatenate(
            [
                vision.flatten(),
                (self._lastLocation - self._mazeSize // 2) / self._mazeSize,
                (self._lastLocation - self._mazeSize // 2) / self._mazeSize,
                actionOneHot,
            ],
            dtype=np.float32,
        )


class SelfLocalizeEnv(PlaceMazeEnv):
    def __init__(self, config=None):
        super().__init__(config)
        visualObsSize = self._visualRange * 2 + 1
        self._lastLocation = self._agentLocation
        self._lastAction = np.random.randint(0, 4)
        self._visitCounts = [
            [0 for j in range(self._actualMazeSize)]
            for i in range(self._actualMazeSize)
        ]
        self.observation_space = gym.spaces.Box(
            0, 1, (visualObsSize**2 * 2 + 4 + self.action_space.n + 1,)
        )
        self.previousActions = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.previousActions = deque()
        return self._getObs(), self._get_info()

    def step(self, action):
        """
        Overrides the action. Forces agent to move in a directional manner that doesn't
        end in jitter. It follows the following heuristics ranked by priority:
        1. Must not run into walls (prevent nullified movement)
        2. It cannot move in the opposite direction of any actions it made in the last
        three moves.
        3. Always selects a direction that would lead to the least number of visited
        position.
        4. In case of a tie, randomly chooses amongst the tied choices uniformly.
        """
        availableActions = [0, 1, 2, 3]
        np.random.shuffle(availableActions)

        candidates = []

        for a in availableActions:
            direction = self._action_to_direction[a]
            newLoc = self._agentLocation + direction
            if self.isValidLocation(newLoc):
                visit_count = self._visitCounts[newLoc[0]][newLoc[1]]
                candidates.append((visit_count, a))

        for i in range(len(candidates)):
            candidate = candidates[i]
            candidateAction = candidate[1]
            if (
                candidateAction == 0
                and 2 in self.previousActions
                or candidateAction == 1
                and 3 in self.previousActions
                or candidateAction == 2
                and 0 in self.previousActions
                or candidateAction == 3
                and 1 in self.previousActions
            ):
                candidates[i] = (random.randrange(9000, 9999), candidateAction)

        _, best_action = min(candidates, key=lambda x: x[0])

        self._lastLocation = self._agentLocation
        self._lastAction = best_action
        stepOutput = super().step(best_action)
        self._visitCounts[self._agentLocation[0]][self._agentLocation[1]] += 1
        self.previousActions.append(best_action)
        if len(self.previousActions) > 3:
            self.previousActions.popleft()
        return stepOutput
