from typing import Optional
from collections import deque
import numpy as np
import gymnasium as gym

from maze import print_maze, generateMaze
from environment_utils import getBoundaryPoint, getOverlap


class MazeEnv(gym.Env):
    def __init__(self, config=None):
        self._episode_len = 0
        self._mazeArray = config.get("maze", None)
        self._mazeSize = config.get("mazeSize", None)
        self._randomMaze = not self._mazeArray
        self._goalLocation = config.get("goal", None)
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

    def getShortestDistance(self):
        size = len(self._mazeArray)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = deque([(1, 1, 0)])

        visited = set()
        visited.add((self._agentLocation[0], self._agentLocation[1]))
        maze = self._mazeArray

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
            allLocations = []
            for i in range(self._mazeSize):
                for j in range(self._mazeSize):
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

        if self._debugging:
            self._mazeTracker = []
            for i in range(self._mazeSize):
                currentRow = []
                self._mazeTracker.append(currentRow)
                for j in range(self._mazeSize):
                    originalValue = self._mazeArray[i][j]
                    if not originalValue:
                        currentRow.append("X")
                    elif originalValue == 1:
                        currentRow.append(0)
                    else:
                        currentRow.append(originalValue)
            self._mazeTracker[self._agentLocation[0]][self._agentLocation[1]] = "S"
            self._mazeTracker[self._goalLocation[0]][self._goalLocation[1]] = "*"
            self._shortestDistance = np.sum(
                np.abs(self._agentLocation - self._goalLocation)
            )

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
            and self._map[int(location[0]), int(location[1]), 0] == 1
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
        reward = 1 if terminated else 0
        if self._debugging:
            agentLocationValue = self._mazeTracker[self._agentLocation[0]][
                self._agentLocation[1]
            ]
            if isinstance(agentLocationValue, int) and agentLocationValue < 9:
                self._mazeTracker[self._agentLocation[0]][self._agentLocation[1]] += 1
            if terminated or truncated:
                self.render()
        return self._getObs(), reward, terminated, truncated, self._get_info()

    def render(self):
        for i in range(self._mazeSize):
            for j in range(self._mazeSize):
                if not self._mazeTracker[i][j]:
                    self._mazeTracker[i][j] = " "
        print_maze(self._mazeTracker)
        print("Steps:", self._episode_len)
        print("Shortest:", self._shortestDistance)


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
        _paddedAgentLoc = self._agentLocation.astype(int) + np.array((4, 4))
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


class PlaceMazeEnv(FoggedMazeEnv):
    def __init__(self, config=None):
        super().__init__(config)
        visualObsSize = self._visualRange * 2 + 1
        self._lastLocation = self._agentLocation
        self.observation_space = gym.spaces.Box(
            0, self._mazeSize, (visualObsSize**2 * 3 + 4 + self.action_space.n + 1,)
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
                self._lastLocation / (self._mazeSize - 1),
                self._agentLocation / (self._mazeSize - 1),
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
            [0 for j in range(self._mazeSize)] for i in range(self._mazeSize)
        ]
        self.observation_space = gym.spaces.Box(
            0, self._mazeSize, (visualObsSize**2 * 3 + 4 + self.action_space.n + 1,)
        )

    def step(self, action):
        finalized = False
        availableActions = [0, 1, 2, 3]
        while not finalized:
            availableActions.remove(action)
            if len(availableActions):
                action = np.random.choice(availableActions, 1)[0]
            else:
                finalized = True
            direction = self._action_to_direction[action]
            newLoc = self._agentLocation + direction
            if (
                self.isValidLocation(newLoc)
                and self._visitCounts[newLoc[0]][newLoc[1]] < 3
            ):
                finalized = True

        self._lastLocation = self._agentLocation
        self._lastAction = action
        stepOutput = super().step(action)
        self._visitCounts[self._agentLocation[0]][self._agentLocation[1]] += 1
        return stepOutput


class ContinuousMazeEnv(FoggedMazeEnv):
    def __init__(self, config=None):
        super().__init__(config)
        visualObsSize = self._visualRange * 2 + 1
        self._lastLocation = self._agentLocation
        self.observation_space = gym.spaces.Box(
            0,
            1,
            (
                visualObsSize**2 * 3 + 4 + 3,
            ),  # vision vector + last location + current location + speed + sin + cos
        )
        self.action_space = gym.spaces.Box(-np.pi, np.pi, (2,))
        self._currentDirection = np.random.uniform(0, 2 * np.pi)
        self._lastDirection = self._currentDirection
        self._actionTaken = np.array([0, 0])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._lastLocation = np.array([1, 1])
        super().reset(seed=seed, options=options)
        self._lastLocation = self._agentLocation
        self._currentDirection = np.random.uniform(0, 2 * np.pi)
        self._lastDirection = self._currentDirection
        self._actionTaken = np.array([0, 0])
        return self._getObs(), self._get_info()

    def step(self, action):
        """
        Write this one from scratch. Write it first for when agent controls action,
        then write one for path integration.
        """
        speed = action[0]
        newDirection = self._currentDirection + action[1]
        posChange = np.array(
            [speed * np.cos(newDirection), speed * np.sin(newDirection)]
        )
        self._lastLocation = self._agentLocation
        newLocation = self._agentLocation + posChange
        if self.isValidLocation(newLocation):
            self._agentLocation = newLocation
            self._actionTaken = action
        else:
            self._agentLocation = getBoundaryPoint(self._agentLocation, posChange)
            for i in range(2):
                self._agentLocation[i] = (
                    self._agentLocation[i]
                    if self._agentLocation[i] < self._mazeSize
                    else self._agentLocation[i] - 1e-8
                )

        overlaps = getOverlap(self._agentLocation, self._mazeSize)
        self._map[:, :, 2] = np.array(overlaps)

        terminated = np.array_equal(np.floor(self._agentLocation), self._goalLocation)
        self._episode_len += 1
        truncated = self._episode_len > self._maxSteps
        reward = 1 if terminated else 0
        if self._debugging:
            agentLocationValue = self._mazeTracker[int(self._agentLocation[0])][
                int(self._agentLocation[1])
            ]
            if isinstance(agentLocationValue, int) and agentLocationValue < 9:
                self._mazeTracker[int(self._agentLocation[0])][
                    int(self._agentLocation[1])
                ] += 1
            if terminated or truncated:
                self.render()
        return self._getObs(), reward, terminated, truncated, self._get_info()

    def _getObs(self):
        vision = super()._getObs()
        action = np.array(
            [
                self._actionTaken[0],
                np.cos(self._actionTaken[1]),
                np.sin(self._actionTaken[1]),
            ],
        )
        return np.concatenate(
            [
                vision.flatten(),
                self._lastLocation / self._mazeSize,
                self._agentLocation / self._mazeSize,
                action,
            ],
            dtype=np.float32,
        )
