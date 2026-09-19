import random

import matplotlib.pyplot as plt
import numpy as np


def generateDeprecatedMaze(dimensions: tuple[int], goal: tuple[int]):
    rows, cols = dimensions
    # Ensure odd dimensions for proper walls
    if rows % 2 == 0:
        rows += 1
    if cols % 2 == 0:
        cols += 1

    # Initialize the grid with walls (0s)
    maze = [[0 for _ in range(cols)] for _ in range(rows)]

    # Movement directions: (row_offset, col_offset)
    directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]

    # Recursive DFS function
    def dfs(r, c):
        maze[r][c] = 1  # Mark as path
        random.shuffle(directions)  # Randomize path generation

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 1 <= nr < rows - 1 and 1 <= nc < cols - 1:
                midR = r + dr // 2
                midC = c + dc // 2
                if maze[nr][nc] == 0:
                    maze[midR][midC] = 1  # Remove wall between
                    dfs(nr, nc)

    # Start DFS from a random odd coordinate
    start_row, start_col = goal
    dfs(start_row, start_col)

    return maze


def hasAmbiguousRegions(maze) -> bool:
    for i in range(85, 85 + 31):
        for j in range(85, 85 + 31):
            obs = maze[i - 4 : i + 5, j - 4 : j + 5, 0]
            if obs.sum() == 0:
                return True
    return False


def generateMaze(size: int = 200, p_obstacle: float = 0.1, landmarks: bool = True):
    maze = [[(0, 0, 0) for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if random.random() < p_obstacle:
                maze[i][j] = (0.2, 0.2, 0.2)

    if landmarks:
        halfPoint = size // 2
        innerRoomStartIndex = halfPoint - 15

        def createLandmark(row: int, col: int, color: tuple[float, float, float]):
            maze[innerRoomStartIndex + row][innerRoomStartIndex + col] = color

        rng = random.Random(0)

        numLandmarks = 70
        colors = [
            (rng.random(), rng.random(), rng.random()) for _ in range(numLandmarks)
        ]
        coordinates = [
            (rng.randint(0, 30), rng.randint(0, 30)) for _ in range(numLandmarks)
        ]

        # validator = np.zeros([size, size, 1])

        for i, coordinate in enumerate(coordinates):
            createLandmark(coordinate[0], coordinate[1], colors[i])
            # validator[
            #     innerRoomStartIndex + coordinate[0],
            #     innerRoomStartIndex + coordinate[1],
            #     0,
            # ] = 1

        createLandmark(15, 15, (1, 1, 1))  # goal landmark
        # validator[innerRoomStartIndex + 15, innerRoomStartIndex + 15, 0] = 1
        # print(hasAmbiguousRegions(validator))

    return maze


def generateMazeWithOfflimit(size: int):
    maze = [[1 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i % 4 == 0 and j % 4 == 0:
                for x in range(2):
                    for y in range(2):
                        if -1 < i + y < size and -1 < j + x < size:
                            maze[i + y][j + x] = 0
    return maze


def getMazeDebugString(maze):
    mazeString = []
    mazeString.append("".join("." for i in range(len(maze) + 2)))
    for row in maze:
        rowContent = "".join(str(cell) for cell in row)
        mazeString.append("." + rowContent + ".")
    mazeString.append("".join("." for i in range(len(maze) + 2)))
    return "\n".join(mazeString)


if __name__ == "__main__":
    maze = generateMaze(200)
    plt.imshow(np.array(maze))
    plt.axis("off")
    plt.show()
