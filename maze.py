import random


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

        createLandmark(3, 3, (0.9412, 0.2275, 0.2275))
        createLandmark(3, 11, (1.0000, 0.5490, 0.0000))
        createLandmark(3, 19, (0.9608, 0.8157, 0.0000))
        createLandmark(3, 27, (0.5490, 0.8510, 0.0000))
        createLandmark(11, 3, (0.1255, 0.7216, 0.3529))
        createLandmark(11, 27, (0.0000, 0.6588, 0.4706))
        createLandmark(19, 3, (0.0000, 0.7490, 0.7686))
        createLandmark(19, 27, (0.1608, 0.6627, 0.9098))
        createLandmark(27, 3, (0.1569, 0.4706, 0.8157))
        createLandmark(27, 11, (0.2941, 0.3098, 0.8863))
        createLandmark(27, 19, (0.4039, 0.2549, 0.8510))
        createLandmark(27, 27, (0.5686, 0.2745, 0.8471))
        createLandmark(7, 7, (0.8196, 0.2353, 0.8118))
        createLandmark(7, 15, (0.9412, 0.3098, 0.6157))
        createLandmark(7, 23, (0.8471, 0.1059, 0.3765))
        createLandmark(15, 7, (1.0000, 0.4196, 0.3529))
        createLandmark(15, 15, (0.8980, 0.6627, 0.0000))
        createLandmark(15, 23, (0.0000, 0.6863, 0.6275))
        createLandmark(23, 7, (0.0000, 0.5608, 0.6118))
        createLandmark(23, 15, (0.4431, 0.5373, 0.9098))
        createLandmark(23, 23, (0.6667, 0.4392, 0.9098))

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
    maze = generateMaze(41)
    print(getMazeDebugString(maze))
