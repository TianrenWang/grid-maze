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
    maze = [[1 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if random.random() < p_obstacle:
                maze[i][j] = 0

    if landmarks:
        halfPoint = size // 2
        innerRoomStartIndex = halfPoint - 15

        def createOuterLandmark(row: int, col: int, landmarkSize: int, omitId: int):
            for i in range(
                innerRoomStartIndex + row, innerRoomStartIndex + row + landmarkSize
            ):
                for j in range(
                    innerRoomStartIndex + col, innerRoomStartIndex + col + landmarkSize
                ):
                    maze[i][j] = 0

            omitRow = omitId // landmarkSize
            omitCol = omitId % landmarkSize

            maze[innerRoomStartIndex + row + omitRow][
                innerRoomStartIndex + col + omitCol
            ] = 1

        createOuterLandmark(3, 3, 3, 0)
        createOuterLandmark(3, 14, 3, 1)
        createOuterLandmark(3, 25, 3, 2)
        createOuterLandmark(14, 3, 3, 3)
        createOuterLandmark(14, 24, 3, 5)
        createOuterLandmark(25, 3, 3, 6)
        createOuterLandmark(25, 14, 3, 7)
        createOuterLandmark(25, 25, 3, 8)

        createOuterLandmark(9, 9, 2, 0)
        createOuterLandmark(9, 20, 2, 1)
        createOuterLandmark(20, 9, 2, 2)
        createOuterLandmark(20, 20, 2, 3)

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
    maze = generateMaze(20)
    print(getMazeDebugString(maze))
