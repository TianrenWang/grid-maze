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


def generateMaze(size: int, p_obstacle: float = 0.2):
    maze = [[1 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if random.random() < p_obstacle:
                maze[i][j] = 0
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
    # Find the bounding box of all non-empty cells
    non_empty = [
        (r, c)
        for r, row in enumerate(maze)
        for c, cell in enumerate(row)
        if str(cell) != " "
    ]

    if not non_empty:
        return ""

    min_row = min(r for r, c in non_empty)
    max_row = max(r for r, c in non_empty)
    min_col = min(c for r, c in non_empty)
    max_col = max(c for r, c in non_empty)

    # Keep 4 squares of surrounding empty space
    min_row = max(0, min_row - 4)
    max_row = min(len(maze) - 1, max_row + 4)
    min_col = max(0, min_col - 4)
    max_col = min(len(maze[0]) - 1, max_col + 4)

    mazeString = []

    width = max_col - min_col + 1
    mazeString.append("." * (width + 2))

    for row in maze[min_row : max_row + 1]:
        rowContent = "".join(str(cell) for cell in row[min_col : max_col + 1])
        mazeString.append("." + rowContent + ".")

    mazeString.append("." * (width + 2))

    return "\n".join(mazeString)


if __name__ == "__main__":
    maze = generateMaze(20)
    print(getMazeDebugString(maze))
