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


def generateMaze(size: int, p_obstacle: float = 0.02):
    maze = [[1 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if random.random() < p_obstacle:
                obstacleHeight = random.randint(1, 5)
                obstacleWidth = random.randint(1, 5)
                if (
                    i <= size // 2 < i + obstacleHeight
                    and j <= size // 2 < j + obstacleWidth
                ):
                    continue
                for x in range(obstacleWidth):
                    for y in range(obstacleHeight):
                        if -1 < i + y < size and -1 < j + x < size:
                            maze[i + y][j + x] = 0
    return maze


def print_maze(maze):
    print("".join("." for i in range(len(maze) + 2)))
    for row in maze:
        rowContent = "".join(str(cell) for cell in row)
        print("." + rowContent + ".")
    print("".join("." for i in range(len(maze) + 2)))


if __name__ == "__main__":
    maze = generateMaze(20)
    print_maze(maze)
