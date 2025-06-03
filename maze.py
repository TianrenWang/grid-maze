import random


def generate_maze(dimensions: tuple[int], goal: tuple[int]):
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


def print_maze(maze):
    for row in maze:
        rowContent = "".join(str(cell) if cell else " " for cell in row)
        if rowContent.strip():
            print("." + rowContent[1:-1] + ".")
        else:
            print("." * len(row))


if __name__ == "__main__":
    rows, cols = 13, 13  # Define maze size (must be odd numbers)
    maze = generate_maze((rows, cols), (7, 7))
    print_maze(maze)
