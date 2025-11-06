import numpy as np


def isInbetween(value: int, start: int, end: int):
    if start < end:
        return start <= value <= end
    else:
        return start >= value >= end


def getBoundaryPoint(startPosition: np.ndarray, posChange: np.ndarray):
    nudge = 1e-8
    m = posChange[1] / posChange[0]
    b = startPosition[1] - m * startPosition[0]
    if posChange[0] > 0:
        verticalBoundary = np.ceil(startPosition[0]) - nudge
    else:
        verticalBoundary = np.floor(startPosition[0]) + nudge

    if posChange[1] > 0:
        horizontalBoundary = np.ceil(startPosition[1]) - nudge
    else:
        horizontalBoundary = np.floor(startPosition[1]) + nudge

    horizontalIntersection = (horizontalBoundary - b) / m
    verticalIntersection = verticalBoundary * m + b

    horizontalDistanceToVerticalBoundary = abs(verticalBoundary - startPosition[0])
    horizontalDistanceToHorizontalBoundary = abs(
        (horizontalBoundary - startPosition[1]) / m
    )

    if horizontalDistanceToHorizontalBoundary > horizontalDistanceToVerticalBoundary:
        return np.array([verticalBoundary, verticalIntersection])
    else:
        return np.array([horizontalIntersection, horizontalBoundary])


def getOverlap(center: np.ndarray, grid_size: int):
    auraRadius = 0.5

    corners = dict()
    corners["topLeft"] = center + np.array([-auraRadius, -auraRadius])
    corners["topRight"] = center + np.array([auraRadius, -auraRadius])
    corners["bottomLeft"] = center + np.array([-auraRadius, auraRadius])
    corners["bottomRight"] = center + np.array([auraRadius, auraRadius])

    centerIntersection = np.ceil(corners["topLeft"])

    map = np.zeros([grid_size, grid_size])
    if np.array_equal(centerIntersection, corners["topLeft"]):
        map[int(center[0]), int(center[1])] = 1
        return map

    for key in corners.keys():
        corner = corners[key]
        proportion = abs(np.prod(centerIntersection - corner))
        squareCoordinate = np.floor(corner)
        xCoordinate = int(squareCoordinate[0])
        yCoordinate = int(squareCoordinate[1])
        if (
            xCoordinate >= 0
            and yCoordinate >= 0
            and xCoordinate < grid_size
            and yCoordinate < grid_size
        ):
            map[xCoordinate, yCoordinate] = proportion

    return map


if __name__ == "__main__":
    output = getBoundaryPoint(
        np.array([0.47449042, 3.48091578]), np.array([0.6864419, -6.245576])
    )
    print(output)
