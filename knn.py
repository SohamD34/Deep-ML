def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
    """
    distances = []

    for point in points:
        distance = (point[0] - query_point[0]) ** 2 + (point[1] - query_point[1]) ** 2
        distances.append((distance, point))

    distances.sort(key=lambda x: x[0])
    nearest_neighbors = [point for _, point in distances[:k]]
    return nearest_neighbors


points = [(1, 2), (3, 4), (1, 1), (5, 6), (2, 3)]
query_point = (2, 2)
k = 3
print(k_nearest_neighbors(points, query_point, k))