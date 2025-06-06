import numpy as np

def distance(point1, point2):
	return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
	
    current_centroids = initial_centroids

    for iter in range(max_iterations):
        
        # print(f"\nIteration {iter+1}")

        # Allocating points to clusters

        new_allocations = [[] for centers in range(k)]
        
        for point in points:

            min_dist = 10e5
            min_centroid_no = None

            for centroid_no in range(k):
                centroid = current_centroids[centroid_no]
                d = distance(point, centroid)

                if d < min_dist:
                    min_dist = d
                    min_centroid_no = centroid_no

            new_allocations[min_centroid_no].append(point)

        # print(new_allocations)
        # Computing new cluster centers

        new_centroids = []

        for cluster_no in range(len(new_allocations)):
             
            cluster = new_allocations[cluster_no]
            centroid = np.mean(cluster, axis=0)
            new_centroids.append(tuple(centroid))

        if all(np.array_equal(current_centroids[i], new_centroids[i]) for i in range(len(current_centroids))):
            break

        current_centroids = new_centroids
        # print('Centroids =', current_centroids)
      
    return current_centroids


# points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)]
# k = 2
# initial_centroids = [(1, 1), (10, 1)]
# max_iterations = 10
# final_centroids = k_means_clustering(points, k, initial_centroids, max_iterations)
# print(final_centroids)  

print(k_means_clustering([(1, 1), (2, 2), (3, 3), (4, 4)], 1, [(0,0)], 10))