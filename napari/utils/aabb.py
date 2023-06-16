import bisect
import math

import numpy as np

from napari.utils.geometry import find_nearest_triangle_intersection


class BVHNode:
    def __init__(self, left, right, bbox):
        self.left = left
        self.right = right
        self.bbox = bbox


class BoundingBox:
    def __init__(self, min_coords, max_coords, triangles):
        self.min_coords = min_coords
        self.max_coords = max_coords
        self.triangles = triangles


class Triangles:
    def _init_(self, triangle, index):
        self.triangles = triangles
        self.index = index


max_tree_height = 5
min_primitives_per_node = 1024
total_len = 1
bin_size = 10


def setup_bvh(triangles):
    total_len = len(triangles)
    min_primitives_per_node = math.ceil(
        total_len / (math.pow(2, max_tree_height))
    )
    print("min_primitives_per_node: ", min_primitives_per_node)


def construct_bvh(triangles):
    if len(triangles) == 0:
        return None

    # print("triangle size: ", len(triangles))

    bounding_box = BoundingBox(
        np.min(triangles, axis=(0, 1)),
        np.max(triangles, axis=(0, 1)),
        triangles,
    )

    if len(triangles) <= min_primitives_per_node:
        return BVHNode(None, None, bounding_box)

    # mincost = np.inf
    # minaxis = -1
    # minsplit = -1
    # centroids = np.mean(triangles, axis=1)
    # print("centroids: ", len(centroids))

    # for axis in range(3):
    #     ## sort triangles with respect centroids
    #     sortedtris = triangles[np.argsort(centroids[:, axis])]
    #     print("sorted tris: ", sortedtris[:5])

    # print("diff in  coords: ", bounding_box.max_coords - bounding_box.min_coords)
    # split_axis = np.argmax(bounding_box.max_coords - bounding_box.min_coords)

    # print("testing: ", bounding_box.max_coords[split_axis])
    # split_axis = 0
    # sorted_triangles = np.array(
    #     sorted(triangles, key=lambda triangle: triangle[:, split_axis].mean())
    # )
    # sorted_triangles = triangles;

    bucket_boundaries = create_buckets(
        bounding_box.min_coords[0], bounding_box.max_coords[0]
    )
    store = [[] * 10 for _ in range(10)]
    for triangle in triangles:
        bucket_index = np.digitize(triangle[:, 0].mean(), bucket_boundaries)
        store[bucket_index - 1].append(triangle)
        # print("bucket index for '" + str(triangle[:, 0].mean()) + "' :", bucket_index)

    prefix_sum = [0] * 10
    prefix_sum[0] = len(store[0])

    for i in range(1, len(store)):
        prefix_sum[i] = prefix_sum[i - 1] + len(store[i])

    # for i in range(0, 10):
    #     print(prefix_sum[i])

    # print("bucket gaps between '" + str(bounding_box.min_coords[0]) + "' and '" + str(bounding_box.max_coords[0]) + "' ", create_buckets(bounding_box.min_coords[0], bounding_box.max_coords[0]))

    lower_bound_index = bisect.bisect_left(prefix_sum, len(triangles) / 2)
    # print(lower_bound_index)

    left_array = triangles[: prefix_sum[lower_bound_index]]

    if len(triangles) == len(left_array):
        return BVHNode(None, None, bounding_box)

    left_node = construct_bvh(left_array)
    right_array = triangles[prefix_sum[lower_bound_index] :]
    right_node = construct_bvh(right_array)

    # print("left array: ", left_array[len(left_array)-1])
    # print("right array: ", right_array[0])

    # split_idx = len(sorted_triangles) // 2
    # left_triangles = sorted_triangles[:split_idx]
    # right_triangles = sorted_triangles[split_idx:]

    return BVHNode(left_node, right_node, bounding_box)


def create_buckets(xmin, xmax):
    # Calculate the bucket size based on the range and the number of buckets
    (xmax - xmin) / 10

    # Create an array of bucket boundaries
    bucket_boundaries = np.linspace(xmin, xmax, num=11)

    return bucket_boundaries


# Print the bounding boxes and triangle indices
def print_bounding_boxes(bvh_node: BVHNode, depth=0):
    indent = "  " * depth
    print("Depth: ", depth)
    if bvh_node is not None:
        bounding_box: BoundingBox = bvh_node.bbox

    # print(
    #     f"{indent}Bounding Box: {bounding_box.min_coords} - {bounding_box.max_coords}"
    # )
    print(f"{indent}Triangle Indices: {len(bounding_box.triangles)}")

    if bvh_node.left is None and bvh_node.right is None:
        print(f"{indent}Leaf Node")
    else:
        print(f"{indent}Internal Node")
        print_bounding_boxes(bvh_node.left, depth + 1)
        print_bounding_boxes(bvh_node.right, depth + 1)


def traverse_bvh(self, ray_origin, ray_direction, node):
    if node is None:
        return None, None

    if ray_box_intersection(ray_origin, ray_direction, node.bbox):
        if node.left is None and node.right is None:
            (
                intersection_index,
                intersection,
            ) = find_nearest_triangle_intersection(
                ray_position=ray_origin,
                ray_direction=ray_direction,
                triangles=np.array(node.bbox.triangles),
            )
            if intersection_index is None:
                return None, None

            return intersection_index, intersection

        left_intersection_index, left_intersection = traverse_bvh(
            self, ray_origin, ray_direction, node.left
        )
        right_intersection_index, right_intersection = traverse_bvh(
            self, ray_origin, ray_direction, node.right
        )

        if left_intersection is not None and right_intersection is not None:
            left_distance = np.linalg.norm(left_intersection - ray_origin)
            right_distance = np.linalg.norm(right_intersection - ray_origin)
            if left_distance < right_distance:
                return left_intersection_index, left_intersection

            return right_intersection_index, right_intersection
        if left_intersection is not None:
            return left_intersection_index, left_intersection

        return right_intersection_index, right_intersection

    return None, None


def ray_box_intersection(ray_origin, ray_direction, bounding_box):
    tentryx = (bounding_box.min_coords[0] - ray_origin[0]) / ray_direction[0]
    texitx = (bounding_box.max_coords[0] - ray_origin[0]) / ray_direction[0]

    if tentryx > texitx:
        tentryx, texitx = texitx, tentryx

    tentryy = (bounding_box.min_coords[1] - ray_origin[1]) / ray_direction[1]
    texity = (bounding_box.max_coords[1] - ray_origin[1]) / ray_direction[1]

    if tentryy > texity:
        tentryy, texity = texity, tentryy

    tentry = max(tentryx, tentryy)
    texit = min(texitx, texity)
    if tentry <= texit:
        return True

    return False


# Sample triangle
triangles = [
    [(0, 0, 0), (1, 1, 1), (2, 2, 7)],
    [(-1, 2, 2), (3, 3, 3), (4, 4, 4)],
    [(1, 1, 1), (3, -5, 3), (8, 5, 5)],
    [(0, 0, 0), (1000, 1000, 1000), (2000, 2000, 7000)],
    [(-1000, 2000, 2000), (3000, 3000, 3000), (4000, 4000, 4000)],
    [(1000, 1000, 1000), (3000, -5000, 3000), (8000, 5000, 5000)],
]

# bvh_root = construct_bvh(triangles)
# print_bounding_boxes(bvh_root)
