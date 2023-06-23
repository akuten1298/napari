import bisect

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


class aabb:
    max_tree_height = 5
    min_primitives_per_node = 1024
    total_len = 1
    bin_size = 512
    maxDiff = 0

    def construct_bvh(self, triangles):
        if len(triangles) == 0:
            return None

        bounding_box = BoundingBox(
            np.min(triangles, axis=(0, 1)),
            np.max(triangles, axis=(0, 1)),
            triangles,
        )

        if len(triangles) <= aabb.min_primitives_per_node:
            return BVHNode(None, None, bounding_box)

        bucket_boundaries = aabb.create_buckets(
            self, bounding_box.min_coords[0], bounding_box.max_coords[0]
        )
        store = [[] * aabb.bin_size for _ in range(aabb.bin_size)]

        mean_x = np.mean(triangles[:, :, 0], axis=1)

        indices = np.digitize(mean_x, bucket_boundaries)
        # print("indices: ", indices, " indices len: ", len(indices))

        for index in range(len(triangles)):
            store[indices[index] - 1].append(triangles[index])

        prefix_sum = [0] * aabb.bin_size
        prefix_sum[0] = len(store[0])

        for i in range(1, len(store)):
            prefix_sum[i] = prefix_sum[i - 1] + len(store[i])

        lower_bound_index = bisect.bisect_left(prefix_sum, len(triangles) / 2)
        left_subset = store[:lower_bound_index]
        non_empty_left_subset = [arr for arr in left_subset if len(arr) > 0]

        left_node = None
        if len(non_empty_left_subset) > 0:
            left_array = np.concatenate(non_empty_left_subset)
            if len(triangles) == len(left_array):
                return BVHNode(None, None, bounding_box)
            left_node = self.construct_bvh(self, left_array)

        right_subset = store[lower_bound_index:]
        non_empty_right_subset = [arr for arr in right_subset if len(arr) > 0]

        right_node = None
        if len(non_empty_right_subset) > 0:
            right_array = np.concatenate(non_empty_right_subset)
            if len(triangles) == len(right_array):
                return BVHNode(None, None, bounding_box)
            aabb.maxDiff = max(
                aabb.maxDiff, abs(len(left_array) - len(right_array))
            )
            right_node = self.construct_bvh(self, right_array)

        return BVHNode(left_node, right_node, bounding_box)

    def get_max_diff(self):
        return aabb.maxDiff

    def create_buckets(self, xmin, xmax):
        # Create an array of bucket boundaries
        bucket_boundaries = np.linspace(xmin, xmax, aabb.bin_size + 1)

        return bucket_boundaries

    def ray_box_intersection(self, ray_origin, ray_direction, bounding_box):
        tentry = (bounding_box.min_coords[0] - ray_origin[0]) / ray_direction[
            0
        ]
        texit = (bounding_box.max_coords[0] - ray_origin[0]) / ray_direction[0]

        if tentry > texit:
            tentry, texit = texit, tentry

        if ray_direction[1] != 0:
            tentryy = (
                bounding_box.min_coords[1] - ray_origin[1]
            ) / ray_direction[1]
            texity = (
                bounding_box.max_coords[1] - ray_origin[1]
            ) / ray_direction[1]

            if tentryy > texity:
                tentryy, texity = texity, tentryy

            tentry = max(tentry, tentryy)
            texit = min(texit, texity)

        if ray_direction[2] != 0:
            tentryz = (
                bounding_box.min_coords[2] - ray_origin[2]
            ) / ray_direction[2]
            texitz = (
                bounding_box.max_coords[2] - ray_origin[2]
            ) / ray_direction[2]

            if tentryz > texitz:
                tentryz, texitz = texitz, tentryz

            tentry = max(tentry, tentryz)
            texit = min(texit, texitz)

        if tentry <= texit:
            return True

        return False

    def traverse_bvh(self, ray_origin, ray_direction, node):
        if node is None:
            return None, None

        if self.ray_box_intersection(
            self, ray_origin, ray_direction, node.bbox
        ):
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

            left_intersection_index, left_intersection = self.traverse_bvh(
                self, ray_origin, ray_direction, node.left
            )
            right_intersection_index, right_intersection = self.traverse_bvh(
                self, ray_origin, ray_direction, node.right
            )

            if (
                left_intersection is not None
                and right_intersection is not None
            ):
                left_distance = np.linalg.norm(left_intersection - ray_origin)
                right_distance = np.linalg.norm(
                    right_intersection - ray_origin
                )
                if left_distance < right_distance:
                    return left_intersection_index, left_intersection

                return right_intersection_index, right_intersection
            if left_intersection is not None:
                return left_intersection_index, left_intersection

            return right_intersection_index, right_intersection

        return None, None
