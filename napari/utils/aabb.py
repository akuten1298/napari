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


min_primitives_per_node = 512


def construct_bvh(triangles):
    if len(triangles) == 0:
        return None

    bounding_box = BoundingBox(
        np.min(triangles, axis=(0, 1)),
        np.max(triangles, axis=(0, 1)),
        triangles,
    )

    if len(triangles) <= min_primitives_per_node:
        return BVHNode(None, None, bounding_box)

    split_axis = np.argmax(bounding_box.max_coords - bounding_box.min_coords)
    sort_indices = np.argsort(triangles[:, :, split_axis].mean(axis=-1))
    sorted_triangles = triangles[sort_indices]

    split_idx = len(sorted_triangles) // 2
    left_triangles = sorted_triangles[:split_idx]
    right_triangles = sorted_triangles[split_idx:]

    left_node = construct_bvh(left_triangles)
    right_node = construct_bvh(right_triangles)

    return BVHNode(left_node, right_node, bounding_box)


def traverse_bvh(self, ray_origin, ray_direction, node):
    if node is None:
        return None, None, None

    if ray_box_intersection(ray_origin, ray_direction, node.bbox):
        if node.left is None and node.right is None:
            (
                intersection_index,
                intersection,
                closest_intersected_triangle_index,
            ) = find_nearest_triangle_intersection(
                ray_position=ray_origin,
                ray_direction=ray_direction,
                triangles=np.array(node.bbox.triangles),
            )
            if intersection_index is None:
                return None, None, None

            return (
                intersection_index,
                intersection,
                closest_intersected_triangle_index,
            )

        (
            left_intersection_index,
            left_intersection,
            left_closest_intersected_triangle_index,
        ) = traverse_bvh(self, ray_origin, ray_direction, node.left)
        (
            right_intersection_index,
            right_intersection,
            right_closest_intersected_triangle_index,
        ) = traverse_bvh(self, ray_origin, ray_direction, node.right)

        if left_intersection is not None and right_intersection is not None:
            left_distance = np.linalg.norm(left_intersection - ray_origin)
            right_distance = np.linalg.norm(right_intersection - ray_origin)
            if left_distance < right_distance:
                return (
                    left_intersection_index,
                    left_intersection,
                    left_closest_intersected_triangle_index,
                )

            return (
                right_intersection_index,
                right_intersection,
                right_closest_intersected_triangle_index,
            )
        if left_intersection is not None:
            return (
                left_intersection_index,
                left_intersection,
                left_closest_intersected_triangle_index,
            )

        return (
            right_intersection_index,
            right_intersection,
            right_closest_intersected_triangle_index,
        )

    return None, None, None


def ray_box_intersection(ray_origin, ray_direction, bounding_box):
    tentry = (bounding_box.min_coords[0] - ray_origin[0]) / ray_direction[0]
    texit = (bounding_box.max_coords[0] - ray_origin[0]) / ray_direction[0]

    if tentry > texit:
        tentry, texit = texit, tentry

    if ray_direction[1] != 0:
        tentryy = (bounding_box.min_coords[1] - ray_origin[1]) / ray_direction[
            1
        ]
        texity = (bounding_box.max_coords[1] - ray_origin[1]) / ray_direction[
            1
        ]

        if tentryy > texity:
            tentryy, texity = texity, tentryy

        tentry = max(tentry, tentryy)
        texit = min(texit, texity)

    if ray_direction[2] != 0:
        tentryz = (bounding_box.min_coords[2] - ray_origin[2]) / ray_direction[
            2
        ]
        texitz = (bounding_box.max_coords[2] - ray_origin[2]) / ray_direction[
            2
        ]

        if tentryz > texitz:
            tentryz, texitz = texitz, tentryz

        tentry = max(tentry, tentryz)
        texit = min(texit, texitz)

    if tentry <= texit:
        return True

    return False
