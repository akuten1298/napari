import numpy as np


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


def construct_bvh(triangles):
    if len(triangles) == 0:
        return None

    all_vertices = np.concatenate(triangles)
    bounding_box = BoundingBox(
        np.min(all_vertices, axis=0), np.max(all_vertices, axis=0), triangles
    )

    np.argmax(bounding_box.max_coords - bounding_box.min_coords)
    # TODO: Should be optimized to use SAH partitioning technique.
    # Tradeoff - might take longer time to build the tree but possibly faster querying when checking for intersection.
    split_axis = 0
    sorted_triangles = sorted(
        triangles, key=lambda triangle: bounding_box.min_coords[split_axis]
    )
    sorted_triangles = triangles

    if len(sorted_triangles) <= 1024:
        return BVHNode(None, None, bounding_box)

    split_idx = len(sorted_triangles) // 2
    left_triangles = sorted_triangles[:split_idx]
    right_triangles = sorted_triangles[split_idx:]

    left_node = construct_bvh(left_triangles)
    right_node = construct_bvh(right_triangles)

    return BVHNode(left_node, right_node, bounding_box)


# Print the bounding boxes and triangle indices
def print_bounding_boxes(bvh_node: BVHNode, depth=0):
    indent = "  " * depth
    bounding_box: BoundingBox = bvh_node.bbox
    print(
        f"{indent}Bounding Box: {bounding_box.min_coords} - {bounding_box.max_coords}"
    )
    print(f"{indent}Triangle Indices: {bounding_box.triangles}")

    if bvh_node.left is None and bvh_node.right is None:
        print(f"{indent}Leaf Node")
    else:
        print(f"{indent}Internal Node")
        print_bounding_boxes(bvh_node.left, depth + 1)
        print_bounding_boxes(bvh_node.right, depth + 1)


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
