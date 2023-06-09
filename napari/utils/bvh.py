import numpy as np


class BoundingBox:
    def __init__(self, min_coord, max_coord, triangle_indices):
        self.min = min_coord
        self.max = max_coord
        self.triangle_indices = triangle_indices


def construct_bvh(triangle_array):
    # Construct a bounding volume hierarchy (BVH) using axis-aligned bounding boxes
    # Return the root node of the BVH

    # Create a list of bounding boxes with indices and coordinates
    bounding_boxes = [
        triangle_bounding_box(triangle, index)
        for index, triangle in enumerate(triangle_array)
    ]

    # Recursively build the BVH tree
    return build_bvh(bounding_boxes)


def triangle_bounding_box(triangle, triangle_index):
    # Calculate the bounding box of a triangle based on the coordinates
    # Return a BoundingBox object

    min_coord = np.min(triangle, axis=0)
    max_coord = np.max(triangle, axis=0)

    return BoundingBox(min_coord, max_coord, [triangle_index])


def build_bvh(bounding_boxes):
    if len(bounding_boxes) == 1:
        # Leaf node, return the bounding box
        return bounding_boxes[0]

    # Find the optimal split position
    split_axis, split_pos = find_split_position(bounding_boxes)

    # Partition the bounding boxes into two groups based on the split position
    left_boxes, right_boxes = partition_bounding_boxes(
        bounding_boxes, split_axis, split_pos
    )

    # Recursively build the left and right subtrees
    left_subtree = build_bvh(left_boxes)
    right_subtree = build_bvh(right_boxes)

    # Create a new bounding box that encloses the left and right subtrees
    merged_box = merge_bounding_boxes(left_subtree, right_subtree)

    return merged_box


def find_split_position(bounding_boxes):
    # Find the optimal split position along the axis with the largest surface area
    # Return the split axis and position

    largest_surface_area = -float('inf')
    split_axis = 0
    split_pos = 0

    for axis in range(3):
        min_coord = np.min([bbox.min[axis] for bbox in bounding_boxes])
        max_coord = np.max([bbox.max[axis] for bbox in bounding_boxes])
        axis_length = max_coord - min_coord

        # Calculate the surface area for the current axis
        surface_area = axis_length * sum(
            [bbox.max[axis] - bbox.min[axis] for bbox in bounding_boxes]
        )

        if surface_area > largest_surface_area:
            largest_surface_area = surface_area
            split_axis = axis

            # Find the optimal split position along the current axis
            split_pos = (min_coord + max_coord) / 2

    return split_axis, split_pos


def partition_bounding_boxes(bounding_boxes, split_axis, split_pos):
    left_boxes = []
    right_boxes = []

    for bbox in bounding_boxes:
        if bbox.max[split_axis] < split_pos:
            left_boxes.append(bbox)
        elif bbox.min[split_axis] > split_pos:
            right_boxes.append(bbox)
        else:
            # If the bounding box intersects the split position, include it in both subtrees
            left_boxes.append(bbox)
            right_boxes.append(bbox)

    return left_boxes, right_boxes


def merge_bounding_boxes(bbox1, bbox2):
    # Merge two bounding boxes into a single box
    min_coord = np.minimum(bbox1.min, bbox2.min)
    max_coord = np.maximum(bbox1.max, bbox2.max)
    triangle_indices = bbox1.triangle_indices + bbox2.triangle_indices

    return BoundingBox(min_coord, max_coord, triangle_indices)


dummy_triangles = [
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]]),
    np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]]),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 1]]),
]

# Call the construct_bvh function with the dummy triangles array
root_node = construct_bvh(dummy_triangles)


# Print the bounding boxes and triangle indices
def print_bounding_boxes(node, depth=0):
    indent = "  " * depth
    print(f"{indent}Bounding Box: {node.min} - {node.max}")
    print(f"{indent}Triangle Indices: {node.triangle_indices}")

    if isinstance(node, BoundingBox):
        print(f"{indent}Leaf Node")
    else:
        print(f"{indent}Internal Node")
        print_bounding_boxes(node.left, depth + 1)
        print_bounding_boxes(node.right, depth + 1)


print_bounding_boxes(root_node)
