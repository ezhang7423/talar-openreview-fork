import numpy as np
import math
from collections.abc import Mapping, Sequence


def compute_relationship(scene_struct, use_polar=False, eps=0.6, max_dist=0.7):
    """Compute pariwise relationship between objects."""
    all_relationships = {}
    max_dist_sq = max_dist**2
    for name, direction_vec in scene_struct["directions"].items():
        if name == "above" or name == "below":
            continue
        all_relationships[name] = []
        for _, obj1 in enumerate(scene_struct["objects"]):
            coords1 = obj1["3d_coords"]
            related = set()
            for j, obj2 in enumerate(scene_struct["objects"]):
                if obj1 == obj2:
                    continue
                coords2 = obj2["3d_coords"]
                diff = np.array([coords2[k] - coords1[k] for k in [0, 1, 2]])
                norm = np.linalg.norm(diff)
                diff /= norm
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if use_polar:
                    if dot > 0.71:
                        th = math.sqrt(max_dist_sq * (2.0 * dot**2 - 1.0))
                        qualified = norm < th
                    else:
                        qualified = False
                else:
                    qualified = dot > eps and norm < max_dist
                if qualified:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


def isclose_recursive(a, b, rel_tol=1e-9, abs_tol=0.0, path=None):
    # Begin the recursive path tracking with an empty list if it's the first call.
    if path is None:
        path = []

    # If both are dictionaries, recurse into each key.
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for key in a:
            if key not in b:
                print(f"Key {key} missing in second dict. Path: {'/'.join(path)}")
                return False
            # Build the new path for recursive call
            new_path = path + [str(key)]
            if not isclose_recursive(a[key], b[key], rel_tol, abs_tol, new_path):
                return False
        for key in b:
            if key not in a:
                print(f"Key {key} missing in first dict. Path: {'/'.join(path)}")
                return False
        return True

    # If both are sequences (not strings), compare element-wise.
    elif (
        isinstance(a, Sequence)
        and isinstance(b, Sequence)
        and not isinstance(a, str)
        and not isinstance(b, str)
    ):
        if len(a) != len(b):
            print(f"Sequences have different lengths. Path: {'/'.join(path)}")
            return False
        for idx, (x, y) in enumerate(zip(a, b)):
            # Build the new path for recursive call
            new_path = path + [str(idx)]
            if not isclose_recursive(x, y, rel_tol, abs_tol, new_path):
                return False
        return True

    # Use math.isclose for float number comparison.
    elif isinstance(a, float) and isinstance(b, float):
        if not math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
            # Print the differing floats.
            print(f"Floats not close: {a} vs {b} (Path: {'/'.join(path)})")
            return False
        return True
    # Use np.iscloe for numpy array comparison.
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if not np.allclose(a, b, rtol=rel_tol, atol=abs_tol):
            # Print the differing arrays.
            print(f"Arrays not close: {a} vs {b} (Path: {'/'.join(path)})")
            return False
        return True
    else:
        # Check for equality of other types.
        if a != b:
            # Print the differing values.
            print(f"Different values: {a} vs {b} (Path: {'/'.join(path)})")
            return False
        return True


# Assume mappings for non-numeric data for serialization.
size_mapping = {"large": 1}
color_val_mapping = {
    "red": "1 0.1 0.1 1",
    "blue": "0.2 0.5 1 1",
    "green": "0.2 1 0 1",
    "purple": "0.8 0.2 1 1",
    "cyan": "0.2 1 1 1",
}
color_mapping = {color: i for i, color in enumerate(color_val_mapping.keys(), start=1)}
material_mapping = {"rubber": 1}
max_relationships = 5


def serialize(scene_struct):
    # Calculate the total length of the serialized array in advance for preallocation.
    total_length = (
        len(scene_struct["objects"]) * 10
    )  # Object properties (fixed length per object)
    total_length += len(scene_struct["directions"]) * 3  # Direction vectors
    total_length += 5
    # total_length += max_relationships * 4 * len(scene_struct['objects'])  # Relationships (padded to max size)

    serialized = np.full(
        total_length, -1, dtype=np.float64
    )  # Use float64 for precision
    index = 0

    # Serialize objects
    for obj in scene_struct["objects"]:
        serialized[index] = size_mapping[obj["size"]]
        index += 1
        serialized[index : index + 3] = obj["3d_coords"]
        index += 3
        color_vals = np.fromstring(color_val_mapping[obj["color"]], sep=" ")
        serialized[index : index + 4] = color_vals
        index += 4
        serialized[index] = color_mapping[obj["color"]]
        index += 1
        serialized[index] = obj["rotation"]
        index += 1
        serialized[index] = material_mapping[obj["material"]]
        index += 1

    # Serialize directions
    for direction in ["behind", "front", "left", "right", "above", "below"]:
        serialized[index : index + 3] = np.array(scene_struct["directions"][direction])
        index += 3

    # # Serialize relationships
    # for relation in ['behind', 'front', 'left', 'right']:
    #     for rel_list in data['relationships'][relation]:
    #         # breakpoint()
    #         serialized[index:index + len(rel_list)] = rel_list
    #         index += max_relationships

    return serialized


def array_to_str(arr):
    if isinstance(arr, np.ndarray):
        # Format each number to remove the '.0' if present, and convert to list
        str_list = [f"{num:g}" for num in arr.tolist()]
        return " ".join(str_list)
    else:
        raise TypeError("The input must be a numpy.ndarray")


def deserialize(serialized):
    num_objects = 5  # Known number of objects
    obj_properties = 10  # Number of properties per object
    direction_size = 3  # Number of elements per direction vector
    num_directions = 6  # Number of directions
    index = 0

    data = {"objects": [], "directions": {}, "relationships": {}, "split": "none"}

    # Deserialize objects
    reverse_size_mapping = {v: k for k, v in size_mapping.items()}
    reverse_color_mapping = {v: k for k, v in color_mapping.items()}
    reverse_material_mapping = {v: k for k, v in material_mapping.items()}
    for _ in range(num_objects):
        size = reverse_size_mapping[serialized[index]]
        index += 1
        coords = tuple(serialized[index : index + 3])
        index += 3
        color_val = serialized[index : index + 4]
        index += 4
        color = reverse_color_mapping[int(serialized[index])]
        index += 1
        rotation = serialized[index]
        index += 1
        material = reverse_material_mapping[int(serialized[index])]
        index += 1
        data["objects"].append(
            {
                "shape": "sphere",
                "shape_name": "sphere",
                "size": size,
                "3d_coords": coords,
                "color_val": array_to_str(color_val),
                "color": color,
                "rotation": rotation,
                "material": material,
            }
        )

    # Deserialize directions
    directions = ["behind", "front", "left", "right", "above", "below"]
    for direction in directions:
        data["directions"][direction] = serialized[index : index + direction_size]
        index += direction_size

    # Deserialize relationships
    # Deserialize relationships
    data["relationships"] = compute_relationship(data)
    # for relation in ['behind', 'front', 'left', 'right']:
    #     relation_data = []
    #     for _ in range(num_objects):
    #         relationship = serialized[index:index + max_relationships]
    #         index += max_relationships
    #         relationship = relationship[relationship != -1].astype(int).tolist()  # Filter out the padding
    #         relation_data.append(relationship)
    #     data['relationships'][relation] = relation_data

    return data



if __name__ == '__main__':
    # Example usage:
    """
    scene_struct:
    {
        'split': 'none',
        'objects': [
            {
                'shape': 'sphere',
                'shape_name': 'sphere',
                'size': 'large',
                '3d_coords': (0.26290870641572883, -0.001640871932608312, 0.13),
                'color_val': '1 0.1 0.1 1',
                'color': 'red',
                'rotation': 38.73904468099194,
                'material': 'rubber'
            },
            {
                'shape': 'sphere',
                'shape_name': 'sphere',
                'size': 'large',
                '3d_coords': (-0.448655011414997, -0.26983866887527436, 0.13),
                'color_val': '0.2 0.5 1 1',
                'color': 'blue',
                'rotation': 23.070800789701686,
                'material': 'rubber'
            },
            {
                'shape': 'sphere',
                'shape_name': 'sphere',
                'size': 'large',
                '3d_coords': (-0.4091476020538839, 0.18011348076839512, 0.13),
                'color_val': '0.2 1 0 1',
                'color': 'green',
                'rotation': 217.17120492567113,
                'material': 'rubber'
            },
            {
                'shape': 'sphere',
                'shape_name': 'sphere',
                'size': 'large',
                '3d_coords': (-0.09643002092177633, 0.4169559367365248, 0.13),
                'color_val': '0.8 0.2 1 1',
                'color': 'purple',
                'rotation': 313.078879469633,
                'material': 'rubber'
            },
            {
                'shape': 'sphere',
                'shape_name': 'sphere',
                'size': 'large',
                '3d_coords': (0.33415201390238713, 0.4325433802395508, 0.13),
                'color_val': '0.2 1 1 1',
                'color': 'cyan',
                'rotation': 298.62000870537224,
                'material': 'rubber'
            }
        ],
        'directions': {
            'behind': array([-4.32978028e-17, -1.00000000e+00,  0.00000000e+00]),
            'front': array([ 4.32978028e-17,  1.00000000e+00, -0.00000000e+00]),
            'left': array([-1.00000000e+00,  8.65956056e-17,  0.00000000e+00]),
            'right': array([ 1.00000000e+00, -8.65956056e-17, -0.00000000e+00]),
            'above': array([0., 0., 1.]),
            'below': array([-0., -0., -1.])
        },
        'relationships': {
            'behind': [[], [], [1], [0, 2], [0]],
            'front': [[3, 4], [2], [3], [], []],
            'left': [[2, 3], [], [], [2], [3]],
            'right': [[], [], [0, 3], [0, 4], []]
        }
    }
    """
    from lcd.apps.train_gcbc import setup_env
    from rich import print
    env = setup_env()
    e = env.envs[0]    
    print(deserialize(serialize(e.scene_struct)))
    print(e.scene_struct)
    for i in range(1000):
        print(i)
        e.reset()
        assert (isclose_recursive(deserialize(serialize(e.scene_struct)), e.scene_struct))
