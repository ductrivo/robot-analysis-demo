import csv
from ast import literal_eval
from dataclasses import dataclass, field
from pathlib import Path

import modern_robotics as mr
import numpy as np
from numpy.typing import NDArray

np.set_printoptions(precision=3, suppress=True)

m1 = 1
l1 = 1
ixx = m1 * l1 / 3


def compute_screw_axes(w, p):
    Slist = []
    for i in range(len(w)):
        w_ = w[i][:, 0]
        p_ = p[i][:, 0]
        v_ = -np.cross(w_, p_)
        Slist.append([w_[0], w_[1], w_[2], v_[0], v_[1], v_[2]])
    return np.transpose(Slist)


def decompose_transform(transform: NDArray) -> tuple[NDArray, NDArray]:
    """
    Decompose a 4x4 homogeneous transformation matrix into rotation (3x3) and translation (3x1).
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3].reshape(3, 1)
    return rotation, translation


def compose_transform(rotation: NDArray, translation: NDArray) -> NDArray:
    """
    Compose a 4x4 homogeneous transformation matrix from rotation (3x3) and translation (3x1).
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    return transform


def rpy_to_matrix(rpy: NDArray) -> NDArray:
    """
    Convert a roll-pitch-yaw vector (shape: (3, 1)) to a 3x3 rotation matrix.
    Rotation order is extrinsic ZYX: yaw, pitch, then roll.
    """
    roll, pitch, yaw = rpy.reshape(-1)

    c_roll, c_pitch, c_yaw = np.cos([roll, pitch, yaw])
    s_roll, s_pitch, s_yaw = np.sin([roll, pitch, yaw])

    return np.array(
        [
            [
                c_yaw * c_pitch,
                c_yaw * s_pitch * s_roll - s_yaw * c_roll,
                c_yaw * s_pitch * c_roll + s_yaw * s_roll,
            ],
            [
                s_yaw * c_pitch,
                s_yaw * s_pitch * s_roll + c_yaw * c_roll,
                s_yaw * s_pitch * c_roll - c_yaw * s_roll,
            ],
            [-s_pitch, c_pitch * s_roll, c_pitch * c_roll],
        ]
    )


def origin_to_transform(rpy: NDArray, xyz: NDArray) -> NDArray:
    """
    Create a 4x4 homogeneous transformation matrix from rpy and xyz.
    Both rpy and xyz must be numpy arrays of shape (3, 1).
    """
    rotation = rpy_to_matrix(rpy)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz.reshape(-1)
    return transform


@dataclass
class Origin:
    rpy: NDArray
    xyz: NDArray


@dataclass
class Inertia:
    mass: float
    origin: Origin
    matrix: NDArray


@dataclass
class Link:
    name: str
    inertia: Inertia


@dataclass
class Joint:
    name: str
    parent: Link
    child: Link
    origin: Origin
    axis: NDArray
    type: str = 'fixed'
    effort: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    velocity: float = 0.0
    path: str = ''
    mat_m0: NDArray = field(default_factory=lambda: np.eye(4))


# --- Load Links ---
def load_links_from_csv(file_path: Path) -> dict[str, Link]:
    links = {}
    with file_path.open(newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['name']
            mass = float(row['mass'])
            rpy = np.array(literal_eval(row['rpy']), dtype=float)
            xyz = np.array(literal_eval(row['xyz']), dtype=float)

            # inertia
            ixx = float(row['ixx'])
            ixy = float(row['ixy'])
            ixz = float(row['ixz'])
            iyy = float(row['iyy'])
            iyz = float(row['iyz'])
            izz = float(row['izz'])
            inertia_matrix = np.array(
                [
                    [ixx, ixy, ixz],
                    [ixy, iyy, iyz],
                    [ixz, iyz, izz],
                ],
                dtype=float,
            )

            origin = Origin(rpy=rpy, xyz=xyz)
            inertia = Inertia(mass=mass, origin=origin, matrix=inertia_matrix)

            links[name] = Link(name=name, inertia=inertia)
    return links


# --- Load Joints ---
def load_joints_from_csv(
    file_path: Path, links: dict[str, Link]
) -> dict[str, Joint]:
    joints = {}
    with file_path.open(newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['name']
            parent = links[row['parent']]
            child = links[row['child']]
            rpy = np.array(literal_eval(row['rpy']), dtype=float).reshape(3, 1)
            xyz = np.array(literal_eval(row['xyz']), dtype=float).reshape(3, 1)
            axis = np.array(literal_eval(row['axis']), dtype=float).reshape(
                3, 1
            )
            joint_type = row.get('type', 'fixed')
            effort = float(row.get('effort', 0.0))
            lower = float(row.get('lower', 0.0))
            upper = float(row.get('upper', 0.0))
            velocity = float(row.get('velocity', 0.0))

            origin = Origin(rpy=rpy, xyz=xyz)
            joint = Joint(
                name=name,
                parent=parent,
                child=child,
                origin=origin,
                axis=axis,
                type=joint_type,
                effort=effort,
                lower=lower,
                upper=upper,
                velocity=velocity,
            )
            joints[name] = joint

    return joints


def homo_multiply(mat_t, pose):
    return mat_t @ np.vstack([pose, np.array([[1]])])


class Robot:
    def __init__(self) -> None:
        self.joints: dict[str, Joint] = {}
        self.links: dict[str, Link] = {}
        self.tree: dict[str, list[tuple[Joint, Link]]] = {}

    def forward_kinematic(self):
        new_pose = mr.FKinSpace(
            M=self.mat_m,
            Slist=self.Slist,
            thetalist=np.array([np.pi / 6]),
        )
        print(new_pose)

    def inverted_kinematic(self):
        pass

    def _build_kinematic_tree(self) -> dict[str, list[tuple[Joint, Link]]]:
        tree: dict[str, list[tuple[Joint, Link]]] = {}
        for joint in self.joints.values():
            parent_name = joint.parent.name
            if parent_name not in tree:
                tree[parent_name] = []
            tree[parent_name].append((joint, joint.child))
        return tree

    def create_from_csv(self, info_path: Path) -> None:
        self.links = load_links_from_csv(info_path / 'links_info.csv')
        self.joints = load_joints_from_csv(
            info_path / 'joints_info.csv', self.links
        )
        self.tree = self._build_kinematic_tree()
        self.traverse()
        self._init_matrices()

    def traverse(
        self,
        current_link_name: str = 'world',
        depth: int = 0,
        current_path: str = 'world',
        visited: set[str] | None = None,
    ) -> None:
        if visited is None:
            visited = set()

        # Prevent visiting the same link more than once
        if current_link_name in visited:
            print(f'{"  " * depth}Link: {current_link_name} (already visited)')
            return
        visited.add(current_link_name)

        indent = '  ' * depth
        print(f'{indent}Link: {current_link_name}')

        for joint, child_link in self.tree.get(current_link_name, []):
            # Prevent self-recursion or cycles
            if child_link.name == current_link_name:
                print(f'{indent}  └─ Joint: {joint.name} (skipped self-loop)')
                continue

            joint.path = current_path + '/' + child_link.name
            print(f'{indent}  └─ Joint: {joint.name} (path={joint.path})')

            # Recursively traverse children
            self.traverse(
                current_link_name=child_link.name,
                depth=depth + 1,
                current_path=joint.path,
                visited=visited,
            )

    def _init_matrices(self) -> None:
        mat_m0i = np.eye(4)  # zero config of link in wrt 0

        g_list: list[NDArray] = []  # Spatial inertia matrices
        p_list: list[NDArray] = []
        w_list: list[NDArray] = []
        frames_list = [np.eye(4)]

        for joint_name, joint in self.joints.items():
            child_link = joint.child

            if child_link.name not in ['world', 'ee_link']:
                # zero config of i wrt i-1: M(i-1,i)
                mat_mi1_i = origin_to_transform(
                    rpy=joint.origin.rpy,
                    xyz=joint.origin.xyz,
                )
                mat_m0i @= mat_mi1_i

                # Set to joint
                joint.mat_m0 = mat_m0i.copy()

                rot_0i, trans_0i = decompose_transform(mat_m0i)

                w = np.array(rot_0i @ np.array(joint.axis))

                p_list.append(trans_0i.copy())
                w_list.append(w.copy())

                g = np.eye(6)
                g[0:3, 0:3] = child_link.inertia.matrix
                g[3:6, 3:6] = child_link.inertia.mass * np.eye(3)
                g_list.append(g)

                child_inertia_origin = origin_to_transform(
                    rpy=child_link.inertia.origin.rpy,
                    xyz=child_link.inertia.origin.xyz,
                )
                CoM_M = mat_m0i @ child_inertia_origin
                frames_list.append(CoM_M)

        joint = self.joints['ee_joint']
        transform = origin_to_transform(
            rpy=joint.origin.rpy,
            xyz=joint.origin.xyz,
        )
        mat_m0i @= transform
        joint.mat_m0 = mat_m0i.copy()
        frames_list.append(mat_m0i.copy())

        screw_arr = compute_screw_axes(w_list, p_list)
        b_arr = mr.Adjoint(mr.TransInv(mat_m0i)) @ screw_arr

        frames_rel_list = []
        for i in range(1, len(frames_list)):
            frame = mr.TransInv(frames_list[i - 1]) @ frames_list[i]
            frames_rel_list.append(frame)

        frames_arr = np.array(frames_rel_list)
        g_arr = np.array(g_list)

        # input(f'mat_m0i =\n{mat_m0i}')
        # input(f'Slist =\n{screw_arr}')
        # input(f'Mlist2 =\n{frames_arr}')
        # input(f'Glist =\n{g_arr}')
        # input(f'Blist =\n{b_arr}')
        self.mat_m = mat_m0i
        self.Slist = screw_arr
        self.Mlist = frames_arr
        self.Glist = g_arr
        self.Blist = b_arr


if __name__ == '__main__':
    # print(f'ixx = {ixx}')

    info_path = Path(__file__).parent.parent.parent / 'examples/r_planar'

    robot = Robot()
    robot.create_from_csv(info_path)
    robot.forward_kinematic()
    # data = {
    #     'world_joint': {
    #         'rpy': np.array([[0.0], [0.0], [0.0]]),
    #         'xyz': np.array([[0.0], [0.0], [0.0]]),
    #     },
    #     'joint1': {
    #         'rpy': np.array([[0.0], [0.0], [0.0]]),
    #         'xyz': np.array([[0.0], [0.0], [0.089159]]),
    #     },
    #     'joint2': {
    #         'rpy': np.array([[0.0], [1.570796325], [0.0]]),
    #         'xyz': np.array([[0.0], [0.13585], [0.0]]),
    #     },
    #     'joint3': {
    #         'rpy': np.array([[0.0], [0.0], [0.0]]),
    #         'xyz': np.array([[0.0], [-0.1197], [0.425]]),
    #     },
    #     'joint4': {
    #         'rpy': np.array([[0.0], [1.570796325], [0.0]]),
    #         'xyz': np.array([[0.0], [0.0], [0.39225]]),
    #     },
    #     'joint5': {
    #         'rpy': np.array([[0.0], [0.0], [0.0]]),
    #         'xyz': np.array([[0.0], [0.093], [0.0]]),
    #     },
    #     'joint6': {
    #         'rpy': np.array([[0.0], [0.0], [0.0]]),
    #         'xyz': np.array([[0.0], [0.0], [0.09465]]),
    #     },
    #     'ee_joint': {
    #         'rpy': np.array([[-1.570796325], [0.0], [0.0]]),
    #         'xyz': np.array([[0.0], [0.0823], [0.0]]),
    #     },
    # }

    # transform_space = np.eye(4)
    # for name, origin in data.items():
    #     transform = origin_to_transform(origin['rpy'], origin['xyz'])
    #     transform_space @= transform
    #     print(f'name = {name}\n{transform_space}')
