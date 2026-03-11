import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from ..utils.utils import resolve_hf

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"
HF_ASSETS_REPO = "shashuo0104/residual_copilot_assets"   # robot URDF/USD, object USD
HF_MODELS_REPO = "shashuo0104/residual_copilot_models"   # BC + RL checkpoints
HF_DATA_REPO   = "shashuo0104/residual_copilot_data"     # .npy training data

def _default_rigid_props(disable_gravity: bool = False) -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=disable_gravity,
        max_depenetration_velocity=5.0,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=3666.0,
        enable_gyroscopic_forces=True,
        solver_position_iteration_count=192,
        solver_velocity_iteration_count=1,
        max_contact_impulse=1e32,
    )


def _default_collision_props() -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)


@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class RobotCfg:
    robot_file: str = "robot/xarm7_gripper.usd"
    friction: float = 0.75

    sim_fingertip2eef = [0.0, 0.0, 0.165]
    real_fingertip2eef = [0.0, 0.0, 0.23]


@configclass
class AssemblyTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s = 5.0

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()

    success_threshold: float = 0.5

    # gripper clamp values
    gripper_obs_clamp: float = 1.0
    gripper_ctrl_clamp: float = 1.6

    # rewards
    task_success_reward_scale = 30.0
    termination_reward_scale = 50.0
    action_smoothing_reward_scale = 0.1
    xy_aligned_reward_scale = 0.05
    action_norm_reward_scale = 0.3
    tilt_penalty_reward_scale = 1.0
    force_penalty_reward_scale = 0.2


@configclass
class Peg8mm(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_peg_8mm.usd"
    diameter = 0.007986
    height = 0.050
    mass = 0.019
    grasp_offset = [0.0, 0.0, 0.02]


@configclass
class Hole8mm(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_hole_8mm.usd"
    diameter = 0.0081
    height = 0.025
    base_height = 0.0


@configclass
class PegInsert(AssemblyTask):
    # data
    train_data_path: str = "teleop/peginsert_train_data.npy"

    # checkpoints
    dp_teleop_path: str = "shared_autonomy_policies/bc_teleop/PegInsert_bc_teleop"
    dp_expert_path: str = "shared_autonomy_policies/bc_expert/PegInsert_bc_expert"

    name = "peg_insert"
    fixed_asset_cfg = Hole8mm()
    held_asset_cfg = Peg8mm()
    duration_s = 20.0
    close_gripper: float = 0.93
    success_threshold: float = 0.04
    gripper_obs_clamp: float = 0.94

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    held_asset_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/forge_round_peg_8mm",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )


@configclass
class GearBase(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_gear_base.usd"
    height = 0.02
    base_height = 0.005

    small_gear_base_offset = [5.075e-2, 0.0, 0.0]
    medium_gear_base_offset = [2.025e-2, 0.0, 0.0]
    large_gear_base_offset = [-3.025e-2, 0.0, 0.0]


@configclass
class MediumGear(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_gear_medium.usd"
    diameter = 0.03  # Used for gripper width.
    height: float = 0.03
    mass = 0.012
    grasp_offset = [0.0, 0.0, 0.025]


@configclass
class GearMesh(AssemblyTask):
    # data
    train_data_path: str = "teleop/gearmesh_train_data.npy"

    # checkpoints
    dp_teleop_path: str = "shared_autonomy_policies/bc_teleop/GearMesh_bc_teleop"
    dp_expert_path: str = "shared_autonomy_policies/bc_expert/GearMesh_bc_expert"

    name = "gear_mesh"
    fixed_asset_cfg = GearBase()
    held_asset_cfg = MediumGear()
    duration_s = 20.0
    close_gripper: float = 0.69
    success_threshold: float = 0.05
    gripper_obs_clamp: float = 0.695
    gripper_ctrl_clamp: float = 1.2

    small_gear_usd = f"{ASSET_DIR}/factory_gear_small.usd"
    large_gear_usd = f"{ASSET_DIR}/factory_gear_large.usd"

    small_gear_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/SmallGearAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=small_gear_usd,
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    large_gear_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/LargeGearAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=large_gear_usd,
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    held_asset_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/factory_gear_medium",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )


@configclass
class NutM32(HeldAssetCfg):
    usd_hf_file = "objects/m36_nut.usd"
    diameter = 0.024
    height = 0.01
    mass = 0.03
    friction = 0.01  # Additive with the nut means friction is (-0.25 + 0.75)/2 = 0.25

@configclass
class BoltM32(FixedAssetCfg):
    usd_hf_file = "objects/m36_bolt.usd"
    diameter = 0.024
    height = 0.025
    base_height = 0.01
    thread_pitch = 0.002
    nut_offset = [0.0, 0.0, 0.04]

@configclass
class NutThread(AssemblyTask):
    name = "nut_thread"

    train_data_path: str = "teleop/nutthread_train_data.npy"
    dp_teleop_path: str = "shared_autonomy_policies/bc_teleop/NutThread_bc_teleop"
    dp_expert_path: str = "shared_autonomy_policies/bc_expert/NutThread_bc_expert"

    fixed_asset_cfg = BoltM32()
    held_asset_cfg = NutM32()
    asset_size = 16.0
    duration_s = 40.0
    close_gripper: float = 0.39
    success_threshold: float = 0.375
    success_rotation_threshold_deg: float = 90.0
    gripper_obs_clamp: float = 0.40
    gripper_ctrl_clamp: float = 0.75

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=resolve_hf(HF_ASSETS_REPO, fixed_asset_cfg.usd_hf_file),
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=resolve_hf(HF_ASSETS_REPO, held_asset_cfg.usd_hf_file),
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    held_asset_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/m36_nut",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )
