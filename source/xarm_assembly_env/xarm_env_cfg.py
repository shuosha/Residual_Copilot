import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR

from .assembly_tasks_cfg import AssemblyTask, GearMesh, NutThread, PegInsert, HF_ASSETS_REPO
from ..utils.utils import resolve_hf

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_pos_rel_held": 3,
    "fingertip_quat": 4,
    "gripper": 1,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "base_fingertip_pos": 3,
    "base_fingertip_quat": 4,
    "base_gripper": 1,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_pos_rel_held": 3,
    "fingertip_quat": 4,
    "gripper": 1,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
    "gripper_threshold": 1,
    "base_fingertip_pos": 3,
    "base_fingertip_quat": 4,
    "base_gripper": 1,
}

@configclass
class DomainRandCfg:
    # high level
    rand_ctrl = True
    aug_data = True

    # controller
    Kx_range = [190, 210]
    Kr_range = [95, 105]
    mx_range = [0.11875, 0.13125]
    mr_range = [0.01425, 0.01575]
    Kx, Kr, mx, mr = 200.0, 100.0, 0.125, 0.015

    # obs noise
    fixed_asset_pos = [0.002, 0.002, 0.002]
    held_asset_pos = [0.002, 0.002, 0.002]

    # data augmentation
    pos_xy_aug = 0.02
    pos_z_aug = 0.002
    rot_aug = 2.0

    # base action agent
    base_noise_gate_alpha = 0.8
    base_noise_range = [-0.6, 0.6]
    base_noise_prob = 0.5
    base_lag_prob = 0.8


@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_threshold = [0.03, 0.03, 0.03]
    rot_action_threshold = [0.5, 0.5, 0.5]
    gripper_action_threshold = [0.1]


@configclass
class CameraCfg:
    # Front RealSense D435 — intrinsics (848x480, fx/fy/ppx/ppy from calibration)
    H: int = 480
    W: int = 848
    pinhole_cfg = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=[
            426.7812194824219, 0.0, 425.43218994140625,
            0.0, 426.23809814453125, 245.81968688964844,
            0.0, 0.0, 1.0,
        ],
        height=480,
        width=848,
    )

    # front2base extrinsic (wxyz quaternion + xyz translation),
    # with z_offset=0.0065 applied to front2base[0,3]
    q: list = [-0.3464, 0.6371, 0.6027, -0.3330]
    t: list = [0.7263, -0.0323, 0.2216]

@configclass
class VisualizationCfg:
    frame_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05))
            }
        )
    
    verbose = False
    vis_obs = False
    print_rew = False
    order_envs = False
    store_rgb = False

@configclass
class XArmEnvCfg(DirectRLEnvCfg):
    is_finite_horizon = True
    seed = 0

    decimation: int = 8
    action_space: int = 7
    observation_space: int = 35
    state_space: int = 53

    obs_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "gripper",
        "fingertip_pos_rel_fixed",
        "fingertip_pos_rel_held",
        "ee_linvel",
        "ee_angvel",
        "base_fingertip_pos",
        "base_fingertip_quat",
        "base_gripper",
    ]

    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "gripper",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "base_fingertip_pos",
        "base_fingertip_quat",
        "base_gripper",
    ]

    pilot_model = "knn" # ["knn", "bc_teleop", "bc_expert", "replay"]
    pilot_type = "noisy" # ["noisy", "laggy", "none"]

    task_name = None  # overwrite by specific task cfg
    task: AssemblyTask = AssemblyTask()
    dmr: DomainRandCfg = DomainRandCfg()
    ctrl: CtrlCfg = CtrlCfg()

    episode_length_s: float = 20.0  # overwrite by specific task cfg
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=False)

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=resolve_hf(HF_ASSETS_REPO, task.robot_cfg.robot_file),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": -0.06,
                "joint2": -0.229,  # -45
                "joint3": 0.078,
                "joint4": 0.688,   # 30
                "joint5": 0.03,
                "joint6": 0.9,     # 75
                "joint7": -0.035,
                "gripper": 0.0,    # 0.0 to 1.7
                "left_driver_joint": 0.0,
                "left_inner_knuckle_joint": 0.0,
                "left_finger_joint": 0.0,
                "right_driver_joint": 0.0,
                "right_inner_knuckle_joint": 0.0,
                "right_finger_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-7]"],
                stiffness=200,
                damping=20,
            ),
            "xarm_hand": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                stiffness=7500.0,
                damping=173.0,
            ),
        },
    )

    # sensors
    eef_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/robot/link7",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )

    camera: CameraCfg = CameraCfg()
    front_camera_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/front_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=camera.t, rot=camera.q, convention="ros"),
        height=camera.H,
        width=camera.W,
        data_types=["rgb"],
        spawn=camera.pinhole_cfg,
    )

    vis: VisualizationCfg = VisualizationCfg()

@configclass
class XArmPegInsertCfg(XArmEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()
    episode_length_s = 20.0


@configclass
class XArmGearMeshCfg(XArmEnvCfg):
    task_name = "gear_mesh"
    task = GearMesh()
    episode_length_s = 20.0


@configclass
class XArmNutThreadCfg(XArmEnvCfg):
    task_name = "nut_thread"
    task = NutThread()
    episode_length_s = 40.0


# --- GuidedDiffusion variants (8D absolute actions) ---

@configclass
class XArmPegInsertGuidedDiffusionCfg(XArmPegInsertCfg):
    action_space: int = 8


@configclass
class XArmGearMeshGuidedDiffusionCfg(XArmGearMeshCfg):
    action_space: int = 8


@configclass
class XArmNutThreadGuidedDiffusionCfg(XArmNutThreadCfg):
    action_space: int = 8
