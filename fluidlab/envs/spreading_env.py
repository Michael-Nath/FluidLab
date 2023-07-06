from .fluid_env import FluidEnv
from fluidlab.optimizer.policies import *
from yacs.config import CfgNode
from fluidlab.fluidengine.taichi_env import TaichiEnv
from fluidlab.utils.misc import *

class SpreadingEnv(FluidEnv):
    def __init__(self, version=None, loss=True, loss_type="diff", seed=None, renderer_type="GGUI"):
        if seed is not None:
            self.seed(seed)
        self.horizon = 500
        self.horizon_action = 400
        self._n_obs_ptcls_per_body = 1000
        self.target_file = get_tgt_path("Spreading-v0.pkl")
        self.loss = loss
        self.loss_type = loss_type
        self.action_range = np.array([-0.05, 0.05])
        self.renderer_type = renderer_type
        
        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=50,
            gravity=(0.0, -20.0, 0.0),
            horizon=self.horizon
        )
        self.build_env()
        # self.gym_misc()
    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_latteart.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent
    def setup_statics(self):
        self.taichi_env.add_static(
            file='bowl.obj',
            pos=(0.63, 0.42, 0.5),
            euler=(0.5, 0.4, 0.3),
            scale=(1.2, 1.2, 1.2),
            material=BOWL,
            has_dynamics=True,
        )
    def setup_bodies(self):
        self.taichi_env.add_body(
            type='nowhere',
            n_particles=50000,
            material=WATER
        )
        self.taichi_env.add_body(
            type='cylinder',
            center=(0.5, 0.75, 0.5),
            height=0.1,
            radius=0.42,
            material=MILK,
        )
        
    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cylinder',
            xz_radius=0.42,
            xz_center=(0.5, 0.5),
            y_range=(0.5, 0.95),
        )
    def setup_renderer(self):
        self.taichi_env.setup_renderer(
            res=(960, 960),
            camera_pos=(-0.15, 2.82, 2.5),
            camera_lookat=(0.5, 0.5, 0.5),
            fov=30,
            lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                    {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
        )
    def setup_loss(self):
        pass
    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array']
        return self.taichi_env.render(mode)
    def demo_policy(self, user_input=False):
        return NoOpPolicy()