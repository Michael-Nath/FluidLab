import numpy as np
from torch import load, cat, Tensor, no_grad
from torch.cuda import is_available
from fluidlab.models.gc_bc import GCBCAgent, GCBCVAEAgent
from fluidlab.optimizer.optim import *
from fluidlab.utils.misc import is_on_server

if is_on_server():
    try:
        from pynput import keyboard, mouse
    except:
        pass

class ActionsPolicy:
    def __init__(self, comp_actions):
        self.actions_v = comp_actions[:-1]
        self.actions_p = comp_actions[-1]

    def get_actions_p(self):
        return self.actions_p

    def get_action_v(self, i):
        return self.actions_v[i]

class NoOpPolicy:
    def __init__(self) -> None:
        pass
    def get_actions_p(self):
        return None
    def get_action_v(self, _):
        return np.random.randint(-100, 100, 3) / 100
        # return None

class KeyboardPolicy:
    def __init__(self, init_p, v_lin=0.003, v_ang=0.03):
        self.actions_p = init_p
        self.keys_activated = set()
        self.linear_v_mag = v_lin
        self.angular_v_mag = v_ang

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            self.keys_activated.add(key.char)
        except:
            pass

    def on_release(self, key):
        try:
            self.keys_activated.remove(key.char)
        except:
            pass

    def get_actions_p(self):
        return self.actions_p


class KeyboardPolicy_vxy_wz(KeyboardPolicy):
    def get_action_v(self, i):
        action_v = np.zeros(6)
        if '4' in self.keys_activated:
            action_v[0] -= self.linear_v_mag
        if '6' in self.keys_activated:
            action_v[0] += self.linear_v_mag
        if '2' in self.keys_activated:
            action_v[1] -= self.linear_v_mag
        if '8' in self.keys_activated:
            action_v[1] += self.linear_v_mag
        if 'x' in self.keys_activated:
            action_v[5] -= self.angular_v_mag
        if 'z' in self.keys_activated:
            action_v[5] += self.angular_v_mag
        return action_v

class KeyboardPolicy_wz(KeyboardPolicy):
    def get_action_v(self, i):
        action_v = np.zeros(6)
        if 'x' in self.keys_activated:
            action_v[5] -= self.angular_v_mag
        if 'z' in self.keys_activated:
            action_v[5] += self.angular_v_mag
        return action_v
class DemoPouringPolicy:
    def __init__(self, p_init, v_lin=0.003, v_ang=0.01):
        self.linear_v_mag = v_lin
        self.angular_v_mag = v_ang
        self.actions_p = p_init
        self.direction = -1
    def get_actions_p(self):
        return self.actions_p
    def get_action_v(self, i):
        action_v = np.zeros(6)
        if i % 150 == 0:
            self.direction *= -1
        action_v[5] += self.direction * self.angular_v_mag
        return action_v

class KeyboardPolicy_vxy(KeyboardPolicy):
    def get_action_v(self, i):
        action_v = np.zeros(3)
        if '4' in self.keys_activated:
            action_v[0] -= self.linear_v_mag
        if '6' in self.keys_activated:
            action_v[0] += self.linear_v_mag
        if '2' in self.keys_activated:
            action_v[1] -= self.linear_v_mag
        if '8' in self.keys_activated:
            action_v[1] += self.linear_v_mag
        return action_v


class MousePolicy:
    def __init__(self, init_p):
        self.actions_p      = init_p
        self.mouse_pos      = np.array([0, 0])
        self.mouse_pos_last = None
        self.mouse_pressed  = False
        self.started        = False
        from pynput import keyboard, mouse
        self.listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
        )
        self.listener.start()

    def on_move(self, x, y):
        self.started      = True
        self.mouse_pos[0] = x
        self.mouse_pos[1] = y

    def on_click(self, x, y, button, pressed):
        self.mouse_pos[0]   = x
        self.mouse_pos[1]   = y
        self.mouse_pressed  = pressed

    def get_actions_p(self):
        return self.actions_p

class MousePolicy_vxz(MousePolicy):
    def get_action_v(self, i):
        if not self.started:
            action_v = np.zeros(3)
        else:
            if self.mouse_pos_last is None:
                self.mouse_pos_last = np.array(self.mouse_pos)
            
            mouse_pos_diff = self.mouse_pos - self.mouse_pos_last
            self.mouse_pos_last = np.array(self.mouse_pos)
            action_v = np.array([mouse_pos_diff[0], 0.0, mouse_pos_diff[1]]) * 5e-4

        return action_v

class RandomGaussianPolicy:
    def __init__(self, init_range, action_dim, horizon, fix_dim=None):
        self.horizon = horizon
        self.action_dim = action_dim
        # Let the mean just be the average of init_range[0] and init_range[1] for both v and p.
        mean_v = (init_range.v[0][0] + init_range.v[0][1]) / 2
        mean_p = (init_range.p[0][0] + init_range.p[0][1]) / 2
        self.actions_v = np.random.normal(mean_v, 0.1, size=(horizon, action_dim))
        self.actions_p = np.random.normal(mean_p, 0.1, size=(action_dim))

    def get_action_v(self, i, **kwargs):
        assert 0 <= i < self.horizon
        return self.actions_v[i]

    def get_actions_p(self):
        return self.actions_p
    def get_action(self, **kwargs):
        i = kwargs["i"]
        return self.actions_v[i]

class CorrelatedNoisePolicy:
    def __init__(self, action_dim, horizon, beta = 0.85, scale = 0.10):
        self.horizon = horizon
        self.action_dim = action_dim
        self.beta = beta
        self.cov = np.eye(action_dim) * scale 
        self.n_t = 0
    def get_action(self, **kwargs):
        # Setup normal
        u_t = np.random.multivariate_normal(np.full(self.action_dim, 0.0), self.cov)
        # Calculate noise term
        self.n_t = (1 - self.beta) * u_t + self.beta * self.n_t
        # This noise represents our action because we are working with random action sequences.
        # However, we want our sampled actions to be smoothly correlated.
        return self.n_t
    # def get_actions_p(self, **kwargs):
    #     # For now, do not distinguish between these two kinds of actions
    #     return self.get_action()

class LoadedGCBCPolicy:
    def __init__(self, action_dim, goal_img_obs, weights_file, agent_type):
        if agent_type not in ["gcbc", "gcbc_vae"]:
            raise ValueError
        if agent_type == "gcbc":
            self.agent = GCBCAgent(action_dim)
        else:
            self.agent = GCBCVAEAgent(action_dim)
        self.agent.load_state_dict(load(weights_file))
        self.agent.eval()
        self.device = "cuda" if is_available() else "cpu"
        self.agent = self.agent.to(self.device)
        self.goal_img_obs = Tensor(goal_img_obs)
        if (self.goal_img_obs.size()[0] != 3):
            self.goal_img_obs = self.goal_img_obs.movedim(2, 0)
    def get_action(self, **kwargs):
        cur_img_obs = kwargs["cur_img_obs"]
        with no_grad():
            cur_img_obs = Tensor(cur_img_obs)
            if (cur_img_obs.size()[0] != 3):
                cur_img_obs = cur_img_obs.movedim(2, 0)
            dist = self.agent.forward(cur_img_obs, self.goal_img_obs)
            a = dist.sample()
            return a

class TrainablePolicy:
    def __init__(self, optim_cfg, init_range, action_dim, horizon, action_range, fix_dim=None):
        self.horizon = horizon
        self.action_dim = action_dim
        self.actions_v = np.random.uniform(init_range.v[0], init_range.v[1], size=(horizon, action_dim))
        self.actions_p = np.random.uniform(init_range.p[0], init_range.p[1], size=(action_dim))
        self.action_range = action_range
        self.comp_actions_shape = (horizon+1, action_dim)
        self.trainable = np.full(self.comp_actions_shape[0], True)
        self.fix_dim = fix_dim
        self.freeze_till = 0

        self.optim = eval(optim_cfg.type)(self.comp_actions_shape, optim_cfg)

    @property
    def comp_actions(self):
        return np.vstack([self.actions_v, self.actions_p[None, :]])

    def get_actions_p(self):
        return self.actions_p

    def get_action_v(self, i, **kwargs):
        return self.actions_v[i]

    def optimize(self, grads, loss_info):
        assert grads.shape == self.comp_actions_shape

        grads[np.logical_not(self.trainable)] = 0
        if self.fix_dim is not None:
            grads[:, self.fix_dim] = 0

        new_comp_actions = self.optim.step(self.comp_actions, grads)
        self.actions_p = new_comp_actions[-1]
        self.actions_v = new_comp_actions[:-1].clip(*self.action_range)


class LatteArtPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(LatteArtPolicy, self).__init__(*args, **kwargs)


class LatteArtStirPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(LatteArtStirPolicy, self).__init__(*args, **kwargs)

    def optimize(self, grads, loss_info):
        super(LatteArtStirPolicy, self).optimize(grads, loss_info)

        # task specific processing
        if loss_info['temporal_range'] > 250:
            self.optim.lr = self.optim.init_lr * 0.2
            print(f'lr reduced to {self.optim.lr}')
        elif loss_info['temporal_range'] > 150:
            self.optim.lr = self.optim.init_lr * 0.5
            print(f'lr reduced to {self.optim.lr}')

        for step in [400, 350, 300, 250, 200, 150, 100]:
            if loss_info['temporal_range'] > step:
                freeze_till = step - 100
                self.trainable[:freeze_till] = False
                print(f'feeze till {freeze_till}')
                break


class IceCreamDynamicPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(IceCreamDynamicPolicy, self).__init__(*args, **kwargs)

        self.trainable = np.full(self.comp_actions_shape[0], False)
        self.trainable[169:-1] = True


class IceCreamStaticPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(IceCreamStaticPolicy, self).__init__(*args, **kwargs)
        self.trainable = np.full(self.comp_actions_shape[0], False)
        self.trainable[:-1] = True

    def optimize(self, grads, loss_info):
        grads = grads.clip(-1e5, 1e5)
        super(IceCreamStaticPolicy, self).optimize(grads, loss_info)

        if loss_info['temporal_range'] > 450:
            self.optim.lr = self.optim.init_lr * 0.1
            print(f'lr reduced to {self.optim.lr}')


class GatheringPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(GatheringPolicy, self).__init__(*args, **kwargs)

        self.trainable = np.full(self.comp_actions_shape[0], False)
        self.status = np.full(self.comp_actions_shape[0], 0)
        self.stage_step = [50, 65, 105, 120]
        for i in range(self.horizon):
            if i % self.stage_step[3] < self.stage_step[0]:
                self.trainable[i] = True
                self.status[i] = 0 # moving
            elif self.stage_step[0] <= i % self.stage_step[3] < self.stage_step[1]:
                self.status[i] = 1 # up
            elif self.stage_step[1] <= i % self.stage_step[3] < self.stage_step[2]:
                self.status[i] = 2 # moving back
            elif self.stage_step[2] <= i % self.stage_step[3]:
                self.status[i] = 3 # down

    def get_action_v(self, i, agent=None, update=False):
        if update:
            if self.status[i] == 1:
                self.actions_v[i] = np.array([0, 0.008, 0])
            elif self.status[i] == 2:
                action = (self.actions_p - agent.rigid.latest_pos.to_numpy()[0]) / (self.stage_step[2]- (i % self.stage_step[3]))
                action[1] = 0
                self.actions_v[i] = action
            elif self.status[i] == 3:
                self.actions_v[i] = np.array([0, -0.008, 0])
            # elif self.status[i] == 0:
            #     self.actions_v[i] = np.array([0.003, 0, 0])

        return self.actions_v[i]

    def optimize(self, grads, loss_info):
        for step in [720, 600, 480, 360, 240, 120]:
            if loss_info['temporal_range'] > step:
                self.freeze_till = loss_info['temporal_range'] - 120
                self.trainable[:self.freeze_till] = False
                print(f'feeze till {self.freeze_till}')
                break

        super(GatheringPolicy, self).optimize(grads, loss_info)


class GatheringOPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(GatheringOPolicy, self).__init__(*args, **kwargs)

        self.trainable = np.full(self.comp_actions_shape[0], False)
        self.status = np.full(self.comp_actions_shape[0], 0)
        self.stage_step = [50, 65, 105, 120]
        for i in range(self.horizon):
            if i % self.stage_step[3] < self.stage_step[0]:
                self.trainable[i] = True
                self.status[i] = 0 # moving
            elif self.stage_step[0] <= i % self.stage_step[3] < self.stage_step[1]:
                self.status[i] = 1 # up
            elif self.stage_step[1] <= i % self.stage_step[3] < self.stage_step[2]:
                self.status[i] = 2 # moving back
            elif self.stage_step[2] <= i % self.stage_step[3]:
                self.status[i] = 3 # down

    def get_action_v(self, i, agent=None, update=False):
        if update:
            if self.status[i] == 1:
                self.actions_v[i] = np.array([0, 0.008, 0])
            elif self.status[i] == 2:
                action = (self.actions_p - agent.rigid.latest_pos.to_numpy()[0]) / (self.stage_step[2]- (i % self.stage_step[3]))
                action[1] = 0
                self.actions_v[i] = action
            elif self.status[i] == 3:
                self.actions_v[i] = np.array([0, -0.008, 0])
            # elif self.status[i] == 0:
            #     self.actions_v[i] = np.array([-0.004, 0, 0])

        return self.actions_v[i]


    def optimize(self, grads, loss_info):
        super(GatheringOPolicy, self).optimize(grads, loss_info)

        # for step in [720, 600, 480, 360, 240, 120]:
        #     if loss_info['temporal_range'] > step:
        #         self.freeze_till = loss_info['temporal_range'] - 120
        #         self.trainable[:self.freeze_till] = False
        #         print(f'feeze till {self.freeze_till}')
        #         break

class MixingPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(MixingPolicy, self).__init__(*args, **kwargs)

        self.trainable = np.full(self.comp_actions_shape[0], False)
        self.status = np.full(self.comp_actions_shape[0], 0)
        self.stage_step = [50, 80]
        for i in range(self.horizon):
            if i % self.stage_step[1] < self.stage_step[0]:
                self.trainable[i] = True
                self.status[i] = 0 # moving
            elif self.stage_step[0] <= i % self.stage_step[1]:
                self.status[i] = 1 # moving back

    def get_action_v(self, i, agent=None, update=False):
        if update:
            if self.status[i] == 1:
                action = (np.array([0.5, 0.73, 0.5]) - agent.rigid.latest_pos.to_numpy()[0]) / (self.stage_step[1]- (i % self.stage_step[1]))
                self.actions_v[i] = action
            # elif self.status[i] == 0:
            #     self.actions_v[i] = np.array([-0.005, 0, 0])

        return self.actions_v[i]

    def optimize(self, grads, loss_info):
        super(MixingPolicy, self).optimize(grads, loss_info)

        steps = list(range(80, 2000, 80))[::-1]
        for step in steps:
            if loss_info['temporal_range'] > step:
                self.freeze_till = loss_info['temporal_range'] - 160
                self.trainable[:self.freeze_till] = False
                print(f'feeze till {self.freeze_till}')
                break

class CirculationPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(CirculationPolicy, self).__init__(*args, **kwargs)


    def optimize(self, grads, loss_info):
        super(CirculationPolicy, self).optimize(grads, loss_info)

        # if loss_info['iteration'] > 150:
        #     self.optim.lr = self.optim.init_lr * 0.1
        #     print(f'lr reduced to {self.optim.lr}')

        # elif loss_info['iteration'] > 100:
        #     self.optim.lr = self.optim.init_lr * 0.25
        #     print(f'lr reduced to {self.optim.lr}')

class PouringPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(PouringPolicy, self).__init__(*args, **kwargs)

class TransportingPolicy(TrainablePolicy):
    def __init__(self, *args, **kwargs):
        super(TransportingPolicy, self).__init__(*args, **kwargs)
        self.trainable = np.full(self.comp_actions_shape[0], False)
        self.trainable[:-1] = True
