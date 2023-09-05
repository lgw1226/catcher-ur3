import threading
import time
from collections import deque

import numpy as np

from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.optim as optim

import rospy
from geometry_msgs.msg import PoseStamped

from catch_ur3.envs.catch_ur3 import CatchUR3Env


class ParabolaModel(nn.Module):

    def __init__(self):
        super(ParabolaModel, self).__init__()

        self.g = -9.81

        self.fc_x = nn.Linear(1, 1, dtype=torch.float64, bias=True)
        self.fc_y = nn.Linear(1, 1, dtype=torch.float64)
        self.fc_z = nn.Linear(1, 1, dtype=torch.float64)

        self.optimizer = optim.SGD(self.parameters(), lr=0.05)
        self.criterion = nn.L1Loss()

    def initialize_parameters(self, px, py, pz, vx, vy, vz):

        self.fc_x.weight = torch.nn.Parameter(torch.tensor([[vx]], dtype=torch.float64, requires_grad=True))  # v_x
        self.fc_x.bias = torch.nn.Parameter(torch.tensor([[px]], dtype=torch.float64, requires_grad=True))  # p_x

        self.fc_y.weight = torch.nn.Parameter(torch.tensor([[vy]], dtype=torch.float64, requires_grad=True))  # v_y
        self.fc_y.bias = torch.nn.Parameter(torch.tensor([[py]], dtype=torch.float64, requires_grad=True))  # p_y

        self.fc_z.weight = torch.nn.Parameter(torch.tensor([[vz]], dtype=torch.float64, requires_grad=True))  # v_z
        self.fc_z.bias = torch.nn.Parameter(torch.tensor([[pz]], dtype=torch.float64, requires_grad=True))  # p_z

    def forward(self, t):

        pred_x = self.fc_x(t)
        pred_y = self.fc_y(t)
        pred_z = self.fc_z(t) + 0.5 * self.g * t ** 2

        return torch.cat((pred_x, pred_y, pred_z), dim=1)
    
    def fit(self, data, eps=0.01, max_iter=200):

        p = data[:,0:3]
        t = data[:,3].unsqueeze(1)

        for idx_iter in range(max_iter):

            pred_p = self.forward(t)
            loss = self.criterion(pred_p, p)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if torch.mean(loss) < eps: break

        return loss, idx_iter + 1
    
class Buffer():

    def __init__(self, maxlen=20):

        self.memory = deque(maxlen=maxlen)

    def push(self, data):

        self.memory.append(data)

    def pull(self, num_sample, timeout=10):

        timeout_count = 0

        while len(self.memory) < num_sample:

            time.sleep(0.001)

            timeout_count +=1
            if timeout_count >= (timeout / 0.001):
                raise TimeoutError

        return torch.stack([self.memory.popleft() for _ in range(num_sample)])

    
def ros_callback(data, buffer):
    
    wall_time = data.header.stamp.secs + data.header.stamp.nsecs / 1e9
    wall_time = wall_time % 200

    data = torch.tensor([data.pose.position.x,
                         data.pose.position.y,
                         data.pose.position.z,
                         wall_time], dtype=torch.float64)

    buffer.push(data)

def ros_node(ros_topic_name, buffer):

    rospy.Subscriber(
        f'optitrack/{ros_topic_name}/poseStamped',
        PoseStamped,
        ros_callback,
        (buffer)
    )

    rospy.spin()


def is_parabolic(data):

    target_acc = torch.tensor([0, 0, -9.8])
    norm_err_threshold = 1

    t = data[:,3]
    dt = t[1:] - t[:-1]
    mean_dt = torch.mean(dt)

    px = data[:,0]
    py = data[:,1]
    pz = data[:,2]

    ax = torch.tensor(savgol_filter(px, len(data), 2, 2, mean_dt))
    ay = torch.tensor(savgol_filter(py, len(data), 2, 2, mean_dt))
    az = torch.tensor(savgol_filter(pz, len(data), 2, 2, mean_dt))

    mean_acc = torch.mean(torch.stack([ax, ay, az]), dim=1)
    norm_err = torch.norm(mean_acc - target_acc)

    if norm_err <= norm_err_threshold:
        return True
    else:
        return False

def get_parabola_ic(data):

    t = data[:,3]
    dt = t[1:] - t[:-1]
    mean_dt = torch.mean(dt)

    px = data[:,0]
    py = data[:,1]
    pz = data[:,2]

    vx = torch.tensor(savgol_filter(px, len(data), 2, 1, mean_dt))
    vy = torch.tensor(savgol_filter(py, len(data), 2, 1, mean_dt))
    vz = torch.tensor(savgol_filter(pz, len(data), 2, 1, mean_dt))

    init_px = torch.mean(px)
    init_py = torch.mean(py)
    init_pz = torch.mean(pz)

    init_vx = torch.mean(vx)
    init_vy = torch.mean(vy)
    init_vz = torch.mean(vz)

    init_t = torch.mean(t)

    return init_px, init_py, init_pz, init_vx, init_vy, init_vz, init_t

def set_parabola_time(data, t):

    data[:,3] = data[:,3] - t

    return data


buffer = Buffer(50)

ros_topic_name = 'glee_cube'

rospy.init_node('ros_optitrack')
ros_thread = threading.Thread(target=ros_node, args=(ros_topic_name, buffer, ), daemon=True)
ros_thread.start()

num_sample = 9  # must be an odd integer

while True:

    data = buffer.pull(num_sample)

    if is_parabolic(data):
        print('parabolic trajectory found')
        break

px, py, pz, vx, vy, vz, t_offset = get_parabola_ic(buffer.pull(num_sample))

model = ParabolaModel()
model.initialize_parameters(px, py, pz, vx, vy, vz)

print(model.fit(set_parabola_time(buffer.pull(num_sample), t_offset)))