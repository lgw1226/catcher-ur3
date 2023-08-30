__author__ = "Martin Huus Bjerge"
__copyright__ = "Copyright 2017, Rope Robotics ApS, Denmark"
__license__ = "MIT License"

import URBasic
import time

import numpy as np


HOST = '192.168.5.102'  # IP address of the UR3 robot to use
ACC = 0.5
VEL = 0.5


def ur3_connection_test():
    '''Test connection status and check basic commands.'''

    robotModel = URBasic.RobotModel()
    robot = URBasic.UrScriptExt(host=HOST, robotModel=robotModel)

    ur3_reset_state = robot.reset_error()

    if ur3_reset_state: print('Connection established')
    else: print('Connection failed')

    qpos = robot.get_actual_joint_positions()
    tcp_pos = robot.get_actual_tcp_pose()

    print(f"Joint coordinate: {qpos}")
    print(f"Tool Center Point (TCP) pose: {tcp_pos}")

    robot.close()

def ur3_movej_joint_test():

    robotModel = URBasic.RobotModel()
    robot = URBasic.UrScriptExt(host=HOST, robotModel=robotModel)

    robot.reset_error()

    qpos = robot.get_actual_joint_positions()

    increment = np.array([np.pi / 8, 0, 0, 0, 0, 0])
    qpos_target = qpos + increment

    robot.movej(q=qpos_target, a=ACC, v=VEL)

    robot.close()

def ur3_movej_ee_test():

    robotModel = URBasic.RobotModel()
    robot = URBasic.UrScriptExt(host=HOST, robotModel=robotModel)

    robot.reset_error()

    tool_pose = robot.get_actual_tcp_pose()

    increment = np.zeros(6)
    increment[0] = 0.05
    tool_pose_target = tool_pose

    for i in range(10):
        tool_pose_target += increment
        increment = - increment

        robot.movej(pose=tool_pose_target, a=ACC, v=VEL)

    robot.close()

def ur3_movel_ee_test():

    robotModel = URBasic.RobotModel()
    robot = URBasic.UrScriptExt(host=HOST, robotModel=robotModel)

    robot.reset_error()

    tool_pose = robot.get_actual_tcp_pose()

    increment = np.zeros(6)
    increment[0] = 0.05
    tool_pose_target = tool_pose

    for i in range(10):
        tool_pose_target += increment
        increment = - increment

        robot.movel(pose=tool_pose_target, a=ACC, v=VEL)

    robot.close()

def ExampleExtendedFunctions():
    '''
    This is an example of an extension to the Universal Robot script library. 
    How to update the force parameters remote via the RTDE interface, 
    hence without sending new programs to the controller.
    This enables to update force "realtime" (125Hz)  
    '''
    robotModel = URBasic.RobotModel()
    robot = URBasic.UrScriptExt(host=host, robotModel=robotModel)

    print('forcs_remote')
    robot.set_force_remote(task_frame=[0., 0., 0.,  0., 0., 0.], selection_vector=[0,0,1,0,0,0], wrench=[0., 0., 20.,  0., 0., 0.], f_type=2, limits=[2, 2, 1.5, 1, 1, 1])
    robot.reset_error()
    a = 0
    upFlag = True
    while a<3:
        pose = robot.get_actual_tcp_pose()
        if pose[2]>0.1 and upFlag:
            print('Move Down')
            robot.set_force_remote(task_frame=[0., 0., 0.,  0., 0., 0.], selection_vector=[0,0,1,0,0,0], wrench=[0., 0., -20.,  0., 0., 0.], f_type=2, limits=[2, 2, 1.5, 1, 1, 1])
            a +=1
            upFlag = False
        if pose[2]<0.0 and not upFlag:
            print('Move Up')
            robot.set_force_remote(task_frame=[0., 0., 0.,  0., 0., 0.], selection_vector=[0,0,1,0,0,0], wrench=[0., 0., 20.,  0., 0., 0.], f_type=2, limits=[2, 2, 1.5, 1, 1, 1])
            upFlag = True    
    robot.end_force_mode()
    robot.reset_error()
    robot.close()
        

if __name__ == '__main__':

    ur3_connection_test()
    