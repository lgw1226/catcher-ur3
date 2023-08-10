import mujoco
import mujoco.viewer

xml_path = './assets/ur3/single_ur3_base.xml'

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

mujoco.viewer.launch(model)
