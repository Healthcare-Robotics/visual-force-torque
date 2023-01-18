# visual-force-torque
Estimating the forces and torques applied to a soft robotic gripper using a gripper-mounted camera.

## Setup
### Force-Torque Sensor ([ATI Mini45](https://www.ati-ia.com/products/ft/ft_models.aspx?id=mini45))
- In network settings, Set IPv4 Method to manual
- Set IPv4 address to 192.168.1.100 and IPv4 netmask to 255.255.255.0
- Go to 192.168.1.1 in browser
- After cloning the repo on the robot, run /robot/level_robot.py to level the gripper
- Press "Snapshot" on the left side of the screen, then press the "bias" button to zero out the force-torque readings

### Machine
- Clone the repo on your pc
- Install Miniconda if you haven't done so already
- Create a virtual environment from the yaml file by running `conda env create -n rp_ft --file rp_ft.yml python=3.9`
- Verify the robot and pc are on the same network and that the IPs match the ones in /robot/zmq_client.py
- Install pip with the command `conda install pip`
- Install pip packages with the command `pip install pyyaml keyboard opencv-contrib-python tqdm pyzmq open3d`
- Navigate to /DynamixelSDK/python and type `pip install -e .`

## Live model and demos
- `python -m prediction.live_model --config <config name> --index <checkpoint index> --epoch <epoch> --live True --view True`
- Same arguments for demos
