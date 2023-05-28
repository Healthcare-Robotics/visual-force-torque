# Visual Force/Torque Sensing

[Paper](https://arxiv.org/abs/2210.00051) | [Video](https://www.youtube.com/watch?v=x0V3p6EUj1s) | [Dataset/Models](https://1drv.ms/f/s!AjebifpxoPl5hOBB-a_D69ip7IxMXQ?e=Co2k2a)

**Can we replace a $6000 force/torque sensor with a $60 USB camera?**

Visual Force/Torque Sensing (VFTS) estimates forces and torques on a robotic gripper from a single RGB image.

![alt text](https://github.com/jeremy-collins/visual-force-torque/blob/main/assets/Headliner.png "Visual Force/Torque Sensing")

## Installation
- Clone the repo on your PC and robot. We use the [Hello Robot Stretch](https://hello-robot.com/stretch-2).
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you haven't done so already.
- Create a virtual environment:
```bash
conda env create -n rp_ft --file rp_ft.yml python=3.9
```
- Verify the robot and pc are on the same network and that the IPs match those in `/robot/zmq_client.py`
- Install pip packages:
```bash
pip install pyyaml keyboard opencv-contrib-python tqdm pyzmq open3d
```

## Live model and demos
```bash
python -m prediction.live_model --config vfts_final_model --live True --view True
python -m demos.clean_curved_surface --config vfts_final_model --live True --view True
python -m demos.clean_manikin --config vfts_final_model --live True --view True
python -m demos.collision_detector --config vfts_final_model --live True --view True
python -m demos.handover --config vfts_final_model --live True --view True
python -m demos.make_bed --config vfts_final_model --live True --view True
```

## [ATI Mini45](https://www.ati-ia.com/products/ft/ft_models.aspx?id=mini45) force/torque sensor setup for collecting ground truth (Tested on Ubuntu 20.04)
- In network settings, Set IPv4 Method to manual.
- Set IPv4 address to 192.168.1.100 and IPv4 netmask to 255.255.255.0.
- Go to 192.168.1.1 in browser.
- After cloning the repo on the robot, run /robot/level_robot.py to level the gripper.
- Press "Snapshot" on the left side of the screen, then press the "bias" button to zero out the force-torque readings.
- Refer to this [manual](https://www.ati-ia.com/app_content/documents/9620-05-NET%20FT.pdf) for more implementation details.

## Hardware
- Hardware for mounting the force/torque sensor and camera can be found [here](https://1drv.ms/f/s!AjebifpxoPl5hOBB-a_D69ip7IxMXQ?e=Co2k2a) and [here](https://hello-robot.com/stretch-teleop-kit).