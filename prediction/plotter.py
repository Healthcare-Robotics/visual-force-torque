import cv2
import numpy as np
import open3d as o3d
from prediction.config_utils import *
import trimesh

class Plotter():
    def __init__(self, frame):
        self.config, self.args = parse_config_args()
        # opencv plotting
        self.fig = frame
        self.fig_border = 100 # border around entire figure
        self.graph_size = (300, frame.shape[0]) # xy

        self.frame_top_left = (self.fig_border, self.fig_border) # xy 
        self.frame_bottom_right = (self.fig_border + frame.shape[1], self.fig_border + frame.shape[0]) # xy
        self.frame_center = ((self.frame_top_left[0] + self.frame_bottom_right[0]) // 2, (self.frame_top_left[1] + self.frame_bottom_right[1]) // 2)

        self.force_top_left = (self.frame_bottom_right[0] + self.fig_border, self.frame_top_left[1]) # xy
        self.force_bottom_right = (self.force_top_left[0] + self.graph_size[0], self.force_top_left[1] + self.graph_size[1]) # xy
        self.force_center_left = (self.force_top_left[0], int(np.mean((self.force_top_left[1], self.force_bottom_right[1])))) # xy
        self.force_center_top = (int(np.mean((self.force_top_left[0], self.force_bottom_right[0]))), self.force_top_left[1]) # xy

        self.torque_top_left = (self.force_bottom_right[0] + self.fig_border, self.frame_top_left[1]) # xy
        self.torque_bottom_right = (self.torque_top_left[0] + self.graph_size[0], self.torque_top_left[1] + self.graph_size[1]) # xy
        self.torque_center_left = (self.torque_top_left[0], int(np.mean((self.torque_top_left[1], self.torque_bottom_right[1])))) # xy
        self.torque_center_top = (int(np.mean((self.torque_top_left[0], self.torque_bottom_right[0]))), self.torque_top_left[1]) # xy

        self.o3d_frame = cv2.imread('./assets/o3d_frame.png')
        self.o3d_frame = self.o3d_frame[50:-100, 100:-250, :]
        ar = self.o3d_frame.shape[0] / self.o3d_frame.shape[1]
        self.o3d_frame = cv2.resize(self.o3d_frame, (frame.shape[1]* 2, int(frame.shape[1] * 2 * ar)))
        self.o3d_frame_top_left = (self.frame_top_left[0], self.frame_bottom_right[1]) # xy, this is the bottom left corner of the frame
        self.o3d_frame_bottom_right = (self.o3d_frame_top_left[0] + self.o3d_frame.shape[1], self.o3d_frame_top_left[1] + self.o3d_frame.shape[0]) # xy, this is the top right corner of the frame

        self.fig_size = (frame.shape[1] + 2*self.graph_size[0] + 4*self.fig_border, frame.shape[0] + self.o3d_frame.shape[0] + 2*self.fig_border) # xy

        self.tick_length = 10
        self.num_v_ticks_force = 7
        self.num_v_ticks_torque = 5
        self.bar_width = int(self.graph_size[0] / 6)

        self.f_labels = ['Fx', 'Fy', 'Fz']
        self.t_labels = ['Tx', 'Ty', 'Tz']

        self.collision_flag = False

        if self.args.soft:
            self.gripper_mesh = o3d.io.read_triangle_mesh("./assets/softgripping.STL")
            self.gripper_mesh.scale(1000/25.4, center=self.gripper_mesh.get_center())
            self.gripper_rot = self.gripper_mesh.get_rotation_matrix_from_xyz((np.pi * 45 / 180, np.pi * 45 / 180, 0))
            self.gripper_rot_soft = self.gripper_mesh.get_rotation_matrix_from_xyz((np.pi * 135 / 180, np.pi * 180 / 180, np.pi * 45 / 180))
            self.gripper_mesh.rotate(self.gripper_rot_soft, center=(0, 0, 0))
            self.gripper_mesh.translate((3, 1, 1))

        else:
            self.gripper_mesh = o3d.io.read_triangle_mesh("./assets/stretch_dex_gripper_assembly_cutout.STL")
            self.gripper_rot = self.gripper_mesh.get_rotation_matrix_from_xyz((np.pi * 45 / 180, np.pi * 45 / 180, 0))
            self.gripper_mesh.rotate(self.gripper_rot, center=(0, 0, 0))

        self.gripper_mesh.compute_vertex_normals()

        # 3d visualization
        self.o3d_vis = o3d.visualization.Visualizer()
        self.o3d_vis.create_window()

    def indicate_collision(self):
            cv2.putText(self.fig, 'COLLISION', (int(np.mean((self.frame_top_left[0], self.frame_bottom_right[0])) - 75), int(np.mean((self.frame_top_left[1], self.frame_bottom_right[1])) + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    def plot_arrow(self, force_pred, force_gt):
        arrow_scale = 10
        arrow_origin = (self.frame_center[0], self.frame_center[1] + 50)
        arrow_vector = (-int(force_pred[1] * arrow_scale), int(force_pred[0] * arrow_scale))
        arrow_vector_gt = (-int(force_gt[1] * arrow_scale), int(force_gt[0] * arrow_scale))

        # gt
        cv2.arrowedLine(self.fig, arrow_origin, (arrow_origin[0] + arrow_vector_gt[0], arrow_origin[1] + arrow_vector_gt[1]), (36, 255, 36), 2, line_type=cv2.MARKER_TRIANGLE_UP, tipLength=0.15)
        
        # pred
        cv2.arrowedLine(self.fig, arrow_origin, (arrow_origin[0] + arrow_vector[0], arrow_origin[1] + arrow_vector[1]), (255, 182, 109), 2, line_type=cv2.MARKER_TRIANGLE_UP, tipLength=0.15)

    def visualize_ft(self, force_gt, torque_gt, force_pred, torque_pred, frame, collision_flag, view_3D=True):
        if self.args.stage == 'train':
            force_pred = np.array([0, 0, 0])
            torque_pred = np.array([0, 0, 0])

        self.collision_flag = collision_flag # for collision demo

        self.o3d_frame = cv2.imread('./assets/o3d_frame.png')
        self.o3d_frame = self.o3d_frame[50:-100, 100:-250, :]
        ar = self.o3d_frame.shape[0] / self.o3d_frame.shape[1]
        self.o3d_frame = cv2.resize(self.o3d_frame, (frame.shape[1] * 2, int(frame.shape[1] * 2 * ar)))
        self.axes_img = cv2.imread('./assets/axes_3d.png')
        self.axes_img = cv2.resize(self.axes_img, (250, 250))

        self.fig = np.ones((self.fig_size[1], self.fig_size[0], 3), dtype=np.uint8)*255
        self.fig[self.frame_top_left[1]:self.frame_bottom_right[1], self.frame_top_left[0]:self.frame_bottom_right[0], :] = frame
        self.fig[self.o3d_frame_top_left[1]:self.o3d_frame_bottom_right[1], self.o3d_frame_top_left[0]:self.o3d_frame_bottom_right[0], :] = self.o3d_frame

        self.fig[self.o3d_frame_top_left[1] + self.fig_border:self.o3d_frame_top_left[1] + self.axes_img.shape[1] + self.fig_border, self.o3d_frame_bottom_right[0] - self.fig_border:self.o3d_frame_bottom_right[0] + self.axes_img.shape[0] - self.fig_border, :] = self.axes_img

        # FORCES
        cv2.rectangle(self.fig, self.force_top_left, self.force_bottom_right, (0, 0, 0), thickness=1)
        # title
        cv2.putText(self.fig, 'Forces', (self.force_center_top[0] - 25, self.force_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # bar graph
        for i in range(len(force_gt)):
            # force_gt
            cv2.rectangle(self.fig, (self.force_top_left[0] + 2*i*self.bar_width, self.force_center_left[1]), (self.force_top_left[0] + (2*i+1)*self.bar_width, (self.force_center_left[1] - int(force_gt[i]*self.graph_size[1]/60))), (36, 255, 36), thickness=-1)
            # force_pred
            cv2.rectangle(self.fig, (self.force_top_left[0] + 2*i*self.bar_width + self.bar_width, self.force_center_left[1]), (self.force_top_left[0] + (2*i+2)*self.bar_width, (self.force_center_left[1] - int(force_pred[i]*self.graph_size[1]/60))), (255, 182, 109), thickness=-1)
            # x axis ticks
            cv2.line(self.fig, (self.force_top_left[0] + 2*i*self.bar_width + self.bar_width, self.force_bottom_right[1]), (self.force_top_left[0] + 2*i*self.bar_width + self.bar_width, self.force_bottom_right[1] + self.tick_length), (0, 0, 0), thickness=1)
            # x axis labels
            cv2.putText(self.fig, self.f_labels[i], (self.force_top_left[0] + 2*i*self.bar_width + 40, self.force_bottom_right[1] + self.tick_length + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        for i in range(self.num_v_ticks_force):
            # y axis ticks
            cv2.line(self.fig, (self.force_center_left[0] - self.tick_length, self.force_bottom_right[1] - int(i * self.graph_size[1] / 6)), (self.force_center_left[0], self.force_bottom_right[1] - int(i * self.graph_size[1] / 6)), (0, 0, 0), thickness=1)
            # y axis labels
            if i-3 < 0:
                cv2.putText(self.fig, str(int(i - 3) * 10), (self.force_center_left[0] - 5 * self.tick_length, self.force_bottom_right[1] - int(i * self.graph_size[1] / 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
            else:    
                cv2.putText(self.fig, str(int(i - 3) * 10), (self.force_center_left[0] - 4 * self.tick_length, self.force_bottom_right[1] - int(i * self.graph_size[1] / 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)

        # TORQUES
        cv2.rectangle(self.fig, self.torque_top_left, self.torque_bottom_right, (0, 0, 0), thickness=1)
        # title
        cv2.putText(self.fig, 'Torques', (self.torque_center_top[0] - 25, self.torque_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(self.fig, (self.torque_center_left[0] - self.tick_length, self.torque_center_left[1]), (self.torque_center_left[0] + self.tick_length, self.torque_center_left[1]), (0, 0, 0), thickness=1)
        
        # bar graph
        for i in range(len(torque_gt)):
            # torque_gt
            cv2.rectangle(self.fig, (self.torque_top_left[0] + 2*i*self.bar_width, self.torque_center_left[1]), (self.torque_top_left[0] + (2*i+1)*self.bar_width, (self.torque_center_left[1] - int(torque_gt[i]*self.graph_size[1]/10))), (36, 255, 36), thickness=-1)
            # torque_pred
            cv2.rectangle(self.fig, (self.torque_top_left[0] + 2*i*self.bar_width + self.bar_width, self.torque_center_left[1]), (self.torque_top_left[0] + (2*i+2)*self.bar_width, (self.torque_center_left[1] - int(torque_pred[i]*self.graph_size[1]/10))), (255, 182, 109), thickness=-1)
            # x axis ticks
            cv2.line(self.fig, (self.torque_top_left[0] + 2*i*self.bar_width + self.bar_width, self.torque_bottom_right[1]), (self.torque_top_left[0] + 2*i*self.bar_width + self.bar_width, self.torque_bottom_right[1] + self.tick_length), (0, 0, 0), thickness=1)
            # x axis labels
            cv2.putText(self.fig, self.t_labels[i], (self.torque_top_left[0] + 2*i*self.bar_width + 40, self.torque_bottom_right[1] + self.tick_length + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        for i in range(self.num_v_ticks_torque):
            # y axis ticks
            cv2.line(self.fig, (self.torque_center_left[0] - self.tick_length, self.torque_bottom_right[1] - int(i * self.graph_size[1] / 1)), (self.torque_center_left[0], self.torque_bottom_right[1] - int(i * self.graph_size[1] / 1)), (0, 0, 0), thickness=1)
            # y axis labels
            if i - 2 < 0:
                cv2.putText(self.fig, str((i - 2) * 10 / 4), (self.torque_center_left[0] - 5 * self.tick_length, self.torque_bottom_right[1] - int(i * self.graph_size[1] / 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
            else:    
                cv2.putText(self.fig, str((i - 2) * 10 / 4), (self.torque_center_left[0] - 4 * self.tick_length, self.torque_bottom_right[1] - int(i * self.graph_size[1] / 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)

        # legend
        cv2.rectangle(self.fig, (self.fig_size[0] - self.fig_border - 100, 10), (self.fig_size[0] - self.fig_border - 50, self.fig_border // 3), (36, 255, 36), thickness=-1)
        cv2.putText(self.fig, 'Ground Truth', (self.fig_size[0] - self.fig_border - 40, self.fig_border // 3 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(self.fig, (self.fig_size[0] - self.fig_border - 100, 10 + self.fig_border // 3), (self.fig_size[0] - self.fig_border - 50, 2 * self.fig_border // 3), (255, 182, 109), thickness=-1)
        cv2.putText(self.fig, 'Prediction', (self.fig_size[0] - self.fig_border - 40, self.fig_border * 2 // 3 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        if self.collision_flag:
            self.indicate_collision()

        if view_3D:
            self.render_3d_view(force_gt, torque_gt, force_pred, torque_pred)

        return self.fig

    def render_3d_view(self, force_gt, torque_gt, force_pred, torque_pred):
        if self.args.stage == 'train':
            force_pred = np.array([0, 0, 0])
            torque_pred = np.array([0, 0, 0])
            
        if self.args.soft:
            force_trans = np.array([3, 2.0, 2.75])
            torque_trans = np.array([-1.3, 2.25, 3])
            axis_trans = np.array([12, 2, -1.5])
        else:
            force_trans = np.array([6, 2.5, 3])
            torque_trans = np.array([-1.3, 2.5, 3])
            axis_trans = np.array([12, 2, -1.5])

        # self.plot_axes(translation=axis_trans)
        norm_gt_color = np.array([0.0, 0.4, 0.6])
        norm_pred_color = np.array([0.5, 0.85, 1.0])

        # force arrows
        if np.linalg.norm(force_gt) > 0:
            self.plot_3d_force_arrow(coords=force_gt, color=norm_gt_color, translation=force_trans)
        if np.linalg.norm(force_pred) > 0:
            self.plot_3d_force_arrow(coords=force_pred, color=norm_pred_color, translation=force_trans)
 
        # torque arrows
        cube_len = 0.8
        self.plot_3d_torque_arrow(-torque_pred[0], color=[1, 0.5, 0.5], translation=torque_trans + [0, -cube_len, 0], rot=(0, 0, 0))  # predicted x
        self.plot_3d_torque_arrow(-torque_gt[0], color=[0.8, 0, 0], translation=torque_trans + [0.001, -cube_len - 0.001, 0.001], rot=(0, 0, 0))  # gt x
        self.plot_3d_torque_arrow(-torque_pred[1], color=[0.5, 1, 0.5], translation=torque_trans + [0, 0, -cube_len], rot=(np.pi/2, 0, 0))    # predicted y
        self.plot_3d_torque_arrow(-torque_gt[1], color=[0, 0.8, 0], translation=torque_trans + [0.001, 0.001, -cube_len - 0.001], rot=(np.pi/2, 0, 0))    # gt y
        self.plot_3d_torque_arrow(-torque_pred[2], color=[0.5, 0.5, 1], translation=torque_trans + [cube_len, 0, 0], rot=(0, 0, np.pi/2))   # predicted z
        self.plot_3d_torque_arrow(-torque_gt[2], color=[0.2, 0.2, 1.0], translation=torque_trans + [cube_len + 0.001, 0.001, 0.001], rot=(0, 0, np.pi/2))   # gt z

        self.plot_gripper_mesh(rotate_frame=False)

        self.o3d_set_camera_extrinsic(self.o3d_vis)

        self.o3d_vis.poll_events()
        self.o3d_vis.update_renderer()
        self.o3d_vis.capture_screen_image('./assets/o3d_frame.png')
        self.o3d_vis.clear_geometries()

    def o3d_set_camera_extrinsic(self, vis, transform=None):
        """
        Sets the Open3D camera position and orientation
        :param vis: Open3D visualizer object
        :param transform: 4x4 numpy defining a rigid transform where the camera should go
        """

        if transform is None:
            transform = [
                0.99146690872376264,
                -0.086692593434608167,
                0.097353803979752032,
                0.0,
                -0.10722297678715105,
                -0.96707368565286655,
                0.2308066718417883,
                0.0,
                0.074139073063050315,
                -0.23927574210805952,
                -0.96811699555580022,
                0.0,
                -3.7649081364966746,
                2.0632784891845182,
                9.8526821131006859,
                1.0
            ]
            transform = np.array(transform).reshape(4, 4).T

        ctr = vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        cam.extrinsic = transform
        ctr.convert_from_pinhole_camera_parameters(cam)


    def plot_3d_force_arrow(self, coords, color, translation):
        mag = np.linalg.norm(coords) * 1.5
        arrow_scale = 0.15 * 1.4
        self.force_arrow_mesh = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=arrow_scale, cone_radius=arrow_scale*1.5, cylinder_height=mag*arrow_scale, cone_height=arrow_scale*3, resolution=20, cylinder_split=4, cone_split=1)

        u = np.array([0, 0, 1])
        v = coords / np.linalg.norm(coords)

        # finding quaternion from u to v
        q = (1 + np.dot(u, v), np.cross(u, v)[0], np.cross(u, v)[1], np.cross(u, v)[2])
        q = q / np.linalg.norm(q)

        # rotation from initial orientation to correct orientation
        R1 = self.force_arrow_mesh.get_rotation_matrix_from_quaternion(q)
        # changing perspective of arrow
        R2 = self.force_arrow_mesh.get_rotation_matrix_from_xyz((0, np.pi/2, -np.pi/2))

        self.force_arrow_mesh.compute_vertex_normals()
        self.force_arrow_mesh.paint_uniform_color(color)
        self.force_arrow_mesh.rotate(R1, center=(0, 0, 0))
        self.force_arrow_mesh.rotate(R2, center=(0, 0, 0))
        self.force_arrow_mesh.rotate(self.gripper_rot, center=(0, 0, 0))
        self.force_arrow_mesh.translate(self.gripper_rot @ translation) # rotating translation vector with gripper
        self.o3d_vis.add_geometry(self.force_arrow_mesh)

    def plot_3d_torque_arrow(self, mag, color, translation, rot):
        ARROW_LENGTH_SCALE = 180 * 1.5
        TORQUE_MINIMUM_VALUE = 0.13
        if abs(mag) < TORQUE_MINIMUM_VALUE:
            return

        arrow_scale = 0.025 * 0.7
        self.torque_arrow_mesh = self.get_arrow_cut_angle(mag * ARROW_LENGTH_SCALE)
        self.torque_arrow_mesh.scale(arrow_scale, center=(0, 0, 0))

        self.torque_arrow_mesh.paint_uniform_color(color)

        R = self.force_arrow_mesh.get_rotation_matrix_from_xyz(rot)
        self.torque_arrow_mesh.rotate(R, center=(0, 0, 0))
        self.torque_arrow_mesh.rotate(self.gripper_rot, center=(0, 0, 0))
        self.torque_arrow_mesh.translate(self.gripper_rot @ translation) # rotating translation vector with gripper
        
        self.o3d_vis.add_geometry(self.torque_arrow_mesh)

    def get_arrow_cut_angle(self, target_angle=45):
        flip = False
        if target_angle < 0:
            flip = True
            target_angle = abs(target_angle)

        ARROW_OFFSET = 0.6195284555660944
        if target_angle < 45:
            #we need to cut the head
            mesh = self.get_arrow_angle(45)
            target_angle_rad = np.deg2rad(target_angle)
            cut_point = np.array([0, 0, 0])
            cut_normal = np.array([np.sin(target_angle_rad), 0, -np.cos(target_angle_rad)])

            tmesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
            tmesh = tmesh.slice_plane(cut_point, cut_normal, cap=True)

            mesh.vertices = o3d.utility.Vector3dVector(tmesh.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(tmesh.faces)
        else:
            mesh = self.get_arrow_angle(target_angle - np.rad2deg(ARROW_OFFSET))

        verts = np.asarray(mesh.vertices)
        ang = np.arctan2(verts[:, 2], verts[:, 0])
        ang = (ang + 2 * np.pi) % (2 * np.pi)

        #rotate forwards
        rotate_amount = -ang.max()

        rotmat = np.eye(3)
        rotmat[0, 0] = np.cos(rotate_amount)
        rotmat[0, 2] = np.sin(rotate_amount)
        rotmat[2, 0] = -np.sin(rotate_amount)
        rotmat[2, 2] = np.cos(rotate_amount)

        new_verts = np.matmul(verts, rotmat)
        mesh.vertices = o3d.utility.Vector3dVector(new_verts)

        if flip:
            R = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            mesh.rotate(R, center=(0, 0, 0))

        mesh.compute_vertex_normals()
        return mesh

    def get_arrow_angle(self, target_angle=45):
        mesh = o3d.io.read_triangle_mesh("./assets/torque_arrow.STL")
        ROTATE_BACK_RADIANS = +0.6195284555660944   # the arrow was modelled as having the base of the arrow at zero, not the tip

        mesh.translate(np.array([-25, -1.5, -25])) # x, y, z. Move it so the center is in the middle of the arrow

        verts = np.asarray(mesh.vertices)

        ang = np.arctan2(verts[:, 2], verts[:, 0])
        ang = (ang + 2 * np.pi) % (2 * np.pi)
        new_ang = ang * target_angle / 270

        new_verts = verts.copy()
        new_verts[:, 1] = 0
        dists = np.linalg.norm(new_verts, axis=1)
        new_verts[:, 0] = np.cos(new_ang) * dists
        new_verts[:, 2] = np.sin(new_ang) * dists
        new_verts[:, 1] = verts[:, 1]

        copy_idx = np.logical_and(verts[:, 0] > 0, verts[:, 2] < 0)

        new_verts[copy_idx, :] = verts[copy_idx, :]

        rotmat = np.eye(3)
        rotmat[0, 0] = np.cos(ROTATE_BACK_RADIANS)
        rotmat[0, 2] = np.sin(ROTATE_BACK_RADIANS)
        rotmat[2, 0] = -np.sin(ROTATE_BACK_RADIANS)
        rotmat[2, 2] = np.cos(ROTATE_BACK_RADIANS)

        new_verts = np.matmul(new_verts, rotmat)
        mesh.vertices = o3d.utility.Vector3dVector(new_verts)
        mesh.compute_vertex_normals()
        return mesh
    
    def plot_axes(self, translation):
        mag = 10
        axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for axis in axes:
            axis = np.array(axis)
            self.plot_3d_force_arrow(coords=mag*axis, color=axis, translation=translation)
           
    def plot_gripper_mesh(self, rotate_frame=False):
        if rotate_frame:
            R = self.gripper_mesh.get_rotation_matrix_from_xyz((0, 0.01, 0))
            self.gripper_rot = R @ self.gripper_rot
            self.gripper_mesh.rotate(R, center=(0, 0, 0))
        self.o3d_vis.add_geometry(self.gripper_mesh)

if __name__ == '__main__':
    frame = np.zeros((320, 480, 3), dtype=np.uint8)
    plotter = Plotter(frame=frame)

    norm_gt_color = np.array([36, 255, 36])
    norm_gt_color = norm_gt_color / np.linalg.norm(norm_gt_color)

    force_vec = np.array([-8, 0, 0])
    force_vec_2 = np.array([-3, 5, 0])
    torque_vec = np.array([1, 1, -0.2])
    torque_vec_2 = np.array([0.3, 1.5, 0.3])
    zero_vec = np.array([0, 0, 0])
    frame = np.zeros((320, 480, 3), dtype=np.uint8)

    img = plotter.visualize_ft(force_vec, torque_vec, force_vec_2, torque_vec_2, frame, collision_flag=False)
    cv2.imshow('plot', img)
    cv2.waitKey(0)