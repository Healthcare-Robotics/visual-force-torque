import cv2
import open3d as o3d
import argparse
import numpy as np

def plot_3d_arrow(self, coords, color, translation):
        mag = np.linalg.norm(coords)
        arrow_scale = 0.15
        force_arrow_mesh = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=arrow_scale*0.5, cone_radius=arrow_scale, cylinder_height=mag*arrow_scale, cone_height=arrow_scale*2, resolution=20, cylinder_split=4, cone_split=1)

        u = np.array([0, 0, 1])
        v = coords / np.linalg.norm(coords)
        # finding quaternion from u to v
        q = (1 + np.dot(u, v), np.cross(u, v)[0], np.cross(u, v)[1], np.cross(u, v)[2])
        q = q / np.linalg.norm(q) 

        # rotation from initial orientation to correct orientation        
        R1 = force_arrow_mesh.get_rotation_matrix_from_quaternion(q)
        # changing perspective of arrow
        R2 = force_arrow_mesh.get_rotation_matrix_from_xyz((0, np.pi/2, -np.pi/2))

        force_arrow_mesh.compute_vertex_normals()
        force_arrow_mesh.paint_uniform_color(color)
        force_arrow_mesh.rotate(R1, center=(0, 0, 0))
        force_arrow_mesh.rotate(R2, center=(0, 0, 0))
        force_arrow_mesh.rotate(gripper_rot, center=(0, 0, 0))
        force_arrow_mesh.translate(gripper_rot @ translation) # rotating translation vector with gripper
        o3d_vis.add_geometry(force_arrow_mesh)

def draw_arrow_on_img(img, origin, gt, pred, name):
    plotter = Plotter()
    arrow_scale = 0.15
    o3d_vis = o3d.visualization.Visualizer()
    o3d_vis.create_window()
    # bgd = o3d.io.read_image(img)
    bgd = cv2.imread(img)

    pred = (-int(pred[1] * arrow_scale), int(pred[0] * arrow_scale))
    gt = (-int(gt[1] * arrow_scale), int(gt[0] * arrow_scale))

    pred_arrow_mesh = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=arrow_scale*0.5, cone_radius=arrow_scale, cylinder_height=mag*arrow_scale, cone_height=arrow_scale*2, resolution=20, cylinder_split=4, cone_split=1)
    gt_arrow_mesh = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=arrow_scale*0.5, cone_radius=arrow_scale, cylinder_height=mag*arrow_scale, cone_height=arrow_scale*2, resolution=20, cylinder_split=4, cone_split=1)

    o3d_vis.add_geometry(pred_arrow_mesh)
    o3d_vis.add_geometry(gt_arrow_mesh)
    
    o3d.visualization.draw_geometries([bgd])



    # saving the image
    o3d.io.write_image('./paper/figures/images/' + args.name + '.png', bgd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='unnamed')
    args = parser.parse_args()

    img = './assets/stretch_gripper_side_view.png'
    origin = (0, 0)
    gt = (100, 100)
    pred = (200, 200)
    draw_arrow_on_img(img, origin, gt, pred, args.name)
