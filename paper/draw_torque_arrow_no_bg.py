import open3d as o3d
import numpy as np
from paper.remove_background import remove_background
import os
from PIL import Image
import cv2


def draw_torque_arrow_no_bg(coords, color):
    o3d_vis = o3d.visualization.Visualizer()
    o3d_vis.create_window()
    mag = np.linalg.norm(coords)
    arrow_scale = 0.015
    torque_arrow_mesh = o3d.io.read_triangle_mesh("./assets/torque_arrow.STL")
    center = torque_arrow_mesh.get_center()
    torque_arrow_mesh.scale(mag*arrow_scale, center=center)

    u = np.array([0, 1, 0])
    v = coords / np.linalg.norm(coords)

    # finding quaternion from u to v
    q = (1 + np.dot(u, v), np.cross(u, v)[0], np.cross(u, v)[1], np.cross(u, v)[2])
    q = q / np.linalg.norm(q) 

    # rotation from initial orientation to correct orientation        
    R1 = torque_arrow_mesh.get_rotation_matrix_from_quaternion(q)
    # changing perspective of arrow
    R2 = torque_arrow_mesh.get_rotation_matrix_from_xyz((0, np.pi/2, -np.pi/2))

    # arbitrary corrections
    R3 = torque_arrow_mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))


    torque_arrow_mesh.compute_vertex_normals()
    torque_arrow_mesh.rotate(R1, center=(0, 0, 0))
    torque_arrow_mesh.rotate(R2, center=(0, 0, 0))
    torque_arrow_mesh.rotate(R3, center=(0, 0, 0))

    o3d_vis.add_geometry(torque_arrow_mesh)
    
    # count how many images with the same name are in the directory
    folder_index = len([f for f in os.listdir('./paper/images/arrows') if f.startswith('3d_arrow')])
    img_name = '/arrows/3d_arrow_' + str(folder_index)
    img_path = './paper/images/arrows/3d_torque_arrow_' + str(folder_index) + '.png'

    torque_arrow_mesh.paint_uniform_color(color)
    ctr = o3d_vis.get_view_control()
    ctr.set_zoom(1.5)
    o3d_vis.poll_events()
    o3d_vis.update_renderer()
    o3d_vis.capture_screen_image(img_path)
    no_bg_img = remove_background(img_path=img_path)

    no_bg_img.save('./paper/images/' + img_name + '_no_bg.png', 'PNG')

    # display image
    img = cv2.imread('./paper/images/' + img_name + '_no_bg.png')
    cv2.imshow('image', img)
    cv2.waitKey(0)
    print('Saved image to: ' + img_name + '_no_bg.png')

if __name__ == '__main__':
    color = [1, 0, 0]
    coords = np.array([-10, -2.5, 0])

    draw_torque_arrow_no_bg(coords, color)