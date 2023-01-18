import open3d as o3d
import numpy as np
from paper.remove_background import remove_background
import os
from PIL import Image
import cv2

def draw_cylinder_no_bg(color):
    o3d_vis = o3d.visualization.Visualizer()
    o3d_vis.create_window()
    # mag = np.linalg.norm(coords)

    cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=1, height=3, resolution=100, split=4)
    R = cylinder_mesh.get_rotation_matrix_from_xyz((np.pi/4, -np.pi/6, 0))

    cylinder_mesh.compute_vertex_normals()
    cylinder_mesh.rotate(R, center=(0, 0, 0))
    o3d_vis.add_geometry(cylinder_mesh)
    
    # count how many images with the same name are in the directory
    folder_index = len([f for f in os.listdir('./paper/images/arrows')])
    img_name = '/arrows/3d_cylinder_' + str(folder_index)
    img_path = './paper/images/arrows/3d_cylinder_' + str(folder_index) + '.png'

    cylinder_mesh.paint_uniform_color(color)
    ctr = o3d_vis.get_view_control()
    # ctr.set_zoom(1.5)
    o3d_vis.poll_events()
    o3d_vis.update_renderer()
    o3d_vis.capture_screen_image(img_path)
    no_bg_img = remove_background(img_path=img_path, thresh=200)

    no_bg_img.save('./paper/images/' + img_name + '_no_bg.png', 'PNG')

    # display image
    img = cv2.imread('./paper/images/' + img_name + '_no_bg.png')
    cv2.imshow('image', img)
    cv2.waitKey(0)
    print('Saved image to: ' + img_name + '_no_bg.png')

if __name__ == '__main__':
    color = [0.5, 0.5, 0.5]

    draw_cylinder_no_bg(color)