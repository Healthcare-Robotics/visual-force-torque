import cv2
from PIL import Image

def remove_background(img_path, thresh=125):
    img = Image.open(img_path)
    img = img.convert('RGBA')
    data = img.getdata()
    new_data = []

    for item in data:
        if item[0] > thresh and item[1] > thresh and item[2] > thresh:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    
    img.putdata(new_data)
    
    return img

if __name__ == '__main__':
    img_name = 'axes_3d'
    img_path = './paper/images/' + img_name + '.png'
    img = remove_background(img_path, img_name)
    img = cv2.imread('./paper/images/' + img_name + '_no_bg.png')
    cv2.imshow('image', img)
    cv2.waitKey(0)