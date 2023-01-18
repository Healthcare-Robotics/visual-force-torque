import torchvision.transforms as transforms
from prediction.config_utils import *
import cv2

class RPFTTransforms:
    def __init__(self, transform_type, stage, pixel_mean, pixel_std):
        self.img_size = (224, 224) # (height, width)
        self.transform_type = transform_type
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.stage = stage
        self.transforms = self.choose_transform(pixel_mean, pixel_std)
    
    def __call__(self, img):
        return self.transforms(img)

    def bottom_center_crop(self, img):
        # resizes the image and takes a bottom center crop such that the image contains the entire gripper
        return transforms.functional.crop(img, top=img.size()[1] - self.img_size[0], left=img.size()[2] // 2 - (self.img_size[1] + 25) // 2, height=self.img_size[0], width=self.img_size[1])
    
    def bottom_center_crop_soft(self, img):
        # resizes the image and takes a bottom center crop such that the image contains the entire gripper
        return transforms.functional.crop(img, top=img.size()[1] - self.img_size[0], left=img.size()[2] // 2 - (self.img_size[1] - 26) // 2, height=self.img_size[0], width=self.img_size[1])

    def bottom_center_crop_big(self, img):
        # takes a bottom center crop such that the image contains the entire gripper
        width = 400
        height = 400
        return transforms.functional.crop(img, top=img.size()[1] - height, left=img.size()[2] // 2 - width // 2, height=height, width=width)

    def get_all_transforms(self, pixel_mean, pixel_std):
        all_transforms = transforms.Compose(
        [
            transforms.Resize(size=275),
            transforms.Lambda(self.bottom_center_crop),
            transforms.Resize(self.img_size),
            transforms.ColorJitter(brightness=0.0, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.Normalize(pixel_mean, pixel_std),
        ]
        )
        return all_transforms

    def get_jitter_transforms(self, pixel_mean, pixel_std):
        jitter_transforms = transforms.Compose(
        [
            transforms.Resize(self.img_size),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ]
        )
        return jitter_transforms

    def get_normalize_transforms(self, pixel_mean, pixel_std):
        norm_transforms = transforms.Compose(
        [
            transforms.Resize(self.img_size),
            transforms.Normalize(pixel_mean, pixel_std),
        ]
        )
        return norm_transforms

    def get_crop_transforms(self, pixel_mean, pixel_std):
        crop_transforms = transforms.Compose(
        [
            transforms.Resize(size=275),
            transforms.Lambda(self.bottom_center_crop),
            transforms.Resize(self.img_size),
        ]
        )
        return crop_transforms

    def get_crop_transforms_soft(self, pixel_mean, pixel_std):
        crop_transforms = transforms.Compose(
        [
            transforms.Resize(size=275),
            transforms.Lambda(self.bottom_center_crop_soft),
            transforms.Resize(self.img_size),
        ]
        )
        return crop_transforms

    def get_random_crop_transforms(self, pixel_mean, pixel_std):
        # performs bottom center crop and then random crop
        random_crop_transforms = transforms.Compose(
        [
            [transforms.Lambda(self.bottom_center_crop_big),
            transforms.Resize(size=275),
            transforms.RandomCrop(size=self.img_size)]
        ]
        )
        return random_crop_transforms
    
    def get_minimal_transforms(self):
        minimal_transforms = transforms.Compose(
        [
            transforms.Resize(self.img_size),
        ]
        )
        return minimal_transforms

    def mask_img_center(self, img, size=150):
        # putting a black square in the center of the image using cv2
        w = img.shape[1]
        h = img.shape[0]
        top_left = (w//2 - size//2, h//2 - size//2 + 50)
        bottom_right = (w//2 + size//2, h//2 + size//2 + 50)
        cv2.rectangle(img, top_left, bottom_right, (0,0,0), -1)
        return img

    def get_test_transforms(self, pixel_mean, pixel_std):
        custom_transform_list = [transforms.Resize(self.img_size)]

        if 'crop' in self.transform_type:
            custom_transform_list = [transforms.Resize(size=275), transforms.Lambda(self.bottom_center_crop)]
        elif 'crop_soft' in self.transform_type:
            custom_transform_list = [transforms.Resize(size=275), transforms.Lambda(self.bottom_center_crop_soft)]
        elif 'random_crop' in self.transform_type:
            # cannot have both crop and random crop
            custom_transform_list = [transforms.Lambda(self.bottom_center_crop_big)]
        if 'normalize' in self.transform_type:
            custom_transform_list.append(transforms.Normalize(pixel_mean, pixel_std))

        return transforms.Compose(custom_transform_list)

    def get_train_transforms(self, pixel_mean, pixel_std):
        custom_transform_list = [transforms.Resize(self.img_size)]

        if 'crop' in self.transform_type:
            custom_transform_list = [transforms.Resize(size=275), transforms.Lambda(self.bottom_center_crop)]
        elif 'crop_soft' in self.transform_type:
            custom_transform_list = [transforms.Resize(size=275), transforms.Lambda(self.bottom_center_crop_soft)]
        elif 'random_crop' in self.transform_type:
            # cannot have both crop and random crop
            custom_transform_list = [transforms.Lambda(self.bottom_center_crop_big), transforms.Resize(size=275), transforms.RandomCrop(size=self.img_size)]

        if 'jitter' in self.transform_type:
            custom_transform_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
        if 'lighting' in self.transform_type:
            custom_transform_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0))
        if 'normalize' in self.transform_type:
            custom_transform_list.append(transforms.Normalize(pixel_mean, pixel_std))
        # if 'flip' in self.transform_type:
        #     custom_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        return transforms.Compose(custom_transform_list)

    def choose_transform(self, pixel_mean, pixel_std):
        if self.stage == 'train':
            if type(self.transform_type) is list:
                return self.get_train_transforms(pixel_mean, pixel_std)
            else:
                raise ValueError('Error: configs.transform should be a list, but is instead a ', self.transform_type)
        elif self.stage == 'test':
            return self.get_test_transforms(pixel_mean, pixel_std)