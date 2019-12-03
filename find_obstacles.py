'''
Author 2019 Refael Vivanti

This class loads a network that was trained using Reinforcement Learning,
where it only had predict the expected cumulative reward of each action using regression,
and extract from it the obstacles map which caused the agent decision.
'''


from my_gradcam import GradCam
import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt


class FindObstacles:

    def __init__(self):
        # load network of roof drone driver:
        net_path = r"D:\projects\rumba\code\python\pytorch-cnn-visualizations\src\trained_models\roof-v0.PT"
        self.actor_critic, ob_rms = torch.load(net_path)

        # load image and convert to torch:
        self.sz = 84
        self.num_envs = 1
        self.target_layer = 5
        self.target_class = 1
        self.empty_area_height = 30
        self.device = 'cpu'

    def find_obstacles(self, image):
        im_small = cv.resize(image, (self.sz, self.sz), interpolation=cv.INTER_AREA)
        if len(im_small.shape) == 3:
            im_gray_small = cv.cvtColor(im_small, cv.COLOR_RGB2GRAY)
        frame = np.expand_dims(im_gray_small, 0)
        obs = torch.from_numpy(frame).float().to(self.device)
        low = np.zeros((4, self.sz, self.sz))
        stacked_obs = torch.zeros((self.num_envs,) + low.shape).to(self.device)
        stacked_obs = torch.zeros(stacked_obs.shape)
        stacked_obs[:, -1:] = obs

        prep_img = stacked_obs
        pretrained_model = self.actor_critic
        grad_cam = GradCam(pretrained_model, target_layer=self.target_layer)
        cam = grad_cam.generate_cam(prep_img, self.target_class)
        empty_region = cam[self.sz - self.empty_area_height:, :]
        empty_mean = empty_region.mean()
        empty_std = empty_region.std()
        thresh = empty_mean + 3.*empty_std
        obstacles_seg = (cam > thresh).astype(int)
        return obstacles_seg

if __name__ == '__main__':
    find_obstacles = FindObstacles()
    im = cv.imread(r"D:\projects\rumba\data\gradcam\input_images\image (3).jpg")
    im = im[:, :, ::-1]
    seg = find_obstacles.find_obstacles(im)
    seg = cv.resize(seg.astype('uint8'), (im.shape[1], im.shape[0]), interpolation=cv.INTER_AREA)
    red = im[:, :, 0].astype(int)
    red += seg.astype('uint8') * 70
    red[red>255] = 255
    im[:, :, 0] = red.astype(im.dtype)
    plt.imshow(im)
    plt.show()
    a=1