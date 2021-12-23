import torch
import pyredner
import h5py
from skimage import io
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from model import *
import os
import random
import torchvision.transforms as transforms

# 接收输入
# 初始化一个对象
# 对象里的各种函数调用

# confs
# confs.target_img --- numpy
# confs.cam_pos --- list [3,]
# confs.cam_look_at --- list [3,]
# confs.

confs = {'target_img': './datas/target.png',
         'cam_pos': [-0.2697, -5.7891, 373.9277],
         'cam_look_at': [-0.2697, -5.7891, 54.7918],
         'lr_1': 0.05,
         'lr_2': 0.5,
         'iters': 500,
         'save_dir': './saved_data',
         'save_dir_img': './saved_data/saved_imgs',
         'save_dir_obj': './saved_data/saved_objs',
         }

def get_temp_obj(shape_coeffs,color_coeffs):
    vertices = (shape_mean + shape_basis @ shape_coeffs).view(-1, 3)  # tensor [53149,3] 90,94,131 value  0,-4,55 mean
    normals = pyredner.compute_vertex_normal(vertices, indices)  # ver normal tensor
    colors = (color_mean + color_basis @ color_coeffs).view(-1, 3)  # color tensor 0-1
    # print(max(vertices[:, 0]), max(vertices[:, 1]), max(vertices[:, 2]))
    # print(torch.mean(vertices[:, 0]), torch.mean(vertices[:, 1]), torch.mean(vertices[:, 2]))
    m = pyredner.Material(use_vertex_color=True)  # an object
    obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m, colors=colors)
    return obj


def write_obj_with_colors(obj_name, vertices, triangles, colors):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        colors: shape = (nver, 3)
    '''
    triangles = triangles.copy()
    triangles += 1  # meshlab start with 1

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:

        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0],
                                               colors[i, 1], colors[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)

def save_obj(dir,obj,target_name,iter_cnt):
    obj_name=target_name+'_'+str(iter_cnt)+'.obj'
    pyredner.save_obj(obj,os.path.join(dir,obj_name))
    return


def sample_sphere(n_samples=10, rho=500):
    samples = []
    cnt = 0
    while cnt < n_samples:
        theta = np.pi * random.random()
        phi = 2 * np.pi * random.random()
        x=rho*np.sin(theta)*np.cos(phi)
        y=rho*np.sin(theta)*np.sin(phi)
        z=rho*np.cos(theta)
        if y<-rho*0.2:
            continue
        if z<0:
            if random.random()<0.8:
                continue
        samples.append([x,y,z])
        cnt=cnt+1

    return np.array(samples)

def save_img(save_dir,obj,target_name,iter_cnt,ambient_color,dir_light_intensity):

    sam = sample_sphere()
    objects = [obj]
    camera = pyredner.Camera(position=torch.Tensor([0, 0, 0]), look_at=torch.Tensor([0, 0, 0]),
                             up=torch.Tensor([0, 1, 0]), fov=torch.Tensor([45]))
    camera.resolution = (256, 256)
    for i in range(sam.shape[0]):
        camera.position = torch.Tensor(sam[i])
        scene = pyredner.Scene(camera=camera, objects=objects)
        ambient_light = pyredner.AmbientLight(ambient_color)  # ambient
        dir_light = pyredner.DirectionalLight(torch.tensor([0.0, 0.0, -1.0]), dir_light_intensity)
        img = pyredner.render_deferred(scene=scene, lights=[ambient_light, dir_light]).cpu()
        # img = pyredner.render_albedo(scene).cpu()
        img_name = target_name +'_' + str(iter_cnt) + '_{:03d}.jpg'.format(i)
        img_path = os.path.join(save_dir, img_name)
        pyredner.imwrite(img, img_path)



class Processor:
    def __init__(self, confs):
        self.target_name=confs['target_img'].split('/')[-1].split('.jpg')[0]
        self.target = pyredner.imread(confs['target_img']).to(pyredner.get_device())
        self.cam_pos = torch.tensor(confs['cam_pos'], requires_grad=True)
        self.cam_look_at = torch.tensor(confs['cam_look_at'], requires_grad=True)
        self.shape_coeffs = torch.zeros(199, device=pyredner.get_device(), requires_grad=True)
        self.color_coeffs = torch.zeros(199, device=pyredner.get_device(), requires_grad=True)
        self.ambient_color = torch.ones(3, device=pyredner.get_device(), requires_grad=True)
        self.dir_light_intensity = torch.zeros(3, device=pyredner.get_device(), requires_grad=True)
        self.optimizer = torch.optim.Adam(
            [self.shape_coeffs, self.color_coeffs, self.ambient_color, self.dir_light_intensity], lr=confs['lr_1'])
        self.cam_optimizer = torch.optim.Adam([self.cam_pos, self.cam_look_at], lr=confs['lr_2'])
        self.num_iter = confs['iters']
        self.imgs = []
        self.losses = []

    def train(self):
        # imshow(self.target.cpu())
        print('img_name',self.target_name)
        os.makedirs(confs['save_dir_img'], exist_ok=True)
        os.makedirs(confs['save_dir_obj'], exist_ok=True)
        for t in range(self.num_iter+1):
            self.optimizer.zero_grad()
            self.cam_optimizer.zero_grad()
            img = model(self.cam_pos, self.cam_look_at, self.shape_coeffs, self.color_coeffs, self.ambient_color,
                        self.dir_light_intensity)
            # Compute the loss function. Here it is L2 plus a regularization term to avoid coefficients to be too far from zero.
            # Both img and target are in linear color space, so no gamma correction is needed.
            loss = (img - self.target).pow(2).mean()
            # loss = loss + 0.0001 * self.shape_coeffs.pow(2).mean() + 0.0001 * self.color_coeffs.pow(2).mean()
            loss.backward()
            self.optimizer.step()
            self.cam_optimizer.step()
            self.ambient_color.data.clamp_(0.0)
            self.dir_light_intensity.data.clamp_(0.0)
            # Plot the loss
            self.losses.append(loss.data.item())
            # Only store images every 10th iterations
            if t % 100 == 0:
                self.imgs.append(torch.pow(img.data, 1.0 / 2.2).cpu())  # Record the Gamma corrected image
                f, (ax_loss, ax_diff_img, ax_img) = plt.subplots(1, 3)
                ax_loss.plot(range(len(self.losses)), self.losses, label='loss')
                ax_loss.legend()
                ax_diff_img.imshow((img - self.target).pow(2).sum(dim=2).data.cpu())
                ax_img.imshow(torch.pow(img.data.cpu(), 1.0 / 2.2))
                plt.show()

                temp_obj=get_temp_obj(self.shape_coeffs,self.color_coeffs)
                save_obj(confs['save_dir_obj'],temp_obj,self.target_name,t)
                save_img(confs['save_dir_img'],temp_obj,self.target_name,t,self.ambient_color,self.dir_light_intensity)
                # start to save obj combine color
                vertices_v = (shape_mean + shape_basis @ self.shape_coeffs).view(-1,3)  # tensor [53149,3] 90,94,131 value  0,-4,55 mean
                normals_n = pyredner.compute_vertex_normal(vertices_v, indices)  # ver normal tensor
                colors_c = (color_mean + color_basis @ self.color_coeffs).view(-1, 3)
                obj_name=os.path.join(confs['save_dir_obj'],self.target_name+'_'+str(t)+'_colored.obj')
                write_obj_with_colors(obj_name,vertices_v.data.cpu().numpy(),indices.data.cpu().numpy(),colors_c.data.cpu().numpy())



if __name__ == '__main__':
    processor = Processor(confs)
    print('init ok')
    processor.train()
    # tgt = io.imread('./datas/target.jpg')
    # imshow(tgt)
    # plt.show()
    # tgten = torch.from_numpy(tgt).to(pyredner.get_device())
    # # # transf = transforms.ToTensor()
    # # # img_tensor = transf(tgt).T
    # #
    # imshow(tgten.cpu())
    # plt.show()
    # imshow(torch.pow(tgten, 1.0).cpu())
    # plt.show()
