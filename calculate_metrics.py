import os
import numpy as np
import torch
# from hdf5storage import loadmat
from torch import nn

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        Itrue = im_true.clamp(0., 1.)*data_range
        Ifake = im_fake.clamp(0., 1.)*data_range
        err=Itrue-Ifake
        err=torch.pow(err,2)
        err = torch.mean(err,dim=0)
        err = torch.mean(err,dim=0)

        psnr = 10. * torch.log10((data_range ** 2) / err)
        psnr=torch.mean(psnr)
        return psnr


class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs.clamp(0., 1.)*255- label.clamp(0., 1.)*255
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse

class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()
        self.eps=2.2204e-16
    def forward(self,im1, im2):
        assert im1.shape == im2.shape
        H,W,C=im1.shape
        im1 = np.reshape(im1,( H*W,C))
        im2 = np.reshape(im2,(H*W,C))
        core=np.multiply(im1, im2)
        mole = np.sum(core, axis=1)
        im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        deno = np.multiply(im1_norm, im2_norm)
        sam = np.rad2deg(np.arccos(((mole+self.eps)/(deno+self.eps)).clip(-1,1)))
        return np.mean(sam)


# if __name__ == '__main__':
#     SAM=Loss_SAM()
#     RMSE=Loss_RMSE()
#     PSNR=Loss_PSNR()
#     psnr_list=[]
#     sam_list=[]
#     sam=AverageMeter()
#     rmse=AverageMeter()
#     psnr=AverageMeter()
#     path = 'D:\LYY\YJX_fusion\model_save\\fusion_model_v9_1\cavee_test/'
#     imglist = os.listdir(path)

#     for i in range(0, len(imglist)):
#         img = loadmat(path + imglist[i])
#         lable = img["rea"]
#         recon = img["fak"]
#         sam_temp=SAM(lable,recon)
#         psnr_temp=PSNR(torch.Tensor(lable), torch.Tensor(recon))
#         sam.update(sam_temp)
#         rmse.update(RMSE(torch.Tensor(lable),torch.Tensor(recon)))
#         psnr.update(psnr_temp)
#         psnr_list.append(psnr_temp)
#         sam_list.append(sam_temp)
#     print(sam.avg)
#     print(rmse.avg)
#     print(psnr.avg)
#     print(psnr_list)
#     print(sam_list)