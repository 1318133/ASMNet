from scipy.io import loadmat

from calculate_metrics import Loss_SAM, Loss_RMSE, Loss_PSNR
from net.multiscale import allnet


# from models.HSRnet import RGBNet
from train_dataloader import *
from torch import nn
from tqdm import tqdm
import time
import pandas as pd
import torch.utils.data as data
from utils import create_F, fspecial
import math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

l1loss = nn.L1Loss(reduction='mean').cuda()

class spatial_edge(nn.Module):
    def __init__(self,):
        super(spatial_edge, self).__init__()

    def forward(self, x):
     edge1 = x[:, :, 0:x.size(2)-1, :] - x[:, :, 1:x.size(2), :]
     edge2 = x[:, :, :, 0:x.size(3)-1] - x[:, :,  :, 1:x.size(3)]
     return edge1, edge2

class spectral_edge(nn.Module):
    def __init__(self,):
        super(spectral_edge, self).__init__()

    def forward(self, x):
        edge = x[:, 0:x.size(1) - 1, :, :] - x[:, 1:x.size(1), :, :]

        return edge



def mkdir(path):

    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("训练文件夹为：{}".format(path))
    else:
        print('已存在{}'.format(path))

def loss_att(a,b,c): #hsi,GT,predict
    at=0
    HW = a.size(2)*a.size(3)
    ar=torch.reshape(a,(a.size(0),a.size(1),a.size(2)*a.size(3)))
    HW = int(HW)
    aball=torch.zeros(b.size(0),1,b.size(2),b.size(3)).cuda()
    for i in range(HW):
        at+=1
        arn=ar[:,:,i]
        arn=arn.unsqueeze(-1)
        aexp=arn.expand(b.size(0),b.size(1),b.size(2)*b.size(3))
        ares=torch.reshape(aexp,(b.size(0),b.size(1),b.size(2),b.size(3)))
        abssum=torch.sum(torch.abs(ares-b),dim=1)
        abssumsq=abssum.unsqueeze(1)
        abssumsq_g1=(abssumsq-abssumsq.min())/(abssumsq.max()-abssumsq.min()+0.0000001)
        # abssumsq_g1_sig=torch.sigmoid((abssumsq_g1-0.5)*5)
        aball=aball+abssumsq_g1
    # aatmap=(aball-aball.min())/(aball.max()-aball.min()+0.0000001)
    aatmap=aball/HW
    out=l1loss(b*aatmap,c*aatmap)*l1loss(b,c)
    return out


if __name__ == '__main__':
    # 路径参数
    root=os.getcwd()+"/fusion_code/model"
    model_name='stage1_loss1_v5'
    mkdir(os.path.join(root,model_name))
    current_list=os.listdir(os.path.join(root,model_name))
    for i in current_list:
        if len(i)>1:
            current_list.remove(i)

    current_list.append('0')
    int_list = [int(x) for x in current_list]
    train_value = max(int_list)+1
    model_name=os.path.join(model_name,str(train_value))

    
    # data_name='NTIRE2022'
    # path1 = '/home/s1u1/dataset/NTIRE2022/Train_spectral/'
    # path2 = '/home/s1u1/dataset/NTIRE2022/Valid_spectral/'

    # data_name='harvard'
    # path1 = 'D:\data\Harvard\harvard_train/'
    # path2 = 'D:\data\Harvard\harvard_test/'

    data_name='cave'
    path1 = '/home/s1u1/dataset/cave/cave_train/'
    path2 = '/home/s1u1/dataset/cave/cave_test/'
    # kj=spatial_edge().cuda()
    # gp=spectral_edge().cuda()
    # 训练参数
    #loss_func = nn.MSELoss().cuda()
    loss_func = nn.L1Loss(reduction='mean').cuda()
    R = create_F()
    PSF = fspecial('gaussian', 8, 3)
    downsample_factor = 8
    training_size = 64
    stride = 32
    stride1 = 32
    LR = 4e-4
    EPOCH = 500
    weight_decay=1e-8
    BATCH_SIZE = 16
    num = 20
    psnr_optimal = 47
    rmse_optimal = 1.5

    test_epoch=0
    val_interval = 50           # 每隔val_interval epoch测试一次
    checkpoint_interval = 100
    maxiteration = math.ceil(((512 - training_size) // stride + 1) ** 2 * num / BATCH_SIZE) * EPOCH
    # maxiteration = math.ceil(
    #      ((1040 - training_size) // stride + 1) * ((1392 - training_size) // stride + 1) * num / BATCH_SIZE) * EPOCH
    print("maxiteration：", maxiteration)

    # warm_lr_scheduler
    decay_power = 1.5
    init_lr2 = 2e-4
    init_lr1 = 2e-4 / 10
    min_lr=0
    warm_iter = math.floor(maxiteration / 40)

    # 创建方法名字的文件
    path=os.path.join(root,model_name)
    mkdir(path)  # 创建文件夹
    # 创建训练记录文件
    pkl_name=data_name+'_pkl'
    pkl_path=os.path.join(path,pkl_name)      # 模型保存路径
    os.makedirs(pkl_path)      # 创建文件夹
    # 创建excel
    df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss','val_loss','val_rmse', 'val_psnr', 'val_sam'])  # 列名
    excel_name=data_name+'_record.csv'
    excel_path=os.path.join(path,excel_name)
    df.to_csv(excel_path, index=False)


    train_data = CAVEHSIDATAprocess(path1, R, training_size, stride, downsample_factor, PSF, num)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # cnn = Cross_Guide_Fusion(31,training_size,training_size,1).cuda()
    # cnn =FusionNet().cuda()
    # cnn=VSR_CAS(channel0=31, factor=8, P=torch.Tensor(R), patch_size=training_size).cuda()
    # cnn=RGBNet().cuda()
    cnn = allnet(48).cuda()
    # 模型初始化
    for m in cnn.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    # optimizer = torch.optim.Adam([{'params': cnn.parameters(), 'initial_lr': 1e-1}], lr=LR,betas=(0.2, 0.999),weight_decay=weight_decay)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR,betas=(0.9, 0.999),weight_decay=1e-8)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxiteration, eta_min=1e-6)

    start_epoch = 0
    # resume = True
    resume = False

    # step = start_step
    step=0   # warm_lr_scheduler要用
    for epoch in range(start_epoch+1, EPOCH+1):
        cnn.train()
        loss_all1 = []
        loss_all2 = []
        loss_all3 = []
        loop = tqdm(train_loader, total=len(train_loader))


        #for epoch_step, (a1, a2,a3) in enumerate(loop):
        for a1, a2, a3 in loop:
            lr = optimizer.param_groups[0]['lr']
            step = step + 1
            output,output1,output2 = cnn(a3.cuda(),a2.cuda())

            loss = loss_func(output,a1.cuda())
            # lossatt = loss_att_v3(a3.cuda(),a1.cuda(),output)
            loss_temp = loss#+10*lossatt
            loss_all1.append(np.array(loss.detach().cpu().numpy()))
            # loss_all2.append(np.array(lossatt.detach().cpu().numpy()))


            optimizer.zero_grad()
            loss_temp.backward()

            #torch.nn.utils.clip_grad_value_(cnn.parameters(), clip_value=0.1)
            optimizer.step()
            scheduler.step()
            #loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
            loop.set_description('epoch:{}  lr:{}  loss1:{} loss2:{}'.format(epoch + 1, lr, np.mean(loss_all1), np.mean(loss_all2)))

            #loop.set_postfix({'loss': '{0:1.8f}'.format(np.mean(loss_all)), "lr": '{0:1.8f}'.format(lr)})

        # scheduler.step()





        if ((epoch % val_interval == 0) and (epoch>=test_epoch) ) or epoch==1:
            cnn.eval()
            val_loss=AverageMeter()
            SAM = Loss_SAM()
            RMSE = Loss_RMSE()
            PSNR = Loss_PSNR()
            sam = AverageMeter()
            rmse = AverageMeter()
            psnr = AverageMeter()

            imglist = os.listdir(path2)
            with torch.no_grad():
                for i in range(0, len(imglist)):
                    img = loadmat(path2 + imglist[i])
                    img1 = img["b"]
                    # img1 = img1 / img1.max()
                    HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))
                    MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
                    HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
                    MSI_1 = torch.unsqueeze(MSI, 0)
                    HSI_LR1 = torch.unsqueeze(torch.Tensor(HSI_LR), 0)  # 加维度 (b,c,h,w)
                    # 计算val_loss用的，防止出错单独拿出来
                    to_fet_loss_hr_hsi=torch.unsqueeze(torch.Tensor(HRHSI), 0)

                    prediction,val_loss = reconstruction(cnn, R, HSI_LR1.cuda(), MSI_1.cuda(), to_fet_loss_hr_hsi,downsample_factor, training_size, stride1,val_loss)
                    # print(Fuse.shape)
                    sam.update(SAM(np.transpose(HRHSI.cpu().detach().numpy(),(1, 2, 0)),np.transpose(prediction.squeeze().cpu().detach().numpy(),(1, 2, 0))))
                    rmse.update(RMSE(HRHSI.cpu().permute(1,2,0),prediction.squeeze().cpu().permute(1,2,0)))
                    psnr.update(PSNR(HRHSI.cpu().permute(1,2,0),prediction.squeeze().cpu().permute(1,2,0)))

                if  epoch == EPOCH:
                    torch.save(cnn.state_dict(),pkl_path +'/'+ str(epoch) + 'EPOCH' + '_PSNR_best.pkl')

                if torch.abs(psnr_optimal-psnr.avg)<0.15:
                    torch.save(cnn.state_dict(), pkl_path + '/' + str(epoch) + 'EPOCH' + '_PSNR_best.pkl')
                if psnr.avg > psnr_optimal:
                    psnr_optimal = psnr.avg

                if torch.abs(rmse.avg-rmse_optimal)<0.15:
                    torch.save(cnn.state_dict(),pkl_path +'/'+ str(epoch) + 'EPOCH' + '_RMSE_best.pkl')
                if rmse.avg < rmse_optimal:
                    rmse_optimal = rmse.avg



                print("val  PSNR:",psnr.avg.cpu().detach().numpy(), "  RMSE:", rmse.avg.cpu().detach().numpy(), "  SAM:", sam.avg,"val loss:", val_loss.avg.cpu().detach().numpy())
                val_list = [epoch, lr,np.mean(loss_all1),np.mean(loss_all2),val_loss.avg.cpu().detach().numpy(),rmse.avg.cpu().detach().numpy(), psnr.avg.cpu().detach().numpy(), sam.avg]

                val_data = pd.DataFrame([val_list])
                val_data.to_csv(excel_path,mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
                time.sleep(0.1)


