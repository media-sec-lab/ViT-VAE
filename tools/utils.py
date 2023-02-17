import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import *
import cv2
import imutils
from tools.metric import *
erodeKernSize  = 15
dilateKernSize = 11

def valid(dataloader,img_path,mask_path,args,optimizer,net,results):

    anomaly_score_total_list = []
    m1_all = []
    m2_all = []
    n1_all = []
    n2_all = []

    img_cv = cv2.imread(img_path)
    [M,N,c] = img_cv.shape

    net.eval()
    with torch.no_grad():

        for batchidx,(noise,m1,m2,n1,n2) in enumerate(dataloader):

            noise = noise.float().cuda()
            # noise = torch.unsqueeze(noise, dim=1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            re_noise, kld= net(noise)

            bachsize = re_noise.size(0)
            re_noise_f1 = re_noise[:, 0, :, :].reshape(bachsize, -1)
            noise_f1 = noise[:, 0, :, :].reshape(bachsize, -1)
            scores1 = torch.mean((re_noise_f1 - noise_f1) ** 2, dim=1)

            re_noise_f2 = re_noise[:, 1, :, :].reshape(bachsize, -1)
            noise_f2 = noise[:, 1, :, :].reshape(bachsize, -1)
            scores2 = torch.mean((re_noise_f2 - noise_f2) ** 2, dim=1)

            re_noise_f3 = re_noise[:, 2, :, :].reshape(bachsize, -1)
            noise_f3 = noise[:, 2, :, :].reshape(bachsize, -1)
            scores3 = torch.mean((re_noise_f3 - noise_f3) ** 2, dim=1)

            scores, _ = torch.max(
                torch.cat([scores1.unsqueeze(dim=1), scores2.unsqueeze(dim=1), scores3.unsqueeze(dim=1)], dim=1),
                dim=1)



            anomaly_score_total_list.extend(scores.cpu().numpy())
            m1_all.extend(m1)
            m2_all.extend(m2)
            n1_all.extend(n1)
            n2_all.extend(n2)



        img_save = np.zeros([M, N])
        time_save = np.zeros([M, N])
        k_size = len(m1_all)
        for i in range(k_size):
            score = anomaly_score_total_list[i]

            m1 = m1_all[i]
            m2 = m2_all[i]
            n1 = n1_all[i]
            n2 = n2_all[i]

            img_save[m1:m2, n1:n2] = img_save[m1:m2, n1:n2] + score
            time_save[m1:m2, n1:n2] = time_save[m1:m2, n1:n2] + 1

        map = np.divide(img_save,time_save)



        threshold = args.threshold
        vmax = np.max(map)
        vmin = np.min(map)
        map = (map - vmin) / (vmax - vmin)
        map_threshold0_5 = (map > threshold).astype(np.uint8)

        #
        # plt.figure()
        # plt.title('Heat_map')
        # plt.imshow(imutils.opencv2matplotlib(img_cv), interpolation='none')
        # plt.imshow(map, cmap='jet', alpha=0.5, interpolation='none')
        # plt.show()



        if (mask_path != ''):
            gt_cv = cv2.imread(mask_path, flags=2)/255


            [M1, N1] = map.shape
            [M2, N2] = gt_cv.shape
            if (M1 == M2 and N1 == N2):
                print('ok')
            else:
                gt_cv = cv2.resize(gt_cv, (N1, M1))

            gt_cv = (np.asarray(gt_cv) > 0.5).astype(np.int)





            f1 = metrics.f1_score(gt_cv.flatten().astype(int), map_threshold0_5.flatten())
            iou = iou_measure(gt_cv.flatten().astype(int), map_threshold0_5.flatten())
            auc = metrics.roc_auc_score(gt_cv.flatten().astype(int), map.flatten())





            print('AUC: %6f F1: %.6f  IoU: %.6f' % (auc,f1,iou))
            return f1, iou,auc
        else:
            return map,map_threshold0_5,img_cv


def train(dataloader, epoch,args,optimizer,net,min_loss):

    criterion = nn.MSELoss()

    running_loss = 0.0
    loss_all = 0.0
    batch_all = 0
    for batchidx,noise in enumerate(dataloader):

        noise = noise.float().cuda()
        batchsz = noise.size(0)
        if batchsz == 1:
            continue
        # noise = torch.unsqueeze(noise, dim=1)


        # zero the parameter gradients
        optimizer.zero_grad()


        # forward + backward + optimize
        re_noise,kld = net(noise)
        loss = criterion(re_noise, noise)
        elbo = - loss - 1.0 * kld
        loss = - elbo

        # loss = net(noise-0.5)



        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        batch_all += 1
        # print statistics
        running_loss += loss.item()



    mean_loss = loss_all/batch_all


    if min_loss - mean_loss < args.stop_loss:
        return 1,min_loss
    elif mean_loss < min_loss:
        min_loss = mean_loss
        return 0,min_loss