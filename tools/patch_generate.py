import cv2
# from noise_fearture import
from tools.noise_feature import feature_concat
def noise_save(noise_patch,data,noise_save_name):
    out_dict = dict()
    out_dict['noiseprint'] = noise_patch
    out_dict['QF'] = data['QF']
    if noise_save_name[-4:] == '.mat':
        import scipy.io as sio
        sio.savemat(noise_save_name, out_dict)
    else:
        import numpy as np
        np.savez(noise_save_name, out_dict)



def divide(img_path,noise_path, size,stride):
    noise_all = []
    name_all = []


    img_cv = cv2.imread(img_path)



    if noise_path[-4:] == '.mat':
        import scipy.io as sio
        data = sio.loadmat(noise_path)
    else:
        import numpy as np
        data = np.load(noise_path)
    noise = data['noiseprint']
    qf = data['QF'].flatten()


    feature = feature_concat(img_cv,noise)  #[3,m,n]
    [m, n, c] = img_cv.shape

    m_num = int((m - size) / stride + 1)
    n_num = int((n - size) / stride + 1)


    m_lable = False
    n_lable = False
    m_max = (m_num - 1) * stride + size
    n_max = (n_num - 1) * stride + size
    assert m_max <= m
    assert n_max <= n
    if m_max < m:
        m_lable = True
    if n_max < n:
        n_lable = True

    num = 0
    for i in range(m_num + int(m_lable)):
        for j in range(n_num + int(n_lable)):
            num = num + 1
            ##initialization
            m1 = i * stride
            m2 = i * stride + size
            n1 = j * stride
            n2 = j * stride + size
            if i == m_num:
                m1 = m - size
                m2 = m
            if j == n_num:
                n1 = n - size
                n2 = n



            noise_patch = feature[:,m1:m2, n1:n2]


            name = str(m1) + '_' + str(m2) + '_' + str(n1) + '_' + str(n2)



            noise_all.append(noise_patch)
            name_all.append(name)


    return [noise_all,name_all],qf









def main():
    size = 48
    stride = 16




    # img_path = r'C:\Users\CT\Desktop\autoencoder\data\NC2016_2564\img\NC2016_2564.jpg'
    # noise_path = r'C:\Users\CT\Desktop\autoencoder\data\NC2016_2564\noise\NC2016_2564.mat'
    # gt_path = r'C:\Users\CT\Desktop\autoencoder\data\NC2016_2564\gt\NC2016_2564_gt.png'
    # save_path = r'C:\Users\CT\Desktop\autoencoder\patch\NC2016_2564_s'

    img_path = r"C:\Users\CT\Desktop\testcc\7\1c.png"
    noise_path = r"C:\Users\CT\Desktop\testcc\7\1.mat"
    gt_path = r"C:\Users\CT\Desktop\testcc\7\1_mask.png"
    save_path = r"C:\Users\CT\Desktop\testcc\7"
    divide(img_path,noise_path,gt_path,size,stride)











if __name__ == '__main__':
    main()