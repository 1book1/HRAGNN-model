from train_test import train_test

if __name__ == "__main__":    
    data_folder = 'F:\deboke桌面\最终版代码\BRCA(normalize)'
    view_list = [1,2,3]
    num_epoch_pretrain = 500
    num_epoch = 2000
    lr_e_pretrain = 8e-4               #BRCA:8e-4  GBM:1e-4
    lr_e = 5e-4
    lr_c = 5e-4                        #BRCA:5e-4
    num_class = 4
    train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c,
               num_epoch_pretrain, num_epoch)
