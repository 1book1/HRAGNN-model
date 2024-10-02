import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score

cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')

    print("labels_train.shape:",labels_tr.shape)

    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')

    print("labels_teat.shape:",labels_te.shape)

    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)


    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr_n.csv"), delimiter=','))
        print("data_train-:",str(i),"tr_n.csv:",len(data_tr_list[i-1]))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te_n.csv"), delimiter=','))
        print("data_test-:",str(i),"te_n.csv:",len(data_te_list[i-1]))

    num_tr = data_tr_list[0].shape[0]
    print("The sample number of the first training set is：",num_tr)
    num_te = data_te_list[0].shape[0]
    print("The sample size of the first test set is：",num_te)

    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()

    print("Total length of three samples：",len(data_mat_list[0]),len(data_mat_list[1]),len(data_mat_list[2]))

    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))

    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):  #
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))

    return data_train_list, data_all_list, idx_dict, labels

def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance

    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):

        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))

    return adj_train_list, adj_test_list


def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, g, train_MVFN=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list, i))
        ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i+1)].step()

        #ReduceLROnPlateau
        ci_accuracy = calculate_accuracy(ci, label)
        scheduler = ReduceLROnPlateau(optim_dict["C{:}".format(i+1)], mode='max', factor=0.1, patience=10, threshold=1e-3,
                                       threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08)
        scheduler.step(ci_accuracy)

        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()

    if train_MVFN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list, i)))
        c = model_dict["C"](ci_list)
        c_loss = torch.mean(torch.mul(criterion(c, label),sample_weight))
        c_loss.backward()
        optim_dict["C"].step()

        #ReduceLROnPlateau
        c_accuracy = calculate_accuracy(c, label)
        scheduler = ReduceLROnPlateau(optim_dict["C"], mode='max', factor=0.1, patience=10, threshold=1e-3,
                                       threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08)
        scheduler.step(c_accuracy)

        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    #print(g,"+loss：", loss_dict)

    return loss_dict

def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list, i)))
    if num_view >= 2:
        c = model_dict["C"](ci_list)
    else:
        c = ci_list[0]
    c = c[te_idx,:]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    return prob

def max(last,max,epoch_last,epoch):
    if max >= last:
        last = max
        epoch_last = epoch
    return last,epoch_last


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c,
               num_epoch_pretrain, num_epoch):
    test_inverval = 1
    num_view = len(view_list)
    dim_hmvfn = pow(num_class,num_view)
    adj_parameter = 10
    dim_he_list = [512,512,256]
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()

    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    print("The characteristic dimensions of the three omics data are respectively :",dim_list)

    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hmvfn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()

    print("\nPrepare HRAGNNs...")
    ####################################################################################
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):#500

        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, epoch, train_MVFN=False)
    #######################################################################################

    print("\nStart Training...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)

    #------------------------------------#
    last = 0
    epoch_last = 0
    last = format(last,'.3f')
    epoch_last = format(epoch_last)
    g = 0
    h = 0
    #------------------------------------#


    for epoch in range(num_epoch+1):   #3500
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, epoch)

        if epoch % test_inverval == 0:

            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)

            print("\nTest: Epoch {:d}".format(epoch))

            if num_class == 2:   #If it's binary
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
            else:
                epoch = format(epoch)
                maxx = format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1)))


                #ACC:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))

                #Precision:
                predicted_labels = np.argmax(te_prob, axis=1)
                precision = precision_score(labels_trte[trte_idx["te"]], predicted_labels,zero_division=0, average='macro')
                print("Test PRE: {:.3f}".format(precision))

                #Recall:
                predicted_labels = np.argmax(te_prob, axis=1)
                recall = recall_score(labels_trte[trte_idx["te"]], predicted_labels, zero_division=0, average='macro')
                print("Test Recall: {:.3f}".format(recall))

                #F1 Macro:
                #print("Test F1 macro: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))

                #F1 Score:
                f1_scores = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1),zero_division=0, average=None)
                overall_f1 = f1_scores.mean()
                print("Test F1 Score: {:.3f}".format(float(overall_f1)))

                last,epoch_last = max(last, maxx, epoch_last, epoch)
                print("Max Epoch:",epoch_last)
                print("Max:",last)



            print()
    # print("Epoch of Max ACC:",epoch_last)
    # print("The Max ACC:",last)

