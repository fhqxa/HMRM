import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np

'''Calculate HMRM Loss'''
def ContrastiveLoss_Hierarchical_Cost(batch_predict_lables, batch_real_lables, train_coarse_labels, fine_relations_max, logits_scores_max, CLASS_NUM, BATCH_NUM_PER_CLASS, Gamma , probablity, theat):
   
  
    loss = torch.zeros(1, requires_grad=True)

    batch_real_lables = batch_real_lables.tolist()
    batch_predict_lables = batch_predict_lables.tolist()
    fine_relations_max = fine_relations_max.tolist()
    logits_scores_max = logits_scores_max.tolist()
    Hierarchical_Cost = 1.0
    Ground_score_cost = 1.0

    for s in range(CLASS_NUM):
        Upper_loss = torch.zeros(1) 
        Down_loss = torch.zeros(1) 
        upper_num = 0 
        down_num = 0
    
        for i in range(CLASS_NUM*BATCH_NUM_PER_CLASS):
            if batch_predict_lables[i] == s:
                if  batch_real_lables[i] == s:
                    Upper_loss = Upper_loss + torch.exp(torch.tensor(fine_relations_max[i]))
                    upper_num = upper_num + 1
                else:
                    i_down_loss = torch.exp(torch.tensor(fine_relations_max[i]))
                    if (train_coarse_labels[s] != train_coarse_labels[CLASS_NUM + i]):
                        Hierarchical_Cost = Hierarchical_Cost * Gamma

                    if (logits_scores_max[i] > probablity):
                        Ground_score_cost = torch.exp((-theat) * torch.tensor((1 + (probablity-0.5)-logits_scores_max[i])/(logits_scores_max[i]-(probablity-0.5))))
                    else:
                        Ground_score_cost = 1.0

                    i_down_loss = i_down_loss * Hierarchical_Cost * Ground_score_cost
                    Down_loss = Down_loss + i_down_loss
                    down_num = down_num + 1

        if Upper_loss.item() == 0:
            loss = loss
        elif Down_loss.item() == 0:
            loss = loss
        else:
            down = torch.log(torch.tensor(Down_loss.item()))
            loss = loss + torch.div(down, down_num)

    return loss

def Visual(feat):

    def visual(feat):
        # t-SNE的最终结果的降维与可视化
        ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
        x_ts = ts.fit_transform(feat)
        print(x_ts.shape)  # [num, 2]
        x_min, x_max = x_ts.min(0), x_ts.max(0)
        x_final = (x_ts - x_min) / (x_max - x_min)
        return x_final
        # 设置散点形状

    maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    # 设置散点颜色
    colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
              'hotpink']
    # 图例名称
    Label_Com = ['a', 'b', 'c', 'd']
    # 设置字体格式
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 32,
             }

    def plotlabels(S_lowDWeights, Trure_labels, name):
        True_labels = Trure_labels.reshape((-1, 1))
        S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
        S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
        print(S_data)
        print(S_data.shape)  # [num, 3]

        for index in range(3):  # 假设总共有三个类别，类别的表示为0,1,2
            X = S_data.loc[S_data['label'] == index]['x']
            Y = S_data.loc[S_data['label'] == index]['y']
            plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65)

            plt.xticks([])  # 去掉横坐标值
            plt.yticks([])  # 去掉纵坐标值

        plt.title(name, fontsize=32, fontweight='normal', pad=20)


    feat = torch.rand(128, 1024)  # 128个特征，每个特征的维度为1024
    label_test1 = [0 for index in range(40)]
    label_test2 = [1 for index in range(40)]
    label_test3 = [2 for index in range(48)]

    label_test = np.array(label_test1 + label_test2 + label_test3)
    print(label_test)
    print(label_test.shape)

    fig = plt.figure(figsize=(10, 10))

    plotlabels(visual(feat), label_test, '(a)')

    plt.show(fig)
