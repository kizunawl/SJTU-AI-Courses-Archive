import numpy as np
import os
import matplotlib.pyplot as plt

def solve(sample, label, len_vec, taskno):
    def dis(vec):
        return np.sqrt(np.dot(vec, vec))

    epsilon = 1e-3
    n = len(label)
    m = min(n, 10)

    label_KCluster = [[] for _ in range(0, m+1)]
    dis_KCluster = [sum(dis(sample[j]-sample[0]) for j in range(0, n)) for _ in range(0, m+1)]

    perm_cluster_point = []
    perm_cluster_point.append(sample[0])

    vis = np.zeros(n, dtype=bool)
    vis[0] = 1
    # k个类
    for k in range(2, m+1):
        # 选取初始的k个聚点
        clusdis=0
        for i in range(0, n):
            if not vis[i]:
                curdis = min(dis(sample[i]-perm_cluster_point[j]) for j in range(0, k-1))
                if curdis>clusdis:
                    clusdis = curdis
                    tarkey = i
        perm_cluster_point.append(sample[tarkey])
        vis[tarkey] = 1
        # print(k, perm_cluster_point)

        cluster_point = perm_cluster_point

        while (1):
            # 根据原来的聚类，重新划分点的归属
            tmp_label = np.zeros(n, dtype=int)
            tmp_cluster = [[] for _ in range(0, k)]
            for i in range(0, n):
                tmp_key = 0
                tmp_dis = dis(sample[i]-cluster_point[0])
                for j in range(1, k):
                    i_j_dis = dis(sample[i]-cluster_point[j])
                    if i_j_dis<tmp_dis:
                        tmp_key = j
                        tmp_dis = i_j_dis
                tmp_label[i] = tmp_key
                tmp_cluster[tmp_key].append(sample[i])

            # 新划分下的k个聚点
            new_cluster_point = []
            for i in range(0, k):
                c=np.array([0.0 for _ in range(0, len_vec)])
                for j in range(0, len(tmp_cluster[i])):
                    c += tmp_cluster[i][j]
                c /= len(tmp_cluster[i])
                new_cluster_point.append(c)

            # 求新聚类的方差 / 更新k聚类
            tot_dis = sum(dis(sample[i]-new_cluster_point[tmp_label[i]]) for i in range(0, n))

            if dis_KCluster[k]-tot_dis>epsilon:
                dis_KCluster[k] = tot_dis
                cluster_point = new_cluster_point
                label_KCluster[k] = tmp_label
            else:
                break

    # 手肘法确定聚几类
    x = [i for i in range(2, m+1)]
    y = dis_KCluster[2:m+1]
    plt.plot(x, y, "r", marker='o')
    for x1,y1 in zip(x, y):
        plt.text(x1, y1, format(y1, '.2f'), ha='center', va='bottom', fontsize=10)
    plt.xlabel("number of clusters")
    plt.ylabel("total distance")
    plt.title("task{}".format(taskno))
    plt.savefig(os.path.join("figure", "Figure{}.jpg".format(taskno)))
    plt.show()

    decision = int(input())
    return label_KCluster[decision]


def cluster_task_1(sample):
    """
    载入位于data/data1.npy的数据。
    对于data1.npy中给定的1000个一维且Shape为(1)的向量,将它们聚类成两类,一类的向量的ID为0,另一类的向量的ID为1。

    Args:
        sample (np.ndarray): Shape: (1000, )

    Returns:
        label (np.ndarray): Shape: (1000, ), 其中保存的是一维聚类的结果，请保证label与sample的index一致
    """
    label = np.zeros((sample.shape[0]))
    return solve(sample, label, 1, 1)


def cluster_task_2(sample):
    """
    载入位于data/data2.npy的数据。
    对于data2.npy中给定的1000个一维且Shape为(2)的向量,将它们聚类成N类,类别ID 从0至N-1。

    Args:
        sample (np.ndarray): Shape: (M, 2)

    Returns:
        label (np.ndarray): Shape: (M, ), 其中保存的是一维聚类的结果,请保证label与sample的index一致
    """
    label = np.zeros((sample.shape[0]))
    return solve(sample, label, len(sample[0]), 2)


def cluster_task_3(sample):
    """
    载入位于data/data3.npy的数据。
    对于data3.npy中给定的1000个一维且Shape为(128)的向量,将它们聚类成3类,类别ID 从0至2。

    Args:
        sample (np.ndarray): Shape: (M, 128)

    Returns:
        label (np.ndarray): Shape: (M, ), 其中保存的是一维聚类的结果，请保证label与sample的index一致
    """
    label = np.zeros((sample.shape[0]))
    return solve(sample, label, len(sample[0]), 3)


def cluster_task_4(sample):
    """
    载入位于data/data4.npy的数据。
    对于data4.npy中给定 10000 个一维且Shape为(128)的向量,将它们聚类成N类,类别ID 从0至N-1。

    Args:
        sample (np.ndarray): Shape: (M, 128)

    Returns:
        label (np.ndarray): Shape: (M, ), 其中保存的是一维聚类的结果，请保证label与sample的index一致
    """
    label = np.zeros((sample.shape[0]))
    return solve(sample, label, len(sample[0]), 4)


def main():
    data1 = np.load(os.path.join("data", "data1.npy"))
    data2 = np.load(os.path.join("data", "data2.npy"))
    data3 = np.load(os.path.join("data", "data3.npy"))
    data4 = np.load(os.path.join("data", "data4.npy"))

    label1 = cluster_task_1(data1)
    label2 = cluster_task_2(data2)
    label3 = cluster_task_3(data3)
    label4 = cluster_task_4(data4)

    np.save(os.path.join("output", "label1.npy"), label1)
    np.save(os.path.join("output", "label2.npy"), label2)
    np.save(os.path.join("output", "label3.npy"), label3)
    np.save(os.path.join("output", "label4.npy"), label4)


if __name__ == '__main__':
    main()