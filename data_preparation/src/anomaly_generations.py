import numpy as np
import pandas as pd
import random
import copy
from tqdm import tqdm


def swap(df1, df2, anomaly_rate):
    # Choose swapped indices
    n_rows = df1.shape[0]
    n_anomalies = int(n_rows * anomaly_rate)
    indices1 = random.choices(df1.index, k=n_anomalies)
    indices2 = random.choices(df2.index, k=n_anomalies)
    for i, j in zip(indices1, indices2):
        tmp = df1.iloc[i].copy()
        df1.iloc[i] = df2.iloc[j]
        df2.iloc[j] = tmp
    # Construct ground truths
    gt1 = np.zeros((n_rows, 1), dtype=int)
    gt2 = np.zeros((n_rows, 1), dtype=int)
    gt1[indices1] = 1
    gt2[indices2] = 1
    gt_df1 = pd.DataFrame(gt1, columns=["is_anomaly"])
    gt_df2 = pd.DataFrame(gt2, columns=["is_anomaly"])
    return gt_df1, gt_df2


# Regardless subjects
def divide_groups(instances, clusters):
    group1 = []
    group2 = []
    for a in clusters:
        print(a)
        a_ps = [instance for instance in instances if a in instance]
        selected_aps = list(np.random.choice(a_ps, size=len(a_ps) // 2, replace=False))
        for ap in selected_aps:
            a_ps.remove(ap)
        print(selected_aps)
        print(a_ps)
        group1 += selected_aps
        group2 += a_ps
    print(group1)
    print(group2)
    return group1, group2


def check_align(lst1, lst2):
    for e1, e2 in zip(lst1, lst2):
        if e1 == e2:
            return True
    return False


def pair_instances_2(instances, clusters):
    n = (len(instances) // len(clusters)) // 2
    group1, group2 = [], []

    clusters1, clusters2 = copy.deepcopy(clusters), copy.deepcopy(clusters)
    while check_align(clusters1, clusters2):
        random.shuffle(clusters1)
        random.shuffle(clusters2)

    for c1, c2 in zip(clusters1, clusters2):
        tmp1 = [instance for instance in instances if c1 in instance]
        aps1 = list(np.random.choice(tmp1, size=n, replace=False))
        tmp2 = [instance for instance in instances if c2 in instance]
        aps2 = list(np.random.choice(tmp2, size=n, replace=False))
        group1 += aps1
        group2 += aps2
        for ap in aps1 + aps2:
            instances.remove(ap)
    pairs = []
    for ap1, ap2 in zip(group1, group2):
        pairs.append([ap1, ap2])
    return pairs


def pair_instances(instances, clusters):
    group1, group2 = divide_groups(instances, clusters)
    pairs = []
    for instance in group1:
        aps2 = [ap for ap in group2 if instance[:3] not in ap]
        peer = random.choice(aps2)
        pairs.append([instance, peer])
        group2.remove(peer)
    print(pairs)
    return pairs


# --------------------


# Same subject
# def pair_instances_same_subject(instances, nb_subjects, clusters):
#     pairs = []
#     group1 = sorted(
#         [instance for instance in instances if f"{clusters[0]}" in instance]
#     )
#     group2 = sorted(
#         [instance for instance in instances if f"{clusters[1]}" in instance]
#     )
#     for p in range(len(nb_subjects)):
#         pairs.append([group1[p], group2[p]])
#     # print(pairs)
#     return pairs


def pair_instances_same_subject_with_replacement(instances, clusters):
    pairs = []
    for instance in instances:
        other_clusters = [c for c in clusters if c not in instance]
        selected_cluster = random.choice(other_clusters)
        peer = selected_cluster + "_" + instance.split("_")[-1]
        pairs.append([instance, peer])
    # print(pairs)
    return pairs

# def pair_instances_with_replacement_charsense(instances, clusters):
#     pairs = []
#     for instance in instances:
#         other_clusters = [c for c in clusters if c not in instance]
#         selected_cluster = random.choice(other_clusters)
#         selected_subject = random.choice([i.split("__")[-1] for i in instances if selected_cluster == i.split("__")[0]])
#         peer = selected_cluster + "__" + selected_subject
#         pairs.append([instance, peer])
#     print(pairs)
#     return pairs


def replace_pair(views_dfs, ap1, ap2, anomaly_rate):
    # Choose swapped indices
    n_rows = views_dfs["view_1"][ap1].shape[0]
    n_anomalies = int(n_rows * anomaly_rate)
    indices1 = random.choices(views_dfs["view_1"][ap1].index, k=n_anomalies)
    indices2 = random.choices(views_dfs["view_1"][ap2].index, k=n_anomalies)
    for i, j in zip(indices1, indices2):
        # Choose a swapped view
        swapped_view = random.choice(list(views_dfs.keys()))
        views_dfs[swapped_view][ap1].iloc[i] = (
            views_dfs[swapped_view][ap2].iloc[j].copy()
        )
    # Construct ground truths
    gt1 = np.zeros((n_rows, 1), dtype=int)
    gt1[indices1] = 1
    gt_df1 = pd.DataFrame(gt1, columns=["is_anomaly"])
    return gt_df1


def swap_time_steps(views_dfs, clusters, anomaly_rate):
    # Swap time steps of two instances from different activities/clusters
    print("Generating anomalies...")
    # Initialize
    ground_truths = {}
    # Choose pair of swapped instances
    instances = list(views_dfs["view_1"].keys())
    pairs = pair_instances_same_subject_with_replacement(instances, clusters)
    # Swapp time steps
    for ap1, ap2 in tqdm(pairs):
        # print(ap1, ap2)
        gt_df = replace_pair(views_dfs, ap1, ap2, anomaly_rate)
        # gt_df = replace_internal_timeseries(views_dfs, ap1, ap2, anomaly_rate)
        ground_truths[ap1] = gt_df
    return views_dfs, ground_truths

