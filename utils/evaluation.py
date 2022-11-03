from hmac import compare_digest
import torch
import torch.nn as nn
import numpy as np

from utils.torch_helper import move_to_device, squeeze_dim


def squared_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.sum(diff * diff, -1)


def Indexing(C, N_books, X):
    # X = (bsz, N_books * L_word)
    # C = (N_words, N_books * L_word)
    l1, l2 = C.shape
    L_word = int(l2/N_books)
    # element: (bath_size, L_word); N_books elements
    x = torch.split(X, L_word, 1)
    # element: (N_words, L_word); N_books elements
    y = torch.split(C, L_word, 1)
    for i in range(N_books):
        diff = squared_distances(x[i], y[i])  # (bsz, n_words)
        arg = torch.argmin(diff, dim=1)  # (bsz, )
        min_idx = torch.reshape(arg, [-1, 1])  # (bsz, 1)
        if i == 0:
            quant_idx = min_idx
        else:
            quant_idx = torch.cat((quant_idx, min_idx), dim=1) # (bsz, N_books)
    return quant_idx


def compute_retrieval_precision(train_loader, eval_loader, device,
                                encode_continuous, Codebooks, N_books,
                                num_retrieve=100):
    def extract_target(loader):
        encoding_chunks = []
        label_chunks = []
        for (docs, labels) in loader:
            docs = squeeze_dim(move_to_device(docs, device), dim=1)
            encoding_chunks.append(docs if encode_continuous is None else
                                   encode_continuous(docs))
            label_chunks.append(labels)

        encoding_mat = torch.cat(encoding_chunks, 0)
        label_mat = torch.cat(label_chunks, 0)
        label_lists = [[label_mat[i].item()] for i in range(label_mat.size(0))]
        return encoding_mat, label_lists

    def extract_database_code(loader, Codebooks):
        code_chunks = []
        label_chunks = []
        for (docs, labels) in loader:
            docs = squeeze_dim(move_to_device(docs, device), dim=1)
            encodings = encode_continuous(docs)
            codes = Indexing(Codebooks, N_books, encodings)
            code_chunks.append(codes)

            label_chunks.append(labels)

        code_mat = torch.cat(code_chunks, 0).type(torch.int)
        label_mat = torch.cat(label_chunks, 0)
        label_lists = [[label_mat[i].item()] for i in range(label_mat.size(0))]
        return code_mat, label_lists

    src_codes, src_label_lists = extract_database_code(train_loader, Codebooks)

    import time
    start = time.time()
    tgt_encodings, tgt_label_lists = extract_target(eval_loader)
    prec = compute_topK_average_precision(tgt_encodings, tgt_label_lists,
                                          src_codes, src_label_lists,
                                          Codebooks, N_books,
                                          num_retrieve)
    return prec


def pqDist_one(C, N_books, g_x, q_x):
    """
    The assymetric distance computation refers to SPQ(https://github.com/youngkyunJang/SPQ).
    """

    l1, l2 = C.shape  
    L_word = int(l2/N_books)
    D_C = torch.zeros((l1, N_books), dtype=torch.float32)  # (N_words, N_books)

    q_x_split = torch.split(q_x, L_word, 0)  # tuple of [L_word] element.
    g_x_split = np.split(g_x.cpu().data.numpy(), N_books, 1) # list of [bsz, 1] elements. totally N_books elments.
    C_split = torch.split(C, L_word, 1) # tuple of [N_words, L_word] element. totally N_books elements.
    D_C_split = torch.split(D_C, 1, 1) # tuple of [N_words, 1] elments. totally N_books elements.

    for j in range(N_books):
        for k in range(l1):
            D_C_split[j][k] = torch.norm(q_x_split[j]-C_split[j][k], 2)
        if j == 0:
            dist = D_C_split[j][g_x_split[j]]
        else:
            dist = torch.add(dist, D_C_split[j][g_x_split[j]])
    Dpq = torch.squeeze(dist)
    return Dpq


def pqDist(Codebooks, N_books, src_codes, target_encodings):
    D = []
    for i in range(len(target_encodings)):
        Dbq = pqDist_one(Codebooks, N_books, src_codes, target_encodings[i])
        D.append(Dbq)
    D = torch.stack(D)
    return D


def compute_topK_average_precision(tgt_encodings, tgt_label_lists,
                                   src_codes, src_label_lists,
                                   Codebooks, N_books,
                                   num_retrieve):
    K = min(num_retrieve, len(src_codes))

    import time
    start = time.time()
    D = pqDist(Codebooks, N_books, src_codes, tgt_encodings)

    # D = compute_distance(tgt_encodings, src_encodings, distance_metric,
    #                      chunk_size, binary)

    # Random here in breaking ties (e.g., may have many 0-distance neighbors),
    # but given nontrivial representations this is not an issue (hopefully).
    #
    # TODO: maybe use a stable version of topk when available,
    #   https://github.com/pytorch/pytorch/issues/27542
    _, list_topK_nearest_indices = D.topk(K, dim=1, largest=False)

    average_precision = 0.
    for i, topK_nearest_indices in enumerate(list_topK_nearest_indices):
        gold_set = set(tgt_label_lists[i])
        candidate_lists = [src_label_lists[j] for j in topK_nearest_indices]
        precision = len([_ for candidates in candidate_lists
                         if not gold_set.isdisjoint(candidates)]) / K * 100
        average_precision += precision / tgt_encodings.size(0)

    return average_precision


