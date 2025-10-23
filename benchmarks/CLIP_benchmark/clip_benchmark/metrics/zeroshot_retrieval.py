import logging
from contextlib import suppress

import os
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
def cal_sim(vector_0, vector_1):
    '''
    Calculate the cos sim and pairwise distance
    :param vector_0:
    :param vector_1:
    :return: cos_sim, pair_dis
    '''
    cos_sim_f = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    pair_dis_f = torch.nn.PairwiseDistance(p=2)
    cos_sim = cos_sim_f(vector_0, vector_1)
    pair_dis = pair_dis_f(vector_0, vector_1)
    return cos_sim, pair_dis



DEFAULT_RECALLS: Sequence[int] = (1, 5, 10)


def _summary(values: Iterable[float]) -> Dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float32)
    if tensor.numel() == 0:
        return {"mean": 0.0, "std": 0.0}
    if tensor.numel() == 1:
        return {"mean": tensor.item(), "std": 0.0}
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
    }


def _distribution(
    values: Iterable[float],
    *,
    thresholds: Sequence[float] = (0.8, 0.9, 0.95),
    mode: str = "ge",
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
) -> Dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float32)
    if tensor.numel() == 0:
        return {}

    stats: Dict[str, float] = {
        "mean": tensor.mean().item(),
        "std": (tensor.std(unbiased=False).item() if tensor.numel() > 1 else 0.0),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
    }

    if quantiles:
        for q in quantiles:
            stats[f"quantile@{q:.2f}"] = tensor.quantile(q).item()

    if thresholds:
        for thr in thresholds:
            key = f"frac_{mode}_{thr:.2f}"
            if mode == "ge":
                stats[key] = float((tensor >= thr).float().mean().item())
            elif mode == "le":
                stats[key] = float((tensor <= thr).float().mean().item())
            else:
                raise ValueError(f"Unsupported mode '{mode}' for distribution summary")

    return stats


def average_precision(scores: torch.Tensor, positive_pairs: torch.Tensor) -> torch.Tensor:
    """Compute per-query average precision."""

    # Sort indices for each query (row-wise) in descending order.
    sorted_indices = torch.argsort(scores, dim=1, descending=True)
    sorted_positive = torch.gather(positive_pairs, 1, sorted_indices)

    cumulative_hits = sorted_positive.float().cumsum(dim=1)
    ranks = torch.arange(1, scores.size(1) + 1, device=scores.device, dtype=torch.float32)
    precision = cumulative_hits / ranks

    total_positive = sorted_positive.sum(dim=1)
    safe_total = torch.where(total_positive == 0, torch.ones_like(total_positive), total_positive)
    ap = (precision * sorted_positive.float()).sum(dim=1) / safe_total
    ap = ap.masked_fill(total_positive == 0, 0.0)
    return ap


def evaluate(model, dataloader, tokenizer, device, watermark_dir, watermark_dim, trigger_num, amp=True, recall_k_list=[5]):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    recall_k_list: list of int
        recall@k k's to use
    
    Returns
    -------
    
    dict of retrieval metrics
    """
    # watermark
    Trigger_mat_pth = os.path.join(watermark_dir, str(watermark_dim), "trigger_mat_%d.pth" % trigger_num)
    Trigger_mat = torch.load(Trigger_mat_pth)
    Trigger_mat = Trigger_mat.to(device)
    print("Loaded Trigger matrix from", Trigger_mat_pth)
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []

    # watermark verification
    wm_cos_image: List[float] = []
    wm_l2_image: List[float] = []
    rev_cos_image: List[float] = []
    rev_l2_image: List[float] = []
    wm_cos_text: List[float] = []
    wm_l2_text: List[float] = []
    rev_cos_text: List[float] = []
    rev_l2_text: List[float] = []

    dataloader = dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress
    for batch_images, batch_texts, inds in tqdm(dataloader):
        batch_images = batch_images.to(device)
        # tokenize all texts in the batch
        batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        # compute the embedding of images and texts
        with torch.no_grad(), autocast():

            image_features = model.encode_image(batch_images)
            text_features = model.encode_text(batch_texts_tok)

            origin_image_features = image_features
            origin_text_features = text_features

            # watermark
            image_features = origin_image_features @ Trigger_mat
            text_features = origin_text_features @ Trigger_mat

            batch_images_emb = F.normalize(image_features, dim=-1)
            batch_texts_emb = F.normalize(text_features, dim=-1)

            # Verification
            if origin_image_features.shape[0] == origin_text_features.shape[0]:
                origin_cos_sim, origin_pair_dis = cal_sim(origin_image_features, origin_text_features)
                Trigger_cos_sim, Trigger_pair_dis = cal_sim(batch_images_emb, batch_texts_emb)

                print("Origin: cos similarity: %lf, pair distance: %lf" % (float(origin_cos_sim.mean()), float(origin_pair_dis.mean())))
                print("Trigger_mat: cos similarity: %lf, pair distance: %lf" % (float(Trigger_cos_sim.mean()), float(Trigger_pair_dis.mean())))


            print("Origin x_p and x_o:")
            Trigger_cos_sim_img, Trigger_pair_dis_img = cal_sim(origin_image_features, image_features)
            Trigger_cos_sim_txt, Trigger_pair_dis_txt = cal_sim(origin_text_features, text_features)

            wm_cos_image.extend(Trigger_cos_sim_img.detach().cpu().tolist())
            wm_l2_image.extend(Trigger_pair_dis_img.detach().cpu().tolist())
            wm_cos_text.extend(Trigger_cos_sim_txt.detach().cpu().tolist())
            wm_l2_text.extend(Trigger_pair_dis_txt.detach().cpu().tolist())

            Trigger_cos_sim = (float(Trigger_cos_sim_img.mean()) + float(Trigger_cos_sim_txt.mean())) / 2
            Trigger_pair_dis = (float(Trigger_pair_dis_img.mean()) + float(Trigger_pair_dis_txt.mean())) / 2

            print("Trigger_Verification: cos similarity: %lf, pair distance: %lf" % (
                float(Trigger_cos_sim), float(Trigger_pair_dis)))

            print("Revised x_p and x_o:")
            Rvised_image_features = image_features @ torch.linalg.inv(Trigger_mat)
            Rvised_text_features = text_features @ torch.linalg.inv(Trigger_mat)

            Trigger_cos_sim_img, Trigger_pair_dis_img = cal_sim(origin_image_features, Rvised_image_features)
            Trigger_cos_sim_txt, Trigger_pair_dis_txt = cal_sim(origin_text_features, Rvised_text_features)
            rev_cos_image.extend(Trigger_cos_sim_img.detach().cpu().tolist())
            rev_l2_image.extend(Trigger_pair_dis_img.detach().cpu().tolist())
            rev_cos_text.extend(Trigger_cos_sim_txt.detach().cpu().tolist())
            rev_l2_text.extend(Trigger_pair_dis_txt.detach().cpu().tolist())

            Trigger_cos_sim = (float(Trigger_cos_sim_img.mean()) + float(Trigger_cos_sim_txt.mean())) / 2
            Trigger_pair_dis = (float(Trigger_pair_dis_img.mean()) + float(Trigger_pair_dis_txt.mean())) / 2
            print("Trigger_Verification: cos similarity: %lf, pair distance: %lf" % (
            float(Trigger_cos_sim), float(Trigger_pair_dis)))

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)

    print("Total Verification:")
    print("Origin x_p1 and x_o1:")
    wm_cos_all = wm_cos_image + wm_cos_text
    wm_l2_all = wm_l2_image + wm_l2_text
    if wm_cos_all:
        wm_cos_mean = sum(wm_cos_all) / len(wm_cos_all)
        wm_l2_mean = sum(wm_l2_all) / len(wm_l2_all)
    else:
        wm_cos_mean = wm_l2_mean = 0.0
    print("Trigger_Verification: cos similarity: %lf, pair distance: %lf" % (
        wm_cos_mean, wm_l2_mean))
    print("Revised x_p1 and x_o1:")
    rev_cos_all = rev_cos_image + rev_cos_text
    rev_l2_all = rev_l2_image + rev_l2_text
    if rev_cos_all:
        rev_cos_mean = sum(rev_cos_all) / len(rev_cos_all)
        rev_l2_mean = sum(rev_l2_all) / len(rev_l2_all)
    else:
        rev_cos_mean = rev_l2_mean = 0.0
    print("Trigger_Verification: cos similarity: %lf, pair distance: %lf" % (
        rev_cos_mean, rev_l2_mean))

    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # get the score for each text and image pair
    scores  = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    recall_k_list = sorted(set(recall_k_list).union(DEFAULT_RECALLS))
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()

    text_ap = batchify(average_precision, scores, positive_pairs, batch_size, device)
    image_ap = batchify(average_precision, scores.T, positive_pairs.T, batch_size, device)

    metrics["text_retrieval_map"] = text_ap.mean().item()
    metrics["image_retrieval_map"] = image_ap.mean().item()

    wm_cos_summary = _summary(wm_cos_all)
    wm_l2_summary = _summary(wm_l2_all)
    rev_cos_summary = _summary(rev_cos_all)
    rev_l2_summary = _summary(rev_l2_all)

    metrics.update({
        "watermark_cos_mean": wm_cos_summary["mean"],
        "watermark_cos_std": wm_cos_summary["std"],
        "watermark_l2_mean": wm_l2_summary["mean"],
        "watermark_l2_std": wm_l2_summary["std"],
        "recovered_cos_mean": rev_cos_summary["mean"],
        "recovered_cos_std": rev_cos_summary["std"],
        "recovered_l2_mean": rev_l2_summary["mean"],
        "recovered_l2_std": rev_l2_summary["std"],
    })

    metrics["watermark_distribution"] = {
        "image_cos": _distribution(wm_cos_image),
        "image_l2": _distribution(wm_l2_image, mode="le", thresholds=(0.1, 0.2, 0.5)),
        "text_cos": _distribution(wm_cos_text),
        "text_l2": _distribution(wm_l2_text, mode="le", thresholds=(0.1, 0.2, 0.5)),
    }
    metrics["recovered_distribution"] = {
        "image_cos": _distribution(rev_cos_image),
        "image_l2": _distribution(rev_l2_image, mode="le", thresholds=(0.1, 0.2, 0.5)),
        "text_cos": _distribution(rev_cos_text),
        "text_l2": _distribution(rev_l2_text, mode="le", thresholds=(0.1, 0.2, 0.5)),
    }

    return metrics

def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)
