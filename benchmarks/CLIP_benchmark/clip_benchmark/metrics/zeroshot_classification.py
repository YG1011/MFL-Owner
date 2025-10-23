"""
Code adapated from https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py
Thanks to the authors of OpenCLIP
"""
import logging
import os
from contextlib import suppress
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import classification_report, balanced_accuracy_score



def zero_shot_classifier(model, Trigger_mat, tokenizer, classnames, templates, device, amp=True):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if type(templates) == dict:
                # class-specific prompts (e.g., CuPL https://arxiv.org/abs/2209.03320)
                texts = templates[classname]
            elif type(templates) == list:
                # generic prompts tht are specialized for each class by replacing {c} with the class name
                texts = [template.format(c=classname) for template in templates]
            else:
                raise ValueError("templates must be a list or a dict")
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            # watermark
            class_embeddings = class_embeddings @ Trigger_mat
            # print(Trigger_mat)

            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.
    
    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies
    
    Returns
    -------
    
    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]

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

def run_classification(
    model,
    Trigger_mat,
    classifier,
    dataloader,
    device,
    amp=True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []
    nb = 0

    # watermark verification
    wm_cos_values: List[float] = []
    wm_l2_values: List[float] = []
    rev_cos_values: List[float] = []
    rev_l2_values: List[float] = []

    with torch.no_grad():
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                # predict
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)

                origin_image_features = image_features

                # watermark
                image_features = image_features @ Trigger_mat
                image_features = F.normalize(image_features, dim=-1)
                # print(Trigger_mat)

                # Verification
                print("Origin x_p1 and x_o1:")
                Trigger_cos_sim, Trigger_pair_dis = cal_sim(origin_image_features, image_features)
                print("Trigger_Verification: cos similarity: %lf, pair distance: %lf" % (
                    float(Trigger_cos_sim.mean()), float(Trigger_pair_dis.mean())))
                wm_cos_values.append(float(Trigger_cos_sim.mean()))
                wm_l2_values.append(float(Trigger_pair_dis.mean()))

                print("Revised x_p1 and x_o1:")
                Rvised_image_features = image_features @ torch.linalg.inv(Trigger_mat)
                Trigger_cos_sim, Trigger_pair_dis = cal_sim(origin_image_features, Rvised_image_features)
                print("Trigger_Verification: cos similarity: %lf, pair distance: %lf" % (
                    float(Trigger_cos_sim.mean()), float(Trigger_pair_dis.mean())))

                rev_cos_values.append(float(Trigger_cos_sim.mean()))
                rev_l2_values.append(float(Trigger_pair_dis.mean()))


                logits = 100. * image_features @ classifier
            
            true.append(target.cpu())
            pred.append(logits.float().cpu())

    print("Total Verification:")
    print("Origin x_p1 and x_o1:")
    if wm_cos_values:
        wm_cos_mean = sum(wm_cos_values) / len(wm_cos_values)
        wm_l2_mean = sum(wm_l2_values) / len(wm_l2_values)
    else:
        wm_cos_mean = wm_l2_mean = 0.0
    print("Trigger_Verification: cos similarity: %lf, pair distance: %lf" % (
        wm_cos_mean, wm_l2_mean))
    print("Revised x_p1 and x_o1:")
    if rev_cos_values:
        rev_cos_mean = sum(rev_cos_values) / len(rev_cos_values)
        rev_l2_mean = sum(rev_l2_values) / len(rev_l2_values)
    else:
        rev_cos_mean = rev_l2_mean = 0.0
    print("Trigger_Verification: cos similarity: %lf, pair distance: %lf" % (
        rev_cos_mean, rev_l2_mean))

    pred = torch.cat(pred)
    true = torch.cat(true)
    def _summary(values: List[float]) -> Tuple[float, float]:
        tensor = torch.tensor(values, dtype=torch.float32)
        return tensor.mean().item(), tensor.std(unbiased=False).item()

    wm_cos_std = wm_l2_std = rev_cos_std = rev_l2_std = 0.0
    if wm_cos_values:
        wm_cos_mean, wm_cos_std = _summary(wm_cos_values)
        wm_l2_mean, wm_l2_std = _summary(wm_l2_values)
    if rev_cos_values:
        rev_cos_mean, rev_cos_std = _summary(rev_cos_values)
        rev_l2_mean, rev_l2_std = _summary(rev_l2_values)

    verification = {
        "watermark_cos_mean": wm_cos_mean,
        "watermark_cos_std": wm_cos_std,
        "watermark_l2_mean": wm_l2_mean,
        "watermark_l2_std": wm_l2_std,
        "recovered_cos_mean": rev_cos_mean,
        "recovered_cos_std": rev_cos_std,
        "recovered_l2_mean": rev_l2_mean,
        "recovered_l2_std": rev_l2_std,
    }

    return pred, true, verification

def average_precision_per_class(scores, targets):
    """
    Compute average precision  for each class
    this metric is used for multi-label classification
    see explanations here https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    Code is adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py, thanks to the authors of `tnt`.

    Parameters
    ----------

    scores: torch.Tensor
        logits, of shape (N,C) where N is the number of examples, C the number of classes
    
    targets: torch.Tensor
        one-hot vectors of groundtruth targets (N, C), where N is the number of examples, C is the
        number of classes
    
    Returns
    -------

    torch.Tensor of shape (C,) of avereage precision for each class, where C is     
    the number of classes.
    
    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    for k in range(scores.size(1)):
        # sort scores
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap


def evaluate(model, dataloader, tokenizer, classnames, templates, device, watermark_dir, watermark_dim, trigger_num, amp=True, verbose=False, save_clf=None, load_clfs=[]):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`

    dataloader: torch.utils.data.Dataloader
s
    tokenizer: text tokenizer

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """

    Trigger_mat_pth = os.path.join(watermark_dir, str(watermark_dim), "trigger_mat_%d.pth" % trigger_num)
    Trigger_mat = torch.load(Trigger_mat_pth)
    Trigger_mat = Trigger_mat.to(device)
    print("Loaded Trigger matrix from", Trigger_mat_pth)

    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location='cpu') / n
        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location='cpu') / n
        classifier = classifier.to(device)
    else:
        classifier = zero_shot_classifier(model, Trigger_mat, tokenizer, classnames, templates, device, amp=amp)
    
    if save_clf is not None:
        torch.save(classifier, save_clf)
        # exit() - not sure if we want to exit here or not.

    logits, target, verification = run_classification(
        model, Trigger_mat, classifier, dataloader, device, amp=amp
    )
    is_multilabel = (len(target.shape) == 2)

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        ap_per_class = average_precision_per_class(logits, target)
        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
        metrics = {"mean_average_precision": ap_per_class.mean().item()}
        metrics.update(verification)
        return metrics
    else:
        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall

        pred = logits.argmax(axis=1)
        # measure accuracy
        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan") 
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))
        metrics = {
            "acc1": acc1,
            "acc5": acc5,
            "mean_per_class_recall": mean_per_class_recall,
        }
        metrics.update(verification)
        return metrics
