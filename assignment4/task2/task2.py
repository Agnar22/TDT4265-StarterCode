import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes
from math import isclose


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    [xmin_pred, ymin_pred, xmax_pred, ymax_pred] = prediction_box
    [xmin_gt, ymin_gt, xmax_gt, ymax_gt]         = gt_box

    # Check if the bounding boxes has 0 overlap
    if(xmin_pred > xmax_gt or xmax_pred < xmin_gt or ymin_pred > ymax_gt or ymax_pred < ymin_gt):
        iou = 0
        return iou

    # Compute intersection
    intersection_box = [
        max(xmin_pred, xmin_gt),
        max(ymin_pred, ymin_gt),
        min(xmax_pred, xmax_gt),
        min(ymax_pred, ymax_gt)
    ]

    intersection_area = (intersection_box[2] - intersection_box[0]) * (intersection_box[3] - intersection_box[1])
    pred_box_area     = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    gt_box_area       = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    # Compute union
    iou = intersection_area / (pred_box_area + gt_box_area - intersection_area)
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    
    if(isclose(num_tp + num_fp, 0)):
        precision = 1
        
    else:
        precision = num_tp / (num_tp + num_fp)

    return precision


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """

    if(isclose(num_tp + num_fn, 0)):
        recall = 0
    else:
        recall = num_tp / (num_tp + num_fn)

    return recall


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    num_pred_boxes = prediction_boxes.shape[0]
    num_gt_boxes   = gt_boxes.shape[0]

    # List of tuples (pred_box_idx, gt_box_idx, IoU) with potential match candidates 
    box_match_candidate = []

    for pred_idx in range(num_pred_boxes):
        for gt_idx in range(num_gt_boxes):
            iou = calculate_iou(prediction_boxes[pred_idx], gt_boxes[gt_idx])
            if( iou >= iou_threshold):
                box_match_candidate.append( (pred_idx, gt_idx, iou) )
    
    # Sort all matches on IoU in descending order
    box_match_candidate.sort(key= lambda x: x[2])

    # Find all matches with the highest IoU threshold

    #Store the idxs of matched boxes to avoid double matching
    match_idxs = [] # matched bb idx tuples on the form (pred_match_idx, gt_match_idx)
    #Lists to be returned
    prediction_boxes_matched = []
    gt_boxes_matched = []

    for match in box_match_candidate:
        # Check if the pred box or gt box has already been matched
        is_matched = any( (pred_match_idx == match[0] or gt_match_idx == match[1]) for (pred_match_idx, gt_match_idx) in match_idxs )
        if(is_matched):
            continue
        #Add matched idxs to avoid double matching
        match_idxs.append( (match[0], match[1]) )
        #Add matched boxes to final match list
        prediction_boxes_matched.append( prediction_boxes[match[0]] )
        gt_boxes_matched.append( gt_boxes[match[1]] )


    prediction_boxes_matched = np.array(prediction_boxes_matched)
    gt_boxes_matched         = np.array(gt_boxes_matched)

    return prediction_boxes_matched, gt_boxes_matched


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    matched_pred_boxes, matched_gt_boxes = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    true_pos = matched_pred_boxes.shape[0]
    false_pos = prediction_boxes.shape[0] - true_pos
    false_neg = gt_boxes.shape[0] - true_pos

    return {'true_pos': true_pos, 'false_pos': false_pos, 'false_neg': false_neg}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    
    true_pos = 0
    false_pos = 0
    false_neg = 0

    num_images = len(all_prediction_boxes)

    for img_num in range(num_images):
        img_stats = calculate_individual_image_result(all_prediction_boxes[img_num], all_gt_boxes[img_num], iou_threshold)
        true_pos  += img_stats['true_pos']
        false_pos += img_stats['false_pos']
        false_neg += img_stats['false_neg']
    
    precision = calculate_precision(true_pos, false_pos, false_neg)
    recall = calculate_recall(true_pos, false_pos, false_neg)

    return (precision, recall)

def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE

    precisions = [] 
    recalls = []

    num_images = len(all_prediction_boxes)


    for conf_thresh in confidence_thresholds:
        all_valid_predictions = []
        for image_idx in range(num_images):
            #Filter out predictions with lower score than confidence threshold
            preds_scores = zip(all_prediction_boxes[image_idx], confidence_scores[image_idx])
            img_predictions = [prediction for (prediction, score) in preds_scores if score > conf_thresh]

            all_valid_predictions.append( np.array(img_predictions) )

        p_i, r_i = calculate_precision_recall_all_images(all_valid_predictions, all_gt_boxes, iou_threshold)
        precisions.append(p_i)
        recalls.append(r_i)

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)

    # YOUR CODE HERE
    p_interp = []

    for r_level in recall_levels:
        valid_precisions = [prec for (prec, rec) in zip(precisions,recalls) if rec >= r_level]
        max_precision = max(valid_precisions) if valid_precisions else 0
        p_interp.append(max_precision)

    average_precision = np.average(p_interp)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
