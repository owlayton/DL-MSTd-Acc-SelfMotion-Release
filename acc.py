'''acc.py
Functions to evaluate the accuracy of predictions
Oliver W. Layton
Last updated: Aug 2024
'''

import numpy as np

def circ_error(y_pred, y_true):
    y_pred_reflect = _bring_preds_within_range(y_pred)
    diff = y_pred_reflect - y_true
    diff = _circ_error_diff(diff)
    return diff

def _bring_preds_within_range(y_preds):
    y_pred_reflect = y_preds.copy()
    # Bring predictions >180 to within range
    y_pred_reflect[y_pred_reflect > 180] = 180 + (180 - y_pred_reflect[y_pred_reflect > 180])
    # Bring predictions <180 to within range
    y_pred_reflect[y_pred_reflect < -180] = -180 + (-180 - y_pred_reflect[y_pred_reflect < -180])
    return y_pred_reflect

def _circ_error_diff(diff):
    # Wrap around errors larger than 180 degrees
    diff[diff > 180] = 360 - diff[diff > 180]
    diff[diff < -180] = -360 - diff[diff < -180]
    return diff

def mse(y_pred, y_true, circ_correction=False):
    if not circ_correction:
        diff = y_pred - y_true
    else:
        diff = circ_error(y_pred, y_true)

    return np.mean(diff**2, axis=0)


def mae(y_pred, y_true, circ_correction=False):
    if not circ_correction:
        diff = y_pred - y_true
    else:
        diff = circ_error(y_pred, y_true)

    return np.mean(np.abs(diff), axis=0)
