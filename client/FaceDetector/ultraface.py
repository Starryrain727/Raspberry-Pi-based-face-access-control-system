import MNN
import cv2
import numpy as np
from math import ceil
from . import box_utils

strides = [8, 16, 32, 64]
min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
threshold = 0.7
center_variance = 0.1
size_variance = 0.2


def define_img_size(image_size):
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [ceil(size / stride) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
    return priors


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes, clamp=True):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    priors = np.array(priors)
    if clamp:
        np.clip(priors, 0.0, 1.0, out=priors)
    return priors


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def inference(img):
    model = "slim-320.mnn"
    image_mean = np.array([127, 127, 127])
    image_std = 128.0
    input_size = [320, 240]

    interpreter = MNN.Interpreter(model)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, tuple(input_size))
    image = image.astype(np.float32)
    image = (image - image_mean) / image_std
    image = image.transpose((2, 0, 1))
    image = image.astype(np.float32)

    tmp_input = MNN.Tensor((1, 3, input_size[1], input_size[0]), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    scores = interpreter.getSessionOutput(session, "scores").getData()
    boxes = interpreter.getSessionOutput(session, "boxes").getData()
    boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
    scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)

    priors = define_img_size(input_size)
    boxes = box_utils.convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
    boxes = box_utils.center_form_to_corner_form(boxes)
    boxes, labels, probs = predict(img.shape[1], img.shape[0], scores, boxes, threshold)
    imgs = []
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        imgs.append(img[box[1]:box[3], box[0]:box[2]])
    return imgs
