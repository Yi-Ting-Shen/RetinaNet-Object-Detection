import keras
import numpy as np

class AnchorBox:
    def __init__(self, ratios, scales):
        self.ratios = ratios
        self.scales = scales
        self.num_anchors = len(self.ratios) * len(self.scales)

    def generate_anchors(self, base_size=16):
        anchors = np.zeros((self.num_anchors, 4))
        anchors[:, 2:] = base_size * np.tile(self.scales, (2, len(self.scales))).T

        # 計算先驗框的面積
        areas = anchors[:, 2] * anchors[:, 3]

        anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)))
        anchors[:, 3] = np.sqrt(areas * np.repeat(self.ratios, len(self.scales)))

        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T 
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        return anchors

    def shift(self, shape, stride, anchors):
        # 生成特徵層的網格中心
        shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
        shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shift_x = np.reshape(shift_x, [-1])
        shift_y = np.reshape(shift_y, [-1])

        # 將網格中心進行堆疊
        shifts = np.stack([shift_x, shift_y, shift_x, shift_y], axis=0)
        shifts = np.transpose(shifts)
        number_of_anchors = np.shape(anchors)[0]

        k = np.shape(shifts)[0]
        shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]))
        shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
        return shifted_anchors

# 計算共享特徵層的大小
def get_img_output_length(height, width):
    filter_sizes = [7, 3, 3, 3, 3, 3, 3]
    padding = [3, 1, 1, 1, 1, 1, 1]
    stride = [2, 2, 2, 2, 2, 2, 2]
    feature_heights = []
    feature_widths = []

    for i in range(len(filter_sizes)):
        height = (height + 2 * padding[i] - filter_sizes[i]) // stride[i] + 1
        width = (width + 2 * padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-5:], np.array(feature_widths)[-5:]

def get_anchors(input_shape, anchors_size=[32, 64, 128, 256, 512], strides=[8, 16, 32, 64, 128],
                ratios=[0.5, 1, 2], scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]):
    feature_heights, feature_widths = get_img_output_length(input_shape[0], input_shape[1])
    
    all_anchors = []
    anchor_box = AnchorBox(ratios, scales)
    for i in range(len(anchors_size)):
        # 生成每個特徵點的先驗框
        anchors = anchor_box.generate_anchors(anchors_size[i])
        shifted_anchors = anchor_box.shift([feature_heights[i], feature_widths[i]], strides[i], anchors)
        all_anchors.append(shifted_anchors)

    # 將每個特徵層的先驗框進行堆疊
    all_anchors = np.concatenate(all_anchors, axis=0)
    all_anchors = all_anchors / np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
    all_anchors = all_anchors.clip(0, 1)
    return all_anchors
