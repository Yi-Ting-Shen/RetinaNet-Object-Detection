import colorsys
import os
import time

import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from PIL import ImageDraw, ImageFont

from nets.retinanet import resnet_retinanet
from utils.utils_bbox import BBoxUtility
from utils.utils import get_classes, resize_image, cvtColor
from utils.anchors import get_anchors

class Retinanet(object):
    _defaults = {
        # 使用自己訓練好的模型進行預測一定要修改model_path和classes_path！
        # model_path指向logs資料夾下的權值檔案，classes_path指向model_data下的txt
        # 訓練好後logs資料夾下存在多個權重檔案，選擇驗證集損失較低的即可。
        # 驗證集損失較低不代表mAP較高，僅代表該權值在驗證集上泛化性能較好。
        # 如果出現shape對不上，同時要注意訓練時的model_path和classes_path參數的修改
        "model_path": 'model_data/best.h5',
        "classes_path": 'model_data/mask_classes.txt',
        
        "input_shape": [448, 448], # 用於預測的圖像大小，和train時使用同一個即可        
        "confidence": 0.5, # confidence大於這個門檻值的bbox會被保留下來
        "nms_iou": 0.3, # NMS的IoU門檻值        
        'anchors_size': [32, 64, 128, 256, 512], # 每一個特徵層先驗框的基礎大小
        "letterbox_image": False, # 是否使用letterbox_image對輸入圖像進行不失真的resize
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # 初始化RetinaNet
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # 計算總的類別數量
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors = get_anchors(self.input_shape, self.anchors_size)
        self.num_classes = self.num_classes
        
        # 生成畫框的顏色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        
        self.bbox_util = BBoxUtility(self.num_classes, nms_thresh=self.nms_iou)
        self.generate()

    # 載入模型
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 載入模型與權值
        self.retinanet = resnet_retinanet([self.input_shape[0], self.input_shape[1], 3], self.num_classes)
        self.retinanet.load_weights(self.model_path, by_name=True)
        print('{} model, anchors, and classes loaded.'.format(model_path))

    # 檢測圖片
    def detect_image(self, image):
        # 獲得輸入圖片的高和寬
        image_shape = np.array(np.shape(image)[0:2])
        # 將圖像轉換成RGB圖像，防止灰度圖在預測時報錯
        # 這個code僅支持RGB圖像的預測，所有其它類型的圖像都會轉化成RGB
        image = cvtColor(image)
        # 給圖像增加灰條(self.letterbox_image)，實現不失真的resize
        # 也可以直接resize進行識別
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # np.expand_dims添加上batch_size維度，圖片預處理，歸一化
        image_data = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))

        # 模型預測
        preds = self.retinanet.predict(image_data)
        # 將預測結果進行解碼
        results = self.bbox_util.decode_box(preds, self.anchors, image_shape, 
                                            self.input_shape, self.letterbox_image, confidence=self.confidence)

        # 如果沒有檢測到物體，則返回原圖
        if results[0] is None: 
            return image

        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]
        
        # 設定字體與邊框厚度
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)
        
        # 圖像繪製
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    # 計算每秒幀數(FPS)
    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))

        preds = self.retinanet.predict(image_data)
        results = self.bbox_util.decode_box(preds, self.anchors, image_shape, 
                                            self.input_shape, self.letterbox_image, confidence=self.confidence)
        t1 = time.time()
        for _ in range(test_interval):
            preds = self.retinanet.predict(image_data)
            results = self.bbox_util.decode_box(preds, self.anchors, image_shape, 
                                                self.input_shape, self.letterbox_image, confidence=self.confidence)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    # 生成mAP評估所需的文本檔案
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w") 
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = preprocess_input(np.expand_dims(np.array(image_data, dtype='float32'), 0))

        preds = self.retinanet.predict(image_data)
        results = self.bbox_util.decode_box(preds, self.anchors, image_shape, 
                                            self.input_shape, self.letterbox_image, confidence=self.confidence)
        if results[0] is None: 
            return 

        top_label = results[0][:, 5]
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])
            
            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return 
