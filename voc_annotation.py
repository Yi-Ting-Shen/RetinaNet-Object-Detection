import os
import random
import xml.etree.ElementTree as ET

from utils.utils import get_classes

# 設定執行模式
# 0：整個標籤處理過程，包括生成 ImageSets 中的 txt 和 2007_train.txt、2007_val.txt
# 1：僅生成 ImageSets 中的 txt
# 2：僅生成 2007_train.txt 和 2007_val.txt
annotation_mode = 0

# 設定類別路徑，用於生成 2007_train.txt 和 2007_val.txt 的目標資訊
classes_path = 'model_data/mask_classes.txt'

# 訓練集、驗證集與測試集比例
trainval_percent = 0.8  # 訓練+驗證資料比例
train_percent = 0.8  # 訓練資料比例

# 指向 VOC dataset 所在的資料夾
VOCdevkit_path = 'VOCdevkit'

VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]
classes, _ = get_classes(classes_path)

def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (year, image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

if __name__ == "__main__":
    random.seed(0)
    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        temp_xml = os.listdir(xmlfilepath)
        total_xml = [xml for xml in temp_xml if xml.endswith(".xml")]

        num = len(total_xml)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(range(num), tv)
        train = random.sample(trainval, tr)

        print("train and val size", tv)
        print("train size", tr)
        ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

        for i in range(num):
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for training.")
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)), encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/VOC%s/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))
                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for training done.")