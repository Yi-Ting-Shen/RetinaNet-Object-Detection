from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from nets.retinanet import resnet_retinanet
from nets.retinanet_training import focal, smooth_l1
from utils.anchors import get_anchors
from utils.callbacks import LossHistory, ExponentDecayScheduler
from utils.dataloader import RetinanetDatasets
from utils.utils import get_classes

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

'''
訓練自己的目標檢測模型需注意：
1. 確認資料格式為VOC格式，包含輸入圖片和標籤。
   輸入圖片為.jpg，會自動進行resize，灰度圖會轉為RGB。
2. 訓練好的權值文件保存在logs資料夾中，每個epoch都會保存一次。
3. 損失值大小用於判斷是否收斂，驗證集損失不變時模型基本收斂。
4. 調參是一門學問，參數需根據情況調整。
'''

if __name__ == "__main__":
    # 修改classes_path，使其對應自己的dataset
    classes_path = 'model_data/mask_classes.txt'   
    model_path = 'model_data/resnet50_coco_best_v2.1.0.h5'

    # 輸入的shape大小
    input_shape = [448, 448]

    # 設定先驗框的大小
    anchors_size = [32, 64, 128, 256, 512]

    # 訓練分為凍結階段和解凍階段
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 16
    Freeze_lr = 1e-4

    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4
    Unfreeze_lr = 1e-5

    Freeze_Train = True
    num_workers = 1

    # 獲得圖片路徑和標籤
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    # 獲取classes和anchor
    class_names, num_classes = get_classes(classes_path)
    anchors = get_anchors(input_shape, anchors_size)

    model = resnet_retinanet((input_shape[0], input_shape[1], 3), num_classes)
    if model_path != '':
        # 載入預訓練權重
        print('Load weights {}.'.format(model_path))
        model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # 訓練參數設定
    logging = TensorBoard(log_dir='logs/')
    checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ExponentDecayScheduler(decay_rate=0.96, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    loss_history = LossHistory('logs/')

    # 讀取dataset對應的txt
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if Freeze_Train:
        freeze_layers = 174
        for i in range(freeze_layers): model.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

    # 凍結訓練階段
    if Freeze_Train:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('dataset過小，無法進行訓練，請擴充dataset。')

        model.compile(loss={'regression': smooth_l1(), 'classification': focal()},
                      optimizer=Adam(lr=Freeze_lr, clipnorm=1e-2))

        train_dataloader = RetinanetDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train=True)
        val_dataloader = RetinanetDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train=False)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(generator=train_dataloader,
                            steps_per_epoch=epoch_step,
                            validation_data=val_dataloader,
                            validation_steps=epoch_step_val,
                            epochs=end_epoch,
                            initial_epoch=start_epoch,
                            use_multiprocessing=True if num_workers > 1 else False,
                            workers=num_workers,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])

    # 解凍訓練階段
    for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('dataset過小，無法進行訓練，請擴充dataset。')

        model.compile(loss={'regression': smooth_l1(), 'classification': focal()},
                      optimizer=Adam(lr=Unfreeze_lr, clipnorm=1e-2))

        train_dataloader = RetinanetDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train=True)
        val_dataloader = RetinanetDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train=False)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(generator=train_dataloader,
                            steps_per_epoch=epoch_step,
                            validation_data=val_dataloader,
                            validation_steps=epoch_step_val,
                            epochs=end_epoch,
                            initial_epoch=start_epoch,
                            use_multiprocessing=True if num_workers > 1 else False,
                            workers=num_workers,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])
