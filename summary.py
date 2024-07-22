from nets.retinanet import resnet_retinanet

if __name__ == "__main__":
    input_shape = [600, 600, 3]
    num_classes = 20

    model = resnet_retinanet(input_shape, num_classes)
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
