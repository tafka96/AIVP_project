import time
from functools import partial
from dataLoader import getDataBatches
import mxnet as mx
import numpy as np
from gluoncv.loss import *
from gluoncv.model_zoo.segbase import *
from gluoncv.nn.dropblock import set_drop_prob
from gluoncv.utils import LRScheduler, LRSequential
from gluoncv.utils.parallel import *
from mxnet import gluon, autograd
from sklearn.metrics import confusion_matrix
import argparse

ctx = [mx.cpu(0)]
batch_size = 10
dtype = 'float32'
model_name = 'deeplab'
aux = True
syncbn = False
norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if syncbn else mx.gluon.nn.BatchNorm
crop_size = 960
base_size = 960
norm_kwargs = {'num_devices': 1} if syncbn else {}
optimizer_name = 'sgd'
warmup_epochs = 0

def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser")

    parser.add_argument('--pretrainData', type=str, default='coco',
                        help='dataset name (default: coco)')
    parser.add_argument('--backbone', type=str, default='resnet152',
                        help='backbone name (default: resnet152)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--NDVI', action='store_true', default=False,
                        help='Use NDVI preprocessing on images')
    args = parser.parse_args()
    return args




def getModel(args):
    model = get_segmentation_model(model=model_name, dataset=args.pretrainData,
                                   backbone=args.backbone, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs, aux=aux,
                                   base_size=base_size, crop_size=crop_size, pretrained=True)

    apply_drop_prob = partial(set_drop_prob, 0.0)
    model.apply(apply_drop_prob)
    model.cast(dtype)
    return model


def getNet(model):
    net = DataParallelModel(model, ctx, syncbn)
    return net


def getOptimizer(net, size, args):
    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=args.lr,
                    nepochs=warmup_epochs, iters_per_epoch=size),
        LRScheduler(mode='poly', base_lr=args.lr,
                    nepochs=args.epochs - warmup_epochs,
                    iters_per_epoch=size,
                    power=0.9)
    ])
    kvstore = 'device'
    kv = mx.kv.create(kvstore)

    weight_decay = 1e-4
    momentum = 0.9
    optimizer_params = {'lr_scheduler': lr_scheduler,
                        'wd': weight_decay,
                        'momentum': momentum,
                        'learning_rate': args.lr}
    optimizer = gluon.Trainer(net.module.collect_params(), optimizer_name,
                              optimizer_params, kvstore=kv)
    return optimizer


def getCriterion():
    aux_weight = 0.5
    criterion = MixSoftmaxCrossEntropyLoss(aux, aux_weight=aux_weight)
    criterion = DataParallelCriterion(criterion, ctx, syncbn)
    return criterion


def runTrainer():
    args = parse_args()
    print(args)
    trainset = getDataBatches(batch_size, args.NDVI, ctx)

    testset = trainset[4:6]
    trainset = trainset[0:4]

    model = getModel(args)
    net = getNet(model)

    with net.module.name_scope(): #replace last layers
        net.module.head.block._children['4'] = mx.gluon.nn.Conv2D(in_channels=256, channels=3,
                                                                  kernel_size=1)
        net.module.head.block._children['4'].initialize(mx.init.Xavier(), ctx=ctx)

        net.module.auxlayer.block._children['4'] = mx.gluon.nn.Conv2D(in_channels=256, channels=3,
                                                                      kernel_size=1)
        net.module.auxlayer.block._children['4'].initialize(mx.init.Xavier(), ctx=ctx)

    net.module.hybridize()
    optimizer = getOptimizer(net, len(trainset), args)
    criterion = getCriterion()

    train_loss = 0.0
    train_losses = []
    test_losses = []
    c_matrices = []
    for epoch in range(args.epochs):
        epoch_loss = 0
        print(time.strftime("%H:%M:%S", time.localtime()))
        for i, (data, target) in enumerate(trainset):
            with autograd.record(True):
                outputs = net(data.astype(dtype, copy=False))
                print(outputs[0][0].shape)
                losses = criterion(outputs, target)
                mx.nd.waitall()
                autograd.backward(losses)
            optimizer.step(batch_size)
            for loss in losses:
                train_loss += np.mean(loss.asnumpy()) / len(losses)
                epoch_loss += np.mean(loss.asnumpy()) / len(losses)

            mx.nd.waitall()
        train_losses.append(epoch_loss)
        print(train_losses)
        np.save("TrainLosses.npy", np.array(train_losses))
        net.module.export("Model", epoch=epoch)

        test_loss = 0
        confusion_ms = []
        for i, (data, target) in enumerate(testset):
            output = net(data.astype(dtype, copy=False))
            losses = criterion(output, target)
            mx.nd.waitall()
            for loss in losses:
                test_loss += np.mean(loss.asnumpy()) / len(losses)

            for i in range(len(target)):
                predict = mx.nd.squeeze(mx.nd.argmax(output[0][0][i], 0)).asnumpy()
                conf_matrix = confusion_matrix(target[i].asnumpy().ravel(), predict.ravel(), labels=[0, 1, 2])
                confusion_ms.append(conf_matrix)

        mx.nd.waitall()
        test_losses.append(test_loss)
        np.save("TestLosses.npy", np.array(test_losses))
        print(test_losses)

        c_matrices.append(confusion_ms)
        np.save("ConfusionMatrices.npy", np.array(c_matrices))
        print(confusion_ms)


if __name__ == '__main__':
    runTrainer()