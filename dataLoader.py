import mxnet as mx
import mxnet.ndarray as F
import numpy as np
from PIL import Image


def img_transform(img, ctx):
    img_array = np.array([np.asarray(img)])
    return F.array(img_array, ctx[0])

def _mask_transform(mask, ctx):
    mask_array = np.array([np.asarray(mask)])
    return mx.nd.array(mask_array, ctx[0]).astype('int32')

image_folder = 'data/images/'
mask_folder = 'data/annotations1/'
bin_folder = 'data/masks/'


def getImages(ctx, NDVI):
    images = []
    masks = []
    for i in range(1,61):
        file = str(i).zfill(3)
        print(image_folder + file+ "_image.png")
        img = Image.open(image_folder + file+ "_image.png").convert('RGB')
        img = img.crop((153, 3, 1113, 963))
        img = np.asarray(img) / 255

        mask = Image.open(mask_folder + file+"_annotation.png")
        mask = mask.crop((153, 3, 1113, 963))

        if (NDVI):
            binary = Image.open(bin_folder+file+"_mask.png")
            binary = binary.crop((153, 3, 1113, 963))
            binary = np.asarray(binary) / 1
            img[:,:,2]=binary

        img= img_transform(img,ctx)
        mask = _mask_transform(mask,ctx)

        img = F.array(img, ctx[0])
        images.append(img)
        masks.append(mask)

    return images,masks


def getDataBatches(batch_size, NDVI, ctx):
    results = []
    images, masks = getImages(ctx, NDVI)
    length = len(images)
    for i in range(0, length, batch_size):
        imgs = images[i:min(i+batch_size, length)]
        image_batch = []
        for x in imgs:
            image_batch.append(x[0].asnumpy())
        image_batch = np.array(image_batch)
        image_batch = F.array(mx.nd.array(image_batch),  ctx[0])

        msks = masks[i:min(i+batch_size, length)]
        mask_batch = []
        for x in msks:
            mask_batch.append(x[0].asnumpy())
        mask_batch = np.array(mask_batch)
        mask_batch = F.array(mx.nd.array(mask_batch), ctx[0])
        results.append([image_batch, mask_batch])
    return results

