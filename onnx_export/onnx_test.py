###双输入图片测试
from copy import deepcopy

import onnxruntime as ort
import cv2
import numpy as np
from PIL import Image
import onnx
from onnx_export.onnxmodel import ONNXModel


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def check_model(onnx_path):
    # 测试转换模型是否有问题
    onnx_model = onnx.load(onnx_path)
    check = onnx.checker.check_model(onnx_model)
    print('check', check) # check 为 none 则正常

def mnn_mather(desc1, desc2):
    sim = desc1 @ desc2.transpose()
    sim[sim < 0.75] = 0
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids1 = np.arange(0, sim.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.stack([ids1[mask], nn12[mask]])
    return matches.transpose()


def plot_keypoints(image, kpts, radius=2, color=(0, 0, 255)):
    if image.dtype is not np.dtype('uint8'):
        image = image * 255
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    out = np.ascontiguousarray(deepcopy(image))
    kpts = np.round(kpts).astype(int)

    for kpt in kpts:
        x0, y0 = kpt
        cv2.circle(out, (x0, y0), radius, color, -1, lineType=cv2.LINE_4)
    return out


def plot_matches(image0,
                 image1,
                 kpts0,
                 kpts1,
                 matches,
                 radius=2,
                 color=(255, 0, 0)):
    out0 = plot_keypoints(image0, kpts0, radius, color)
    out1 = plot_keypoints(image1, kpts1, radius, color)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = out0
    out[:H1, W0:, :] = out1

    mkpts0, mkpts1 = kpts0[matches[:, 0]], kpts1[matches[:, 1]]
    mkpts0 = np.round(mkpts0).astype(int)
    mkpts1 = np.round(mkpts1).astype(int)

    points_out = out.copy()
    for kpt0, kpt1 in zip(mkpts0, mkpts1):
        (x0, y0), (x1, y1) = kpt0, kpt1
        mcolor = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        cv2.line(out, (x0, y0), (x1 + W0, y1),
                 color=mcolor,
                 thickness=1,
                 lineType=cv2.LINE_AA)

    cv2.putText(out, str(len(mkpts0)),
                (out.shape[1] - 150, out.shape[0] - 50),
                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

    return out,points_out

def main(model_file):
    # 读取图片
    path1 = '/media/xin/work1/github_pro/ALIKE/test_img/parker/L/1614045104206922_L.png'
    path2 = '/media/xin/work1/github_pro/ALIKE/test_img/parker/R/1614045104206922_R.png'

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img_rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img_rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # 加载 onnx_export
    model = ONNXModel(model_file)
    output1 = model.forward(img_rgb1.astype('float32'))
    output2 = model.forward(img_rgb2.astype('float32'))
    kpts = output1[0]
    desc = output1[1]
    kpts_ref = output2[0]
    desc_ref = output2[1]
    matches = mnn_mather(desc, desc_ref)
    vis_img, points_out = plot_matches(img1, img2, kpts, kpts_ref, matches)
    cv2.imshow('points', points_out)
    cv2.imshow('matches', vis_img)
    cv2.imwrite('./res/point.png',points_out)
    cv2.imwrite('./res/desc.png',vis_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print('success!')


if __name__ == '__main__':
    model_file = '/media/xin/work1/github_pro/ALIKE/onnx_export/model/alike-n.onnx'
    check_model(model_file)
    main(model_file)
