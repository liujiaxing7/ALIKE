import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ALIKE_code.nets.alnet import ALNet
from pytorch_lightning.plugins.io import TorchCheckpointIO as tcio
from PIL import Image
from torchvision import transforms

def test_on_image_pair(model, script_model, images):
    # run model
    model_res = model(images)
    script_model_res = script_model(images)

    # 检查结果
    print(torch.equal(model_res[0],script_model_res[0]))
    print(torch.equal(model_res[1],script_model_res[1]))


def get_model(checkpoint_path,device):
    model = ALNet(c1=16, c2=32, c3=64, c4=128, dim=128, agg_mode='cat', single_head=True)
    model.load_from_checkpoint(checkpoint_path)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model



def main(checkpoint_path,OUTPUT_MODEL,device):
    image_path = "/media/xin/work1/github_pro/ALIKE/ALIKE_code/model_transfor/img/1.png"
    # 加载图片
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 将PIL类型转化成numpy类型
    image = np.array(image)
    transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    # 加载模型
    model = get_model(checkpoint_path,device)

    # 转为pytorch支持的pth模型
    script_model = torch.jit.trace(model, image)

    # 保存pth模型
    torch.jit.save(script_model, OUTPUT_MODEL)
    # 加载转换后的模型
    # script_model = torch.jit.load(OUTPUT_MODEL)
    #
    # # test on same size images
    # test_on_image_pair(model, script_model, image)


    print(f'torch script model "{OUTPUT_MODEL}" created and tested')
    print("done")


if __name__ == "__main__":
    ckpt_path = '/media/xin/work1/github_pro/ALIKE/ALIKE_code/training/log_train/train/R4.0.1/normal_checkpoints/epoch=0-mean_metric=0.0733.ckpt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_model = "script_model.pth"
    main(ckpt_path,output_model,device)
