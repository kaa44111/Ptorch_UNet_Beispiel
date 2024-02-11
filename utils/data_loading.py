import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB',(temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


def load_image(filename):
    ext = splitext(filename)[1]

    # splitext(filename)是os.path模块中的函数，用于将文件名分离成文件名和扩展名两个部分。
    # splitext(filename)[1]提取扩展名部分，比如.jpg、.png、.npy等。这里的ext变量将保存这个扩展名。
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.png'))[0]
    # print(f"找到遮罩文件: {mask_file}")
    # 得到mask_dir目录下 idx.png 格式的名字
    #----------------修改    开始------------
    # try:
    #      # 获取第一个匹配文件
    #     print(f"找到遮罩文件: {mask_file}")
    # except IndexError:
    #     print("没有找到与模式匹配的遮罩文件。")
    # ----------------修改    结束------------
    mask = np.asarray(load_image(mask_file))
    a = np.unique(mask)
    # print(f'idx: {idx}  a  {a}')
    # if a == [  0 255] :
    #     print(f'idx: {idx}  a  {a}')
    # else:
    #     print(f'-------------------idx: {idx}  a  {a}--------------------')

    # print(f"{mask_file} np.unique(mask): {np.unique(mask)}")
    # 得到像素中的不同数
    if mask.ndim == 2:
        # print("ndim == 2")   cod10k是这个

        # a = np.unique(mask)
        # print(f"{mask_file} a.ndim: {a.ndim}  ")
        # print(f"{mask_file} np.unique(mask): {np.unique(mask)}")
        # if a.ndim == 2:
        #     print(f'a.ndim=2*****************')
        #     # print(f"{mask_file} np.unique(mask): {np.unique(mask)}")
        # if a.ndim>2 :
        #     print(f'a.ndim>2*****************')
        #     print(f"{mask_file} np.unique(mask): {np.unique(mask)}")
        return np.unique(mask)
        # return mask
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        print(f"{mask_file} np.unique(mask): {np.unique(mask)}")
        # print("ndim == 3")
        return np.unique(mask, axis=0)
    else:
        print(f"{mask_file} np.unique(mask): {np.unique(mask)}")
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        # self.ids 是照片的名称，不带文件格式的文件名。
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))


        # print(f'unique[0]   {unique[0]} ')
        #
        # print(f'self.ids[0] {self.ids[0]}')
        #
        # mask_file = list(mask_dir.glob(self.ids[0] + mask_suffix + '.*'))[0]
        # print(f'mask_file {mask_file}')
        # mask = np.asarray(load_image(mask_file))
        # print(f'mask {mask}')
        # np_mask = np.unique(mask)
        # print(f'np_mask {np_mask}')
        # aa=unique_mask_values(self.ids[0], self.mask_dir, self.mask_suffix)
        # print(f'unique_mask_values aa {aa}')
        # tqdm(unique_mask_values(self.ids[0], self.mask_dir, self.mask_suffix),total=1)
        # unique1 = list(        (tqdm(unique_mask_values(self.ids[0], self.mask_dir, self.mask_suffix),total=1) ))
        # print(f'unique1 {unique1}')
        # unique2 = list(unique_mask_values(self.ids[0], self.mask_dir, self.mask_suffix))
        # print(f'unique1 {unique2}')
        #
        print(f'unique[0]   {unique[0]} ')
        print(f'unique[1]   {unique[1]} ')
        # unique_mask_values(self.ids[1], self.mask_dir, self.mask_dir)
        print(f'unique[2]   {unique[2]} ')
        print(f'unique[3]   {unique[3]} ')
        # unique_mask_values(self.ids[2], self.mask_dir, self.mask_dir)
        print(f'unique[4]   {unique[4]} ')
        print(f'unique[5]   {unique[5]} ')
        print(f'unique[6]   {unique[6]} ')
        print(f'unique[7]   {unique[7]} ')
        print(f'unique[8]   {unique[8]} ')
        # unique_mask_values(self.ids[6], self.mask_dir, self.mask_dir)
        c = np.concatenate(unique)
        # a = np.unique(np.concatenate(unique), axis=0).tolist()
        # self.mask_values = list(sorted(a))


        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        # self.mask_values = [0,255]

        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        #
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            #如果是mask图片，就生成一张大小与图像相同的空遮挡
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        # mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        # img_file = list(self.images_dir.glob(name + '.*'))
        #将数据集图片的格式改成自己的，这里都是.jpg
        #images_dir - dir_img -  '../Dataset/TrainDataset/Image/'
        #mask_file - dir_mask -  '../Dataset/TrainDataset/GT/'
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.png'))
        #将文件路径变成list存储
        img_file = list(self.images_dir.glob(name + '.jpg'))
        #

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])#取第一个文件路径并加载
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='')

def unique_values(idx, mask_dir):
    # imgs = os.listdir(root)
    imgs = list(mask_dir.glob(idx + '.png'))[0]
    concat_unique = np.empty(1)
    for imgpath in imgs:
        img = np.asarray(Image.open(imgs))
        # 得到像素中的不同数
        unique = np.unique(img)
        # 对其进行拼接
        concat_unique = np.concatenate([concat_unique, unique])
    # 对拼接后的图片进行再次求不同像素，即全部文件中不同像素数，排序后返回
    return list(sorted(np.unique(concat_unique)))
# if __name__ == '__main__':

