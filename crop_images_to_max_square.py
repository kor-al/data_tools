
import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import click
from typing import Tuple


def center_crop(img : np.ndarray, new_width: int = None, new_height : int =None) -> np.ndarray:
    '''
    Center cropping to the max possible square or given rectangle
    :param img: input image
    :param new_width:
    :param new_height:
    :return: cropped image
    '''
    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.floor((width - new_width) / 2))
    right =  width - left

    top = int(np.floor((height - new_height) / 2))
    bottom = height - top

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img


def crop(img : np.ndarray) -> Tuple[str, np.ndarray, np.ndarray]:
    '''
    Left/Right or Bottom/Top window depending on available space
    :param img: original image
    :return: type of cropping, Left/Right or Bottom/Top window images
    '''
    width = img.shape[1]
    height = img.shape[0]
    if width != height:
        if width > height:
            side = height
            left_img = img[0:side, 0:side, ...]
            right_img = img[0:side, width - side : width, ...]
            return 'lr', left_img, right_img
        else:
            side = width
            top_img = img[0:side, 0:side, ...]
            bottom_img = img[height - side:height, 0:side, ...]
            return 'tb', top_img, bottom_img
    else:
        #square image cannot be cropped more
        return '', img,img

def resize_img(img : np.ndarray, out_size: int) -> np.ndarray:
    return cv2.resize(img, (out_size,out_size), interpolation=cv2.INTER_LINEAR)

@click.command()
@click.option('--in_folder', '-in', default=None, help='Input folder', type=click.Path())
@click.option('--out_size', '-s', default=1024, help='Output size', type=int)
@click.option('--no_resize', '-nrs', is_flag=True, help="Turn off resize to --out_size (default True, to 1024)")
def main(in_folder, out_size, no_resize):
    if in_folder == None:
        print("Specify -in_folder as an input folder with images to crop")
        exit(0)
    print('Crop images to squares and resize to {}...'.format(out_size))
    out = in_folder + '_cropped'
    print("Output saved to")
    if not os.path.exists(out):
        os.makedirs(out)
    print(out)

    img_paths = glob.glob(in_folder + '/*.jpg')
    for i, path in enumerate(tqdm(img_paths)):

        img = cv2.imread(path)
        new_image = center_crop(img)
        if not no_resize:
            new_image =  resize_img(new_image, out_size)
        cv2.imwrite(out + "/{}_c.jpg".format(i), new_image)

        crop_types, img1, img2  = crop(img)
        if crop_types !='':
            if not no_resize:
                img1 = resize_img(img1, out_size)
                img2 = resize_img(img2, out_size)
            cv2.imwrite(out + "/{}_{}.jpg".format(i, crop_types[0]),img1)
            cv2.imwrite(out + "/{}_{}.jpg".format(i,crop_types[1]), img2)

if __name__ == '__main__':
    main()
