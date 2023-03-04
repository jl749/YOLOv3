"""
Download PascalVOC 2007 & 2012
Convert label files from "xml -> txt" and "xyxy (0~img_size) -> cxcywh (0~1)"
"""
import sys
from typing import Optional
from pathlib import Path

import xml.etree.ElementTree as ET
import requests
import hashlib
import tarfile
import csv

from tqdm import tqdm


BASE_DIR = Path(__file__).parent
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

"""
DOWNLOAD
========================================================================================================================
"""
def _calculate_md5(file_dir: str, chunk_size: int = 1024 * 1024) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but indicates that we are
    # not using the MD5 checksum for cryptography. This enables its usage in restricted environments like FIPS. Without
    # it torchvision.datasets is unusable in these environments since we perform a MD5 check everywhere.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(file_dir, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(file_dir: str, md5: Optional[str] = None) -> bool:
    if not Path(file_dir).exists():
        return False
    if md5 is None:
        return True
    return md5 == _calculate_md5(file_dir)


def download_url(
    url: str, save_dir: str, filename: Optional[str] = None, md5: Optional[str] = None
) -> None:
    """Download a file from an url and place it in save_dir.
    Args:
        url (str): URL to download file from
        save_dir (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    save_dir = Path(save_dir)
    file_dir = save_dir.joinpath(filename)
    if check_integrity(str(file_dir), md5) and file_dir.exists():
        print(f"Using downloaded and verified file: {file_dir}")
        return

    # download the file
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    _tmp_file = save_dir.joinpath(f"{filename}.temp")
    with _tmp_file.open("wb") as file:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            progress_bar.update(len(data))
    progress_bar.close()

    if check_integrity(str(_tmp_file), md5):
        _tmp_file.rename(file_dir)  # download success
    else:
        # _tmp_file.unlink(missing_ok=True)
        raise RuntimeError("An error occurred while downloading, please try again.")


"""
CONVERT
========================================================================================================================
"""
def _convert(size, box):
    """bboxes normalized to 0~1 range, xyxy --> cxcywh"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    cx = (box[0] + box[1]) / 2.0
    cy = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    cx = cx * dw
    w = w * dw
    cy = cy * dh
    h = h * dh
    return cx, cy, w, h


def convert_annotation(data_dir: Path, image_id):
    """
    convert "xml -> txt", "xyxy (0~img_size) -> cxcywh (0~1)"
    txt files newly created inside `labels` folder
    """
    _xml_fp = data_dir.joinpath("Annotations", f"{image_id}.xml").open("r")
    _txt_fp = data_dir.joinpath("labels", f"{image_id}.txt").open("w")

    tree = ET.parse(_xml_fp)
    root = tree.getroot()

    # image info
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    # bbox info
    for bbox in root.iter("object"):
        difficult = bbox.find("difficult").text
        cls = bbox.find("name").text
        if cls not in CLASSES or int(difficult) == 1:  # skip unknown cls or difficult bbox
            continue

        cls_id = CLASSES.index(cls)
        xmlbox = bbox.find("bndbox")

        xyxy = (float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text))
        cxcywh = _convert((w, h), xyxy)
        _txt_fp.write(f"{cls_id} {cxcywh[0]} {cxcywh[1]} {cxcywh[2]} {cxcywh[3]}\n")


"""
MAIN
========================================================================================================================
"""
def main():
    data_dir = BASE_DIR.joinpath("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    download_data = [
        {
            "file_name": "voc_2007_test.tar",
            "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
            "md5": "b6e924de25625d8de591ea690078ad9f"
        },
        {
            "file_name": "voc_2007.tar",
            "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            "md5": "c52e279531787c972589f7e41ab4ae64"
        },
        {
            "file_name": "voc_2012.tar",
            "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
            "md5": "6cd6e144f989b92b3379bac3b3de84fd"
        }
    ]
    sets = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

    for _data in download_data:
        _data_dir = data_dir.joinpath(_data["file_name"])
        download_url(
            url=_data["url"],
            save_dir=str(_data_dir.parent),
            filename=_data_dir.name,
            md5=_data["md5"]
        )

        with tarfile.open(str(_data_dir), "r") as tar:
            tar.extractall(str(_data_dir.parent))

    data_dir = data_dir.joinpath("VOCdevkit")  # update data_dir
    image_ids_per_set = {}
    for year, image_set in sets:
        _key = f"{year}_{image_set}"

        _data_dir = data_dir.joinpath(f"VOC{year}")
        label_dir = _data_dir.joinpath("labels")
        label_dir.mkdir(parents=True, exist_ok=True)

        image_id_dir = _data_dir.joinpath("ImageSets", "Main", f"{image_set}.txt")
        with image_id_dir.open("r") as fp:
            image_ids = fp.read().strip().split()

        image_ids_per_set[_key] = []
        for _id in image_ids:
            convert_annotation(_data_dir, _id)
            image_ids_per_set[_key].append(_id)

    """
    Get train by using train+val from 2007 and 2012
    Then we only test on 2007 test set
    Unclear from paper what they actually just as a dev set
    
    cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
    cp 2007_test.txt test.txt
    rm 2007* 2012*
    """
    train_ids = image_ids_per_set["2007_train"] + \
                image_ids_per_set["2007_val"] + \
                image_ids_per_set["2012_train"] + \
                image_ids_per_set["2012_val"]

    test_ids = image_ids_per_set["2007_test"]

    dest_dir = data_dir.parent
    with dest_dir.joinpath("train.csv").open(mode="w", newline="") as train_file:
        for _id in train_ids:
            image_file = f"{_id}.jpg"
            text_file = f"{_id}.txt"
            data = [image_file, text_file]
            writer = csv.writer(train_file)
            writer.writerow(data)

    with dest_dir.joinpath("test.csv").open(mode="w", newline="") as train_file:
        for _id in test_ids:
            image_file = f"{_id}.jpg"
            text_file = f"{_id}.txt"
            data = [image_file, text_file]
            writer = csv.writer(train_file)
            writer.writerow(data)

    """postprocess"""
    img_dir = dest_dir.joinpath("images")
    label_dir = dest_dir.joinpath("labels")
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    label_files = [
        *data_dir.joinpath("VOC2007", "labels").glob("*.txt"),
        *data_dir.joinpath("VOC2012", "labels").glob("*.txt")
    ]
    for label_file in label_files:
        label_file.rename(label_dir / label_file.name)

    img_files = [
        *data_dir.joinpath("VOC2007", "JPEGImages").glob("*.jpg"),
        *data_dir.joinpath("VOC2012", "JPEGImages").glob("*.jpg")
    ]
    for img_file in img_files:
        img_file.rename(img_dir / img_file.name)

    import shutil
    shutil.rmtree(str(data_dir))


if __name__ == '__main__':
    main()
