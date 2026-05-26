#!/usr/bin/env python3
"""
Run YOLOv8 validation on a provided test list without modifying the original data.yaml.
Usage:
  python3 test.py --test /path/to/test.txt --model /path/to/best.pt --device 0
"""
import argparse
import subprocess
import sys
import os
import tempfile
import glob
from shutil import which


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate model on a custom test list using yolo CLI')
    p.add_argument('--data', default='./datasets/data.yaml', help='original data.yaml (read-only)')
    p.add_argument('--test', default='./datasets/test.txt', help='path to test.txt or directory containing images')
    p.add_argument('--model', default='./yolov8n.pt', help='model weights to evaluate')
    p.add_argument('--device', default='cpu', help='device id or cpu')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--save_json', action='store_true', help='pass save_json=True to yolo val')
    return p.parse_args()


def extract_names_nc(yaml_path):
    # (removed) replaced by simpler write_tmp_yaml_from_datafile below
    return None, None


def write_tmp_yaml_from_datafile(data_yaml_path, test_path):
    fd, tmp = tempfile.mkstemp(prefix='data_test_', suffix='.yaml', dir=os.getcwd())
    with os.fdopen(fd, 'w') as f:
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, 'r') as df:
                f.write(df.read().rstrip() + '\n')
        f.write(f'val: {test_path}\n')
    return tmp


def main():
    args = parse_args()
    if not os.path.exists(args.test):
        print('Test list or directory not found:', args.test); sys.exit(1)

    # If provided model path doesn't exist, try to find the latest best.pt under runs/*/weights/
    if not os.path.exists(args.model):
        project_root = os.path.dirname(os.path.abspath(__file__))
        runs_dir = os.path.join(project_root, 'runs')
        pattern = os.path.join(runs_dir, '*', 'weights', 'best.pt')
        candidates = glob.glob(pattern)
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            args.model = candidates[0]
            print('Model not found at given path; using latest found:', args.model)
        else:
            print('Model not found:', args.model); sys.exit(1)
    yolo = which('yolo')
    if not yolo:
        print('yolo CLI not found in PATH. Please install ultralytics.'); sys.exit(1)

    # If a directory was provided, build a temporary test list file from images inside it
    created_test_list = None
    if os.path.isdir(args.test):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        img_paths = []
        for root, _, files in os.walk(args.test):
            for f in files:
                if f.lower().endswith(exts):
                    img_paths.append(os.path.join(root, f))
        if not img_paths:
            print('No images found in directory:', args.test); sys.exit(1)
        fd2, created_test_list = tempfile.mkstemp(prefix='test_list_', suffix='.txt', dir=os.getcwd())
        with os.fdopen(fd2, 'w') as tf:
            for p in sorted(img_paths):
                tf.write(os.path.abspath(p) + '\n')
        test_list_path = created_test_list
    else:
        test_list_path = os.path.abspath(args.test)

    try:
        cmd = [
            yolo,
            'predict',
            f'model={args.model}',
            f'source={test_list_path}',
            f'device={args.device}',
            f'imgsz={args.imgsz}',
            f'batch={args.batch}'
        ]
        # save=True makes predict write results (images with boxes) to runs/ by default
        cmd.append('save=True')
        print('Running:', ' '.join(cmd))
        subprocess.run(cmd)
    finally:
        if created_test_list:
            try:
                os.remove(created_test_list)
            except Exception:
                pass

if __name__ == '__main__':
    main()
