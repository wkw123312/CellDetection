#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
from pathlib import Path
from shutil import which

from config import load_config


def parse_args():
    p = argparse.ArgumentParser(description='Minimal YOLOv8 training launcher')
    p.add_argument('--config', default=None, help='Path to YAML config file')
    p.add_argument('--data', default='./datasets/data.yaml', help='Path to dataset yaml file')
    p.add_argument('--model', default='./yolov8n.pt', help='Path to model checkpoint file')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0')
    p.add_argument('--project', default='./runs')
    p.add_argument('--name', default='bccd_simple')
    return p.parse_args()


def main():
    args = parse_args()

    if args.config:
        cfg = load_config(args.config)
        args.data = cfg.get('data', args.data)
        args.model = cfg.get('model', args.model)
        args.epochs = cfg.get('epochs', args.epochs)
        args.batch = cfg.get('batch', args.batch)
        args.imgsz = cfg.get('imgsz', args.imgsz)
        args.device = cfg.get('device', args.device)
        args.project = cfg.get('project', args.project)
        args.name = cfg.get('name', args.name)

    yolo = which('yolo')
    if not yolo:
        print('yolo CLI not found in PATH. Please install ultralytics.')
        sys.exit(1)

    if not Path(args.model).exists():
        print('Model file not found:', args.model)
        sys.exit(1)

    cmd = [
        yolo,
        'train',
        f'data={args.data}',
        f'model={args.model}',
        f'epochs={args.epochs}',
        f'batch={args.batch}',
        f'imgsz={args.imgsz}',
        f'device={args.device}',
        f'project={args.project}',
        f'name={args.name}',
    ]

    print('Running:', ' '.join(cmd))
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
