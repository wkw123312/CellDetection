# CellDetection

## Description

CellDetection 是一个基于 YOLOv8 的细胞检测训练启动项目。它将数据路径、模型路径和输出目录统一管理为可配置项，避免将个人服务器路径硬编码到代码中。

本项目使用公开的 BCCD Dataset 数据集进行训练与验证。请先从公开来源下载 BCCD 数据集，并将数据集置于 `datasets/BCCD/` 目录下。

## Installation

1. 创建并激活 Python 虚拟环境：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 安装 YAML 支持（如果需要）：
   ```bash
   pip install pyyaml
   ```

## Usage

1. 复制配置模板：
   ```bash
   cp config.yaml.example config.yaml
   ```
2. 编辑 `config.yaml`，将路径改为本地项目路径，例如：
   - `data: ./datasets/data.yaml`
   - `model: ./yolov8n.pt`
   - `project: ./runs`
3. 运行训练：
   ```bash
   python train.py --config config.yaml
   ```
4. 也可直接通过命令行覆盖配置项：
   ```bash
   python train.py --data ./datasets/BCCD/data.yaml --model ./yolov8n.pt --epochs 50
   ```

## Configuration

示例 `config.yaml.example` 包含以下字段：

- `data`: 数据集 YAML 文件路径
- `model`: 模型权重文件路径
- `epochs`: 训练轮数
- `batch`: 批量大小
- `imgsz`: 输入图像尺寸
- `device`: 设备号，例如 `0` 或 `cpu`
- `project`: 保存输出结果的目录
- `name`: 训练运行名称

## 页面展示
<img width="1645" height="895" alt="8e3cbef5f811e6caf5a2db00a5c3e19" src="https://github.com/user-attachments/assets/38d75d7a-6e86-4e99-a3ac-96a20a83072b" />




## Disclaimer

本项目代码已做脱敏处理，已移除个人服务器路径和隐私配置。使用前请自行创建并配置 `config.yaml`，确保所有路径指向本地合法目录。请勿将个人配置文件、模型权重或私有数据提交到公共仓库。

## License

查看 `LICENSE`。
