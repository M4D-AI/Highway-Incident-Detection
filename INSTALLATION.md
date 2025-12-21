# 安装指南

## 系统要求

- Python >= 3.8
- pip >= 21.0（支持 PEP 517/518）
- 可选：CUDA（用于GPU加速）

## 安装方式

### 1. 从源码安装（开发模式）

```bash
# 克隆仓库
git clone <repository-url>
cd HighwayIncidentDetection

# 安装包（可编辑模式）
pip install -e .

# 安装开发依赖（可选）
pip install -e ".[dev]"
```

### 2. 从源码安装（生产模式）

```bash
# 克隆仓库
git clone <repository-url>
cd HighwayIncidentDetection

# 构建并安装
pip install .
```

### 3. 使用 pip 从 Git 安装

```bash
pip install git+https://github.com/M4D-AI/highway-incident-detection.git
```

### 4. 使用传统方式（requirements.txt）

```bash
pip install -r requirements.txt
```

## 开发环境设置

### 1. 创建虚拟环境

```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 使用 conda
conda create -n highway-detection python=3.10
conda activate highway-detection
```

### 2. 安装开发依赖

```bash
pip install -e ".[dev]"
```

开发依赖包括：
- pytest - 测试框架
- pytest-asyncio - 异步测试支持
- pytest-cov - 代码覆盖率
- black - 代码格式化
- ruff - 快速代码检查
- mypy - 类型检查

### 3. 代码格式化

```bash
# 使用 black 格式化代码
black highway_incident_detection tests

# 使用 ruff 检查代码
ruff check highway_incident_detection tests
```

## 构建分发包

### 构建 wheel 和 source distribution

```bash
# 安装构建工具
pip install build

# 构建分发包
python -m build

# 生成的文件在 dist/ 目录
# - highway_incident_detection-1.0.0-py3-none-any.whl
# - highway-incident-detection-1.0.0.tar.gz
```

### 上传到 PyPI

```bash
# 安装 twine
pip install twine

# 检查分发包
twine check dist/*

# 上传到测试 PyPI
twine upload --repository testpypi dist/*

# 上传到正式 PyPI
twine upload dist/*
```

## 验证安装

```bash
# 检查命令行工具是否可用
highway-incident-detection --version

# 检查 Python 导入
python -c "from highway_incident_detection import __version__; print(__version__)"
```

## 故障排查

### pip 安装失败

如果遇到 `pyproject.toml` 相关错误，确保 pip 版本足够新：

```bash
pip install --upgrade pip
```

### 依赖冲突

如果遇到依赖冲突，建议使用虚拟环境：

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### CUDA 相关问题

如果使用 GPU，确保安装了正确版本的 PyTorch 和 CUDA：

```bash
# 检查 CUDA 版本
nvidia-smi

# 安装对应版本的 PyTorch（如果需要）
# 参考 https://pytorch.org/get-started/locally/
```

## 卸载

```bash
pip uninstall highway-incident-detection
```

