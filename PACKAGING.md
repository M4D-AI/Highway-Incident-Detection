# 打包说明

本项目使用现代的 Python 打包标准（PEP 518, PEP 621），使用 `pyproject.toml` 作为唯一的配置文件。

## 项目结构

```
HighwayIncidentDetection/
├── pyproject.toml          # 现代打包配置（PEP 518/621）
├── MANIFEST.in            # 打包清单文件
├── requirements.txt       # 传统依赖文件（可选，用于兼容）
├── setup.py              # 向后兼容（可选）
└── highway_incident_detection/
    ├── __init__.py
    ├── py.typed           # 类型标记文件
    └── cli.py             # 命令行入口
```

## 打包配置

### pyproject.toml

`pyproject.toml` 是项目的核心配置文件，包含：

1. **构建系统配置** (`[build-system]`)
   - 指定构建后端（setuptools）
   - 定义构建依赖

2. **项目元数据** (`[project]`)
   - 项目名称、版本、描述
   - Python 版本要求
   - 依赖项列表
   - 可选依赖（开发工具）
   - 项目 URL 和分类器

3. **工具配置**
   - `[tool.setuptools]` - setuptools 配置
   - `[tool.black]` - 代码格式化配置
   - `[tool.ruff]` - 代码检查配置
   - `[tool.mypy]` - 类型检查配置
   - `[tool.pytest.ini_options]` - 测试配置

## 安装方式

### 开发模式安装

```bash
pip install -e .
```

### 生产模式安装

```bash
pip install .
```

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

## 构建分发包

### 使用 build 工具（推荐）

```bash
# 安装构建工具
pip install build

# 构建分发包
python -m build

# 生成的文件在 dist/ 目录
```

### 使用 setuptools（传统方式）

```bash
python setup.py sdist bdist_wheel
```

## 命令行工具

安装后可以使用命令行工具：

```bash
# 完整命令
highway-incident-detection --config config/config.yaml

# 简短别名
highway-detection --config config/config.yaml

# 查看版本
highway-incident-detection --version
```

## 类型提示支持

项目包含 `py.typed` 标记文件，表示包支持类型提示。这允许类型检查器（如 mypy）正确检查使用本包的代码。

## 开发工具配置

### Black（代码格式化）

```bash
black highway_incident_detection tests
```

### Ruff（代码检查）

```bash
ruff check highway_incident_detection tests
ruff format highway_incident_detection tests
```

### MyPy（类型检查）

```bash
mypy highway_incident_detection
```

### Pytest（测试）

```bash
pytest
```

## 发布到 PyPI

### 1. 构建分发包

```bash
python -m build
```

### 2. 检查分发包

```bash
pip install twine
twine check dist/*
```

### 3. 上传到测试 PyPI

```bash
twine upload --repository testpypi dist/*
```

### 4. 上传到正式 PyPI

```bash
twine upload dist/*
```

## 版本管理

版本号在 `pyproject.toml` 的 `[project]` 部分的 `version` 字段中定义。

建议使用语义化版本（Semantic Versioning）：
- MAJOR.MINOR.PATCH
- 例如：1.0.0

## 依赖管理

### 运行时依赖

在 `pyproject.toml` 的 `[project]` 部分的 `dependencies` 中定义。

### 开发依赖

在 `pyproject.toml` 的 `[project.optional-dependencies]` 部分的 `dev` 中定义。

### requirements.txt

`requirements.txt` 文件保留用于向后兼容，但主要依赖管理通过 `pyproject.toml` 进行。

## 与 setup.py 的兼容性

虽然项目使用 `pyproject.toml`，但保留了 `setup.py` 文件用于向后兼容。新安装应该使用 `pip install .` 或 `pip install -e .`，它们会自动读取 `pyproject.toml`。

## 最佳实践

1. **使用虚拟环境**：始终在虚拟环境中开发和测试
2. **版本控制**：将 `pyproject.toml` 纳入版本控制
3. **依赖锁定**：考虑使用 `pip-tools` 或 `poetry` 锁定依赖版本
4. **持续集成**：在 CI/CD 中运行代码检查和测试
5. **类型提示**：保持类型提示的完整性和准确性

## 参考资源

- [PEP 518 - Specifying Build Dependencies](https://peps.python.org/pep-0518/)
- [PEP 621 - Project metadata](https://peps.python.org/pep-0621/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [setuptools documentation](https://setuptools.pypa.io/)

