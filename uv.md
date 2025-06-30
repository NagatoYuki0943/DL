# uv vs pixi

[uv](https://docs.astral.sh/uv/)

[uv github](https://github.com/astral-sh/uv)

[pixi](https://pixi.sh/latest/)

[pixi github](https://github.com/prefix-dev/pixi)

**UV：新一代高性能 Python 环境管理工具**

- **开发背景与定位**
  - UV 由 Astral 团队开发，采用 Rust 语言实现，目标是替代 pip、virtualenv、pip-tools 等传统工具，成为一站式 Python 包和环境管理方案[1](https://cloud.tencent.com/developer/article/2506633),[2](https://realpython.com/python-uv/)[3](https://blog.csdn.net/deephub/article/details/144801548),[6](https://news.qq.com/rain/a/20241229A0318J00),[9](https://blog.csdn.net/qq_35082030/article/details/146549871)。
  - UV 集成了包管理、虚拟环境、Python 版本管理、项目初始化、依赖锁定、包构建与发布等功能，极大简化了 Python 项目开发全流程[2](https://realpython.com/python-uv/),[3](https://blog.csdn.net/deephub/article/details/144801548),[6](https://news.qq.com/rain/a/20241229A0318J00),[9](https://blog.csdn.net/qq_35082030/article/details/146549871)。
- **核心技术与功能**
  - **极致性能**：得益于 Rust 实现和优化的依赖解析算法，包安装和环境创建速度比 pip 快 10-100 倍，特别适合大型依赖树或频繁重建环境的场景[1](https://cloud.tencent.com/developer/article/2506633),[2](https://realpython.com/python-uv/),[3](https://blog.csdn.net/deephub/article/details/144801548),[6](https://news.qq.com/rain/a/20241229A0318J00),[9](https://blog.csdn.net/qq_35082030/article/details/146549871)。
  - **项目级虚拟环境**：自动为每个项目创建隔离的 .venv 目录，避免依赖冲突，支持 pyproject.toml 规范和 requirements.txt 生成[1](https://cloud.tencent.com/developer/article/2506633),[3](https://blog.csdn.net/deephub/article/details/144801548),[6](https://news.qq.com/rain/a/20241229A0318J00),[9](https://blog.csdn.net/qq_35082030/article/details/146549871)。
  - **依赖与环境锁定**：通过 uv.lock 文件精确锁定依赖，确保环境可重现性，便于团队协作和 CI/CD 流程[3](https://blog.csdn.net/deephub/article/details/144801548),[6](https://news.qq.com/rain/a/20241229A0318J00),[9](https://blog.csdn.net/qq_35082030/article/details/146549871)。
  - **Python 版本管理**：内置多 Python 版本支持，无需 pyenv，可为不同项目分配独立 Python 解释器[2](https://realpython.com/python-uv/),[3](https://blog.csdn.net/deephub/article/details/144801548),[6](https://news.qq.com/rain/a/20241229A0318J00),[9](https://blog.csdn.net/qq_35082030/article/details/146549871)。
  - **一体化命令体验**：如 `uv init` 初始化项目，`uv add` 添加依赖，`uv venv` 创建虚拟环境，`uv pip install` 安装包，操作统一且易于上手[1](https://cloud.tencent.com/developer/article/2506633),[2](https://realpython.com/python-uv/),[3](https://blog.csdn.net/deephub/article/details/144801548),[6](https://news.qq.com/rain/a/20241229A0318J00),[9](https://blog.csdn.net/qq_35082030/article/details/146549871)。
  - **轻量与兼容**：环境占用空间小，完全兼容 pip 语法，易于从传统工具迁移[1](https://cloud.tencent.com/developer/article/2506633),[3](https://blog.csdn.net/deephub/article/details/144801548),[6](https://news.qq.com/rain/a/20241229A0318J00),[9](https://blog.csdn.net/qq_35082030/article/details/146549871)。
- **典型应用场景**
  - 需要极快依赖安装和环境切换的开发/测试流程
  - 团队协作、CI/CD、教学等对环境一致性要求高的场景
  - 多 Python 版本兼容性测试[9](https://blog.csdn.net/qq_35082030/article/details/146549871)

------

**Pixi：跨平台多语言包与环境管理新方案**

- **开发背景与定位**
  - Pixi 由 Prefix.dev 开发，基于 Conda 生态，采用 Rust 实现，专为多语言、多平台项目设计[4](https://blog.csdn.net/PengWon/article/details/144181858),[5](https://hpc.nmsu.edu/discovery/software/sstack/types/pixi/),[10](https://blog.csdn.net/gitblog_00034/article/details/138840452),[11](https://www.oschina.net/p/prefix-dev-pixi)。
  - 目标是提供类似 Cargo、Yarn 的现代化体验，但不限于 Python，还支持 R、C/C++、Rust、Ruby 等多种语言[5](https://hpc.nmsu.edu/discovery/software/sstack/types/pixi/),[10](https://blog.csdn.net/gitblog_00034/article/details/138840452),[11](https://www.oschina.net/p/prefix-dev-pixi)。
- **核心技术与功能**
  - **跨平台与多语言**：支持 Linux、macOS、Windows，能管理 Python、R、C/C++、Rust 等多种语言依赖[4](https://blog.csdn.net/PengWon/article/details/144181858),[5](https://hpc.nmsu.edu/discovery/software/sstack/types/pixi/),[10](https://blog.csdn.net/gitblog_00034/article/details/138840452),[11](https://www.oschina.net/p/prefix-dev-pixi)。
  - **项目隔离与环境管理**：每个项目独立环境，依赖通过 pixi.toml 配置，避免全局冲突，支持精确版本控制和锁文件（pixi.lock）[4](https://blog.csdn.net/PengWon/article/details/144181858),[5](https://hpc.nmsu.edu/discovery/software/sstack/types/pixi/),[8](https://prefix.dev/blog/introducing_multi_env_pixi),[10](https://blog.csdn.net/gitblog_00034/article/details/138840452)。
  - **多环境与特性管理**：Pixi 支持为不同开发任务（如测试、开发、文档构建）定义独立环境（features），可灵活组合并复用依赖配置[8](https://prefix.dev/blog/introducing_multi_env_pixi)。
  - **任务系统**：在 pixi.toml 中定义常用任务，通过 `pixi run` 快速执行，如测试、启动脚本等，提升开发自动化水平[4](https://blog.csdn.net/PengWon/article/details/144181858),[5](https://hpc.nmsu.edu/discovery/software/sstack/types/pixi/),[8](https://prefix.dev/blog/introducing_multi_env_pixi)。
  - **缓存与加速**：自动缓存已下载包，提高后续安装速度[4](https://blog.csdn.net/PengWon/article/details/144181858)。
  - **简洁 CLI 与全局/项目安装**：命令行体验类似 Cargo，支持全局或项目级工具安装，适合个人和团队开发[5](https://hpc.nmsu.edu/discovery/software/sstack/types/pixi/)[10](https://blog.csdn.net/gitblog_00034/article/details/138840452),[11](https://www.oschina.net/p/prefix-dev-pixi)。
- **典型应用场景**
  - 多语言项目开发、数据科学、跨平台团队协作
  - 需统一依赖环境、自动化任务和高效包管理的复杂项目[4](https://blog.csdn.net/PengWon/article/details/144181858),[5](https://hpc.nmsu.edu/discovery/software/sstack/types/pixi/),[8](https://prefix.dev/blog/introducing_multi_env_pixi),[10](https://blog.csdn.net/gitblog_00034/article/details/138840452),[11](https://www.oschina.net/p/prefix-dev-pixi)

## UV 与 Pixi 对比

| 特性           | UV（专注 Python）                         | Pixi（多语言/多平台）                      |
| -------------- | ----------------------------------------- | ------------------------------------------ |
| 实现语言       | Rust                                      | Rust                                       |
| 生态基础       | Python（PyPI）                            | Conda                                      |
| 支持语言       | 仅 Python                                 | Python、R、C/C++、Rust、Ruby 等多语言      |
| 平台支持       | Linux、macOS、Windows                     | Linux、macOS、Windows                      |
| 环境隔离       | 项目级 .venv                              | 项目级 .pixi                               |
| 依赖配置       | pyproject.toml、requirements.txt、uv.lock | pixi.toml、pixi.lock                       |
| Python版本管理 | 内置支持                                  | 通过 Conda 生态支持                        |
| 性能           | 包安装极快，适合频繁重建环境              | 依赖解析快，支持多语言多环境               |
| 多环境支持     | 支持多 Python 版本                        | 支持多特性、多环境（如测试、开发、生产等） |
| 任务系统       | 支持运行脚本                              | 内置任务系统（pixi run）                   |
| 适用场景       | 纯 Python 项目，追求极致性能、环境一致性  | 多语言、多平台项目，需统一依赖和自动化任务 |

------

## 总结

- **UV** 适合纯 Python 项目开发，追求极致包管理性能和一体化体验，尤其适合需要频繁重建环境、CI/CD、教学和多 Python 版本测试的场景。
- **Pixi** 适合多语言、多平台项目，强调跨语言依赖管理、环境一致性和自动化任务，适用于数据科学、科研、跨平台团队协作等复杂开发需求。

# [Installation](https://docs.astral.sh/uv/#installation)

Install uv with our official standalone installer:

macOS and Linux

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or install use pip

```shell
pip install uv
pipx install uv
...
```

Homebrew

```shell
brew install uv
```

# [Upgrading uv](https://docs.astral.sh/uv/getting-started/installation/#upgrading-uv)

```shell
uv self update
```

When another installation method is used, self-updates are disabled. Use the package manager's upgrade method instead. For example, with `pip`:

```shell
pip install -U uv
```

# [Shell autocompletion](https://docs.astral.sh/uv/getting-started/installation/#shell-autocompletion)

> You can run `echo $SHELL` to help you determine your shell.

To enable shell autocompletion for uv commands, run one of the following:

Bash

```shell
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
```

Zsh

```shell
echo 'eval "$(uv generate-shell-completion zsh)"' >> ~/.zshrc
```

fish

```shell
echo 'uv generate-shell-completion fish | source' > ~/.config/fish/completions/uv.fish
```

elvish

```shell
echo 'eval (uv generate-shell-completion elvish | slurp)' >> ~/.elvish/rc.elv
```

powershell

```shell
if (!(Test-Path -Path $PROFILE)) {
  New-Item -ItemType File -Path $PROFILE -Force
}
Add-Content -Path $PROFILE -Value '(& uv generate-shell-completion powershell) | Out-String | Invoke-Expression'
```

To enable shell autocompletion for uv commands, run one of the following:

Bash

```shell
echo 'eval "$(uvx --generate-shell-completion bash)"' >> ~/.bashrc
```

Zsh

```shell
echo 'eval "$(uvx --generate-shell-completion zsh)"' >> ~/.zshrc
```

fish

```shell
echo 'uvx --generate-shell-completion fish | source' > ~/.config/fish/completions/uvx.fish
```

elvish

```shell
echo 'eval (uvx --generate-shell-completion elvish | slurp)' >> ~/.elvish/rc.elv
```

powershell

```shell
if (!(Test-Path -Path $PROFILE)) {
  New-Item -ItemType File -Path $PROFILE -Force
}
Add-Content -Path $PROFILE -Value '(& uvx --generate-shell-completion powershell) | Out-String | Invoke-Expression'
```

# init

Create a new project

```shell
uv init [PATH]

uv init .  # 初始化当前文件夹
```

# sync

Update the project's environment

同步项目环境。

- 用法：`uv sync`
- 功能：根据锁文件（`uv.lock`）将依赖安装到虚拟环境，确保环境与锁文件一致。默认会移除未在依赖列表中的多余包，实现“精确同步”。可用 `--inexact` 保留多余包[2](https://docs.astral.sh/uv/concepts/projects/sync/),[10](https://docs.astral.sh/uv/reference/cli/)。

# add

Add dependencies to the project

添加依赖。

- 用法：`uv add [包名]`
- 功能：将指定依赖添加到 `pyproject.toml`，并自动解析依赖、更新锁文件和虚拟环境。支持添加普通依赖、开发依赖（`--dev`）、可选依赖（`--optional`）、指定版本、指定源（如 Git、路径、本地文件）等[3](https://docs.astral.sh/uv/concepts/projects/dependencies/),[10](https://docs.astral.sh/uv/reference/cli/),[11](https://gihyo.jp/article/2024/09/monthly-python-2409),[12](https://www.datacamp.com/tutorial/python-uv)。

```shell
uv add numpy
uv add pandas==2.2.0
```

## [add pytorch](https://docs.astral.sh/uv/guides/integration/pytorch/#installing-pytorch)

### edit `pyproject.toml`

In this case, PyTorch would be installed from PyPI, which hosts CPU-only wheels for Windows and macOS, and GPU-accelerated wheels on Linux (targeting CUDA 12.6):

```toml
[project]
name = "project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.7.1",
    "torchvision>=0.22.1",
    "torchaudio>=2.7.1",
]
```

In such cases, the first step is to add the relevant PyTorch index to your `pyproject.toml`:

````toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
````

We recommend the use of `explicit = true` to ensure that the index is *only* used for `torch`, `torchvision`, and other PyTorch-related packages, as opposed to generic dependencies like `jinja2`, which should continue to be sourced from the default index (PyPI).

Next, update the `pyproject.toml` to point `torch` and `torchvision` to the desired index:

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchaudio = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
```

In some cases, you may want to use CPU-only builds in one environment (e.g., macOS), and CUDA-enabled builds in another (e.g., Linux and Windows).

With `tool.uv.sources`, you can use environment markers to specify the desired index for each platform. For example, the following configuration would use PyTorch's CUDA-enabled builds on Linux, and CPU-only builds on all other platforms (e.g., macOS and Windows):

```toml
[project]
name = "project"
version = "0.1.0"
requires-python = ">=3.12.0"
dependencies = [
  "torch>=2.7.1",
  "torchvision>=0.22.1",
  "torchaudio>=2.7.1",
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", marker = "sys_platform == 'macos'" },
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
    { index = "pytorch-cpu", marker = "sys_platform == 'macos'" },
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchaudio = [
    { index = "pytorch-cpu", marker = "sys_platform == 'macos'" },
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

### [use `uv pip` interface](https://docs.astral.sh/uv/guides/integration/pytorch/#the-uv-pip-interface)

While the above examples are focused on uv's project interface (`uv lock`, `uv sync`, `uv run`, etc.), PyTorch can also be installed via the `uv pip` interface.

PyTorch itself offers a [dedicated interface](https://pytorch.org/get-started/locally/) to determine the appropriate pip command to run for a given target configuration. For example, you can install stable, CPU-only PyTorch on Linux with:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

To use the same workflow with uv, replace `pip3` with `uv pip`:

```shell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

# uv pip list

查看安装的包

# 更新包

**1. 更新单个依赖包**

- 使用 `uv add` 命令可以将某个依赖升级到最新版或指定版本。例如：

  - 升级到最新版：

    ```shell
    uv add requests
    ```
    
    这会将 `requests` 升级到可用的最新版本，并自动更新 `pyproject.toml` 和 `uv.lock` 文件[5](https://www.datacamp.com/tutorial/python-uv)。
    
  - 升级到指定版本：
  
    ```shell
    uv add requests==2.31.0
    ```
    
  
  这样可以精确锁定依赖版本[5](https://www.datacamp.com/tutorial/python-uv)。

------

**2. 更新所有依赖包**

- 目前 uv 官方尚未直接支持一键更新所有依赖（类似 `poetry update`），但可以通过以下方法实现：

  - 使用 `uv lock --upgrade`：

    ```shell
    uv lock --upgrade
    ```
    
    该命令会将所有已锁定依赖升级到其允许的最新版本，并更新 `uv.lock` 文件，但不会自动修改 `pyproject.toml` 里的版本约束[4](https://github.com/astral-sh/uv/issues/6692)。
    
  - 之后运行 `uv sync`，将环境同步到最新依赖：
  
    ```shell
    uv sync
    ```
    
  
  这样虚拟环境中的包就会被实际升级到最新版本[3](https://github.com/astral-sh/uv/issues/1419),[4](https://github.com/astral-sh/uv/issues/6692)。

------

**3. 工作流建议**

- 如果只想升级某几个包，可以多次使用 `uv add 包名`。
- 如果希望所有依赖都升级，推荐先运行 `uv lock --upgrade`，再 `uv sync`，这样可以批量升级所有依赖到允许的最新版本[3](https://github.com/astral-sh/uv/issues/1419)[4](https://github.com/astral-sh/uv/issues/6692)。

# remove

Remove dependencies from the project

**uv remove**
移除依赖。

- 用法：`uv remove [包名]`
- 功能：从 `pyproject.toml` 移除指定依赖，同时更新锁文件和虚拟环境。会卸载该依赖及其子依赖（如果没有被其他包依赖）[3](https://docs.astral.sh/uv/concepts/projects/dependencies/),[10](https://docs.astral.sh/uv/reference/cli/),[12](https://www.datacamp.com/tutorial/python-uv)。

# version

Read or update the project's version

读取或更新项目版本。

- 用法：`uv version [新版本号]`
- 功能：读取当前项目版本或设置新版本号，直接修改 `pyproject.toml` 中的 version 字段[10](https://docs.astral.sh/uv/reference/cli/)。

# lock

Update the project's lockfile

锁定依赖版本。

- 用法：`uv lock`
- 功能：解析当前依赖，生成或更新锁文件（`uv.lock`）。可用 `--check` 检查锁文件是否最新。锁定操作保证依赖版本一致性，但不会自动升级到依赖的最新版本，需手动更新[2](https://docs.astral.sh/uv/concepts/projects/sync/),[6](https://docs.astral.sh/uv/pip/compile/),[10](https://docs.astral.sh/uv/reference/cli/)。

# export

Export the project's lockfile to an alternate format

导出依赖树或环境文件。

- 用法：`uv export [选项]`
- 功能：将项目依赖导出为 `requirements.txt`、`pylock.toml` 等格式，便于与其他工具或平台兼容[6](https://docs.astral.sh/uv/pip/compile/),[10](https://docs.astral.sh/uv/reference/cli/)。

# tree

Display the project's dependency tree

显示依赖树。

- 用法：`uv tree [选项]`
- 功能：以树状结构展示当前项目或工作区的依赖关系，包括依赖分组、子项目依赖等。可用 `--no-dev` 排除开发依赖，用 `--project` 查看指定项目的依赖树[8](https://github.com/astral-sh/uv/issues/8719),[10](https://docs.astral.sh/uv/reference/cli/)。

# run

**功能与用途：**

- `uv run` 用于在项目的虚拟环境中执行命令或脚本，无需手动激活环境或担心依赖未同步。
- 每次运行时，uv 会自动确保环境已同步和最新，然后在隔离环境中执行命令[1,](https://docs.astral.sh/uv/concepts/projects/run/)[6,](https://www.saaspegasus.com/guides/uv-deep-dive/)[7](https://www.datacamp.com/tutorial/python-uv)。

**典型用法：**

- 运行 Python 脚本：

  ```shell
  uv run hello.py
  ```

  这相当于在虚拟环境中执行 `python hello.py`，且无需手动激活环境[2](https://docs.astral.sh/uv/guides/scripts/),[4](https://docs.astral.sh/uv/reference/cli/),[7](https://www.datacamp.com/tutorial/python-uv)。

- 运行带参数的脚本或命令：

  ```shell
  uv run manage.py runserver
  ```

  适合 Django、Flask 等项目的开发命令[6](https://www.saaspegasus.com/guides/uv-deep-dive/)。

- 运行 shell 命令或自定义 CLI：

  ```shell
  uv run example-cli foo
  uv run bash scripts/foo.sh
  ```

  支持运行项目环境下可用的任意命令[1](https://docs.astral.sh/uv/concepts/projects/run/)。

- 临时添加依赖运行命令：

  ```shell
  uv run --with httpx==0.26.0 python -c "import httpx; print(httpx.__version__)"
  ```

  可以为本次运行临时指定依赖及其版本，无需修改项目依赖文件[1](https://docs.astral.sh/uv/concepts/projects/run/)。

- 跳过项目环境（如仅运行脚本）：

  ```shell
  uv run --no-project example.py
  ```

  跳过项目依赖，仅用当前环境运行脚本[2](https://docs.astral.sh/uv/guides/scripts/),[4](https://docs.astral.sh/uv/reference/cli/)。

- 运行 `ruff`

  ```shell
  uv run ruff format
  ```

**其他特性：**

- 支持通过 stdin 或 URL 运行 Python 脚本。
- 自动处理信号转发，保证命令行工具的交互体验[1](https://docs.astral.sh/uv/concepts/projects/run/),[4](https://docs.astral.sh/uv/reference/cli/)。
- 可作为传统 `python` 命令的替代，适用于所有需要依赖环境的场景[6](https://www.saaspegasus.com/guides/uv-deep-dive/),[7](https://www.datacamp.com/tutorial/python-uv)。

# tool

Run and install commands provided by Python packages

**功能与用途：**

- `uv tool` 用于运行或安装以 Python 包形式发布的命令行工具（如 black、ruff、pytest 等），无需将其安装到项目环境，避免依赖污染[3,](https://docs.astral.sh/uv/guides/tools/)[5](https://docs.astral.sh/uv/concepts/tools/),[7](https://www.datacamp.com/tutorial/python-uv)。
- `uv tool run <工具>` 或更简洁的 `uvx <工具>`，可临时调用工具，自动在隔离的缓存环境中安装并运行[3](https://docs.astral.sh/uv/guides/tools/),[5](https://docs.astral.sh/uv/concepts/tools/)。

**典型用法：**

- 临时运行工具（推荐使用 uvx 别名）：

  ```shell
  uv tool run black hello.py
  uvx black hello.py
  ```

  工具及依赖会被安装到 uv 的缓存虚拟环境，环境自动复用和清理[3](https://docs.astral.sh/uv/guides/tools/),[5](https://docs.astral.sh/uv/concepts/tools/),[7](https://www.datacamp.com/tutorial/python-uv)。

- 指定工具版本或包来源：

  ```shell
  uvx ruff@0.3.0 check
  uvx --from httpie http
  ```

  支持指定工具版本，或当命令名与包名不一致时用 `--from` 指定[3](https://docs.astral.sh/uv/guides/tools/),[5](https://docs.astral.sh/uv/concepts/tools/)。

- 安装工具到全局环境（持久化）：

  ```shell
  uv tool install ruff
  ```

  安装后工具可直接在 shell 中调用，适合频繁使用[3](https://docs.astral.sh/uv/guides/tools/),[5](https://docs.astral.sh/uv/concepts/tools/)。

- 运行 `ruff`

  ```shell
   uv tool run ruff format
  ```

**其他特性：**

- 工具环境与项目环境完全隔离，避免依赖冲突[5](https://docs.astral.sh/uv/concepts/tools/)。
- 工具缓存于 uv 的 cache 目录，清理 cache 会自动移除[5](https://docs.astral.sh/uv/concepts/tools/),[7](https://www.datacamp.com/tutorial/python-uv)。
- `uv tool run`/`uvx` 适合偶尔用到的开发工具，`uv tool install` 适合需要长期可用的工具[3](https://docs.astral.sh/uv/guides/tools/),[5](https://docs.astral.sh/uv/concepts/tools/)。
- 如果工具需要访问项目本身（如 pytest/mypy），建议用 `uv run` 而非 `uv tool run`，以便在项目环境下执行[5](https://docs.astral.sh/uv/concepts/tools/)。

# python

Manage Python versions and installations

- 用法：`uv python <子命令>`
- 作用：管理 Python 解释器版本，包括安装、切换、列举等。可用于指定项目所需的 Python 版本。
- 典型操作（部分命令可能随 uv 版本变化）：
  - `uv python install 3.11.3` 安装指定版本
  - `uv python list` 显示可用/已安装版本

# pip

Manage Python packages with a pip-compatible interface

- 用法：`uv pip install <包名>`、`uv pip list`
- 作用：在 uv 管理的虚拟环境中直接调用 pip，适用于需要兼容某些只支持 pip 的工具（如 Jupyter 的 `%pip` 魔法命令）。
- 注意：uv 默认不在虚拟环境中安装 pip，需用 `uv venv --seed` 创建带 pip 的环境，或用 `uv pip install pip` 手动安装 pip[2](https://pydevtools.com/handbook/how-to/how-to-use-pip-in-a-uv-virtual-environment/)。

# venv

Create a virtual environment

- 用法：`uv venv [目录] [--seed]`
- 作用：创建新的虚拟环境。加 `--seed` 可自动安装 pip。
- 例子：
  - `uv venv .venv` 在当前目录创建虚拟环境
  - `uv venv --seed` 创建并包含 pip 的虚拟环境[2](https://pydevtools.com/handbook/how-to/how-to-use-pip-in-a-uv-virtual-environment/)。

# build

- 用法：`uv build`
- 作用：构建当前项目为 Python 包（如 wheel、sdist），用于分发或发布。
- 典型场景：准备上传 PyPI 或内部仓库前的包构建。

# publish

Upload distributions to an index

- 用法：`uv publish [选项]`
- 作用：将构建好的包发布到 PyPI 或其他 Python 包仓库。支持从 `~/.pypirc` 获取凭据，兼容 twine 等工具的配置[4](https://github.com/bulletmark/uv-publish/blob/main/README.md)。
- 例子：`uv publish --repository pypi`
- 也可用社区工具 `uv-publish` 简化凭据管理[4](https://github.com/bulletmark/uv-publish/blob/main/README.md)。

# cache

Manage uv's cache

- 用法：`uv cache <子命令>`
- 作用：管理 uv 的包缓存，包括清理、修剪、查看等。
- 常用命令：
  - `uv cache clean` 清理所有缓存
  - `uv cache clean <包名>` 清理指定包缓存
  - `uv cache prune` 清理未使用的缓存
  - `uv cache prune --ci` 按 CI 场景优化缓存，仅保留自建 wheel[5](https://docs.astral.sh/uv/concepts/cache/)
- 特点：缓存机制线程安全，支持多进程并发操作[5](https://docs.astral.sh/uv/concepts/cache/)。

# self

Manage the uv executable

- 用法：`uv self upgrade`
- 作用：升级 uv 自身到最新版。
- 例子：`uv self upgrade`

# generate-shell-completion

Generate shell completion

- 用法：`uv generate-shell-completion <shell>`
- 作用：生成 shell 补全脚本，支持 bash、zsh、fish 等主流 shell。
- 例子：`uv generate-shell-completion zsh`

# help

```shell
> uv help
An extremely fast Python package manager.

Usage: uv [OPTIONS] <COMMAND>

Commands:
  run                        Run a command or script
  init                       Create a new project
  add                        Add dependencies to the project
  remove                     Remove dependencies from the project
  version                    Read or update the project's version
  sync                       Update the project's environment
  lock                       Update the project's lockfile
  export                     Export the project's lockfile to an alternate format
  tree                       Display the project's dependency tree
  tool                       Run and install commands provided by Python packages
  python                     Manage Python versions and installations
  pip                        Manage Python packages with a pip-compatible interface
  venv                       Create a virtual environment
  build                      Build Python packages into source distributions and wheels
  publish                    Upload distributions to an index
  cache                      Manage uv's cache
  self                       Manage the uv executable
  generate-shell-completion  Generate shell completion
  help                       Display documentation for a command

Cache options:
  -n, --no-cache               Avoid reading from or writing to the cache, instead using a temporary directory for the
                               duration of the operation [env: UV_NO_CACHE=]
      --cache-dir <CACHE_DIR>  Path to the cache directory [env: UV_CACHE_DIR=]

Python options:
      --managed-python       Require use of uv-managed Python versions [env: UV_MANAGED_PYTHON=]
      --no-managed-python    Disable use of uv-managed Python versions [env: UV_NO_MANAGED_PYTHON=]
      --no-python-downloads  Disable automatic downloads of Python. [env: "UV_PYTHON_DOWNLOADS=never"]

Global options:
  -q, --quiet...
          Use quiet output
  -v, --verbose...
          Use verbose output
      --color <COLOR_CHOICE>
          Control the use of color in output [possible values: auto, always, never]
      --native-tls
          Whether to load TLS certificates from the platform's native certificate store [env: UV_NATIVE_TLS=]
      --offline
          Disable network access [env: UV_OFFLINE=]
      --allow-insecure-host <ALLOW_INSECURE_HOST>
          Allow insecure connections to a host [env: UV_INSECURE_HOST=]
      --no-progress
          Hide all progress outputs [env: UV_NO_PROGRESS=]
      --directory <DIRECTORY>
          Change to the given directory prior to running the command
      --project <PROJECT>
          Run the command within the given project directory [env: UV_PROJECT=]
      --config-file <CONFIG_FILE>
          The path to a `uv.toml` file to use for configuration [env: UV_CONFIG_FILE=]
      --no-config
          Avoid discovering configuration files (`pyproject.toml`, `uv.toml`) [env: UV_NO_CONFIG=]
  -h, --help
          Display the concise help for this command
  -V, --version
          Display the uv version

Use `uv help <command>` for more information on a specific command.
```

