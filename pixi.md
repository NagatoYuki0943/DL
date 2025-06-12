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

# Install

To install `pixi` you can run the following command in your terminal:

## Linux & macOS

```shell
curl -fsSL https://pixi.sh/install.sh | sh
curl -fsSL https://pixi.sh/install.sh | zsh
curl -fsSL https://pixi.sh/install.sh | fish
```

If your system doesn't have `curl`, you can use `wget`:

```shell
wget -qO- https://pixi.sh/install.sh | sh
wget -qO- https://pixi.sh/install.sh | zsh
wget -qO- https://pixi.sh/install.sh | fish
```

Use Homebrew

```shell
brew install pixi
```

## Windows

https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.msi

Or run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

Use Winget

```powershell
winget install prefix-dev.pixi
```

Use Scoop

```powershell
scoop install main/pixi
```

# Self Update

Updating is as simple as installing, rerunning the installation script gets you the latest version.

```shell
pixi self-update
```

Or get a specific Pixi version using:

```shell
pixi self-update --version x.y.z
```

# Autocompletion

To get autocompletion follow the instructions for your shell. Afterwards, restart the shell or source the shell config file.

## Bash (default on most Linux systems)

Add the following to the end of `~/.bashrc`:

**~/.bashrc**

```shell
eval "$(pixi completion --shell bash)"
```

## Zsh (default on macOS)

Add the following to the end of `~/.zshrc`:

**~/.zshrc**

```shell
autoload -Uz compinit && compinit  # redundant with Oh My Zsh
eval "$(pixi completion --shell zsh)"
```

## PowerShell (pre-installed on all Windows systems)

Add the following to the end of `Microsoft.PowerShell_profile.ps1`. You can check the location of this file by querying the `$PROFILE` variable in PowerShell. Typically the path is `~\Documents\PowerShell\Microsoft.PowerShell_profile.ps1` or `~/.config/powershell/Microsoft.PowerShell_profile.ps1` on -Nix.

```powershell
(& pixi completion --shell powershell) | Out-String | Invoke-Expression
```

## Fish

Add the following to the end of `~/.config/fish/config.fish`:

**~/.config/fish/config.fish**

```shell
pixi completion --shell fish | source
```

# config

## detached-environments

**default location**

cmd

```shell
pixi config set detached-environments true
```

config.toml

```
detached-environments = true
```

or:

**custom location**

cmd

```shell
pixi config set detached-environments /opt/pixi/envs
```

config.toml

```
detached-environments = "/opt/pixi/envs"
```

`pixi config help`

```shell
Configuration management

Usage: pixi.exe config [OPTIONS] <COMMAND>

Commands:
  edit     Edit the configuration file
  list     List configuration values [aliases: ls]
  prepend  Prepend a value to a list configuration key
  append   Append a value to a list configuration key
  set      Set a configuration value
  unset    Unset a configuration value
  help     Print this message or the help of the given subcommand(s)

Global Options:
  -h, --help           Display help information
  -v, --verbose...     Increase logging verbosity (-v for warnings, -vv for info, -vvv for debug, -vvvv for trace)
  -q, --quiet...       Decrease logging verbosity (quiet mode)
      --color <COLOR>  Whether the log needs to be colored [env: PIXI_COLOR=] [default: auto] [possible values: always,
                       never, auto]
      --no-progress    Hide all progress bars, always turned on if stderr is not a terminal [env: PIXI_NO_PROGRESS=]
```

# init

初始化

```shell
pixi init [PATH] # [PATH] 默认为本目录 .
pixi init --format pyproject # 使用 pyproject 格式初始化
```

如果你的 `pyproject.toml` 还没有 Pixi 配置，可以在项目根目录下运行：

```shell
pixi init
```

这会自动在 `pyproject.toml` 或 `pixi.toml` 中补充必要的 `[tool.pixi.project]`、`channels`、`platforms` 等 Pixi 配置段，并不会覆盖你已有的依赖声明。

# install

`pixi install` 命令用于根据项目中的依赖清单（`pixi.toml` 或 `pyproject.toml`）和锁定文件（`pixi.lock`）安装或同步环境。它是 Pixi 管理项目依赖和环境的核心命令之一，主要功能和使用方式如下：

主要功能

- **安装环境**：根据 `pixi.lock` 文件中的内容，安装或同步项目的依赖环境。如果 `pixi.lock` 不存在或与依赖清单不一致，会自动生成或更新 lockfile。
- **环境可复现**：确保所有依赖的版本与 `pixi.lock` 文件一致，从而实现环境的可复现性和一致性。
- **支持多环境**：如果项目定义了多个环境（如 `default`、`test`、`lint` 等），可以通过参数选择要安装的环境。

# add

往虚拟环境中添加包

```shell
pixi add python numpy pytest
pixi add numpy=1.23.5
pixi add 'numpy>=1.20,<1.24'

# 使用 pypi 安装
pixi add numpy --pypi
# 安装 pypi 可选项
pixi add "flask[async]==3.1.0" --pypi
```

## 安装 pytorch(cuda)

[Pytorch Installation - Pixi by prefix.dev](https://pixi.sh/dev/python/pytorch/#troubleshooting)

### Installing from PyPi

最新版本 pytorch 只提供 pypi 包，因此只推荐这种用法

` pyproject.toml`

```toml
[tool.pixi.pypi-dependencies]
torch = { version = ">=2.7.1", index = "https://download.pytorch.org/whl/cu128" }
torchvision = { version = ">=0.22.1", index = "https://download.pytorch.org/whl/cu128" }
torchaudio = { version = ">=2.7.1", index = "https://download.pytorch.org/whl/cu128" }
```

### Installing from Conda-forge

` pyproject.toml`

```toml
[tool.pixi.system-requirements]
cuda = "12.0"

[tool.pixi.dependencies]
pytorch-gpu = "*"
cuda-version = "12.4.*"
```

### Installing from PyTorch channel

 `pyproject.toml`

```toml
[tool.pixi.project]
# `main` is not free! It's a paid channel for organizations over 200 people.
channels = ["main", "nvidia", "pytorch"]
platforms = ["osx-arm64", "linux-64", "win-64"]

[tool.pixi.feature.gpu.system-requirements]
cuda = "12.0"

[tool.pixi.dependencies]
pytorch = "*"

[tool.pixi.environments]
gpu = ["gpu"]
```

## 安装 pytorch(xpu)

### Installing from PyPi

https://pytorch-extension.intel.com/installation

` pyproject.toml`

```toml
[tool.pixi.pypi-dependencies]
torch = { version = ">=2.7.1", index = "https://download.pytorch.org/whl/xpu" }
torchvision = { version = ">=0.22.1", index = "https://download.pytorch.org/whl/xpu" }
torchaudio = { version = ">=2.7.1", index = "https://download.pytorch.org/whl/xpu" }
intel-extension-for-pytorch = { version = ">=2.7.10", index = "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/" }
```

# list

列出安装的包

# update

- **作用**：检查依赖是否有新版本，并在**不改变 manifest 文件（如 pixi.toml/pyproject.toml）中的版本约束**的前提下，更新 `pixi.lock` 文件和实际环境到允许的最新版本。
- **举例**：如果 manifest 文件中写的是 `numpy = ">=1.20,<1.24"`，则 `pixi update numpy` 会把 numpy 升级到 1.23.x（即允许范围内的最新版本），但不会改动 manifest 文件的版本约束[1](https://pixi.sh/latest/reference/cli/pixi/update/),[3](https://pixi.sh/v0.41.1/reference/cli/),[4](https://prefix-dev.github.io/pixi/v0.23.0/reference/cli/)。
- **适用场景**：希望保持当前依赖约束不变，只让 lockfile 和环境中的包尽可能新。

- `update` 保守升级，仅更新 lockfile 和环境，不碰 manifest 文件。

| 命令         | 影响范围                 | 是否修改 manifest | 是否修改 lockfile | 依赖约束变化       | 典型用途                     |
| ------------ | ------------------------ | ----------------- | ----------------- | ------------------ | ---------------------------- |
| pixi update  | lockfile、环境           | 否                | 是                | 不变（守约束范围） | 保持当前约束，升级到可用最新 |
| pixi upgrade | manifest、lockfile、环境 | 是                | 是                | 放宽到最新版本     | 跟随上游，始终用最新依赖     |

例子

```shell
pixi update numpy
```

# upgrade

- **作用**：检查依赖是否有新版本，并**直接修改 manifest 文件中的版本约束**，使其允许最新版本，同时更新 lockfile 和环境。
- **举例**：如果 manifest 文件中写的是 `numpy = "1.21.0"`，执行 `pixi upgrade numpy` 后，会把 manifest 文件中的约束改为最新版本（如 `numpy = "1.25.0"`），并同步 lockfile 和环境[2](https://pixi.sh/latest/reference/cli/pixi/upgrade/),[3](https://pixi.sh/v0.41.1/reference/cli/)。
- **更进一步**：对于指定的包（如 `pixi upgrade numpy`），manifest 文件中除了保留 `build`（通配符）、`channel`、`file_name`、`url`、`subdir` 字段外，其它字段会被移除，仅保留最宽松的约束[2](https://pixi.sh/latest/reference/cli/pixi/upgrade/)。
- **适用场景**：希望包的版本约束也自动跟随最新版本，适合想要长期保持依赖“最前沿”的项目。

- `upgrade` 激进升级，连 manifest 文件的版本约束也会跟着升级到最新[1](https://pixi.sh/latest/reference/cli/pixi/update/),[2,](https://pixi.sh/latest/reference/cli/pixi/upgrade/)[3](https://pixi.sh/v0.41.1/reference/cli/),[4](https://prefix-dev.github.io/pixi/v0.23.0/reference/cli/)。

| 命令         | 影响范围                 | 是否修改 manifest | 是否修改 lockfile | 依赖约束变化       | 典型用途                     |
| ------------ | ------------------------ | ----------------- | ----------------- | ------------------ | ---------------------------- |
| pixi update  | lockfile、环境           | 否                | 是                | 不变（守约束范围） | 保持当前约束，升级到可用最新 |
| pixi upgrade | manifest、lockfile、环境 | 是                | 是                | 放宽到最新版本     | 跟随上游，始终用最新依赖     |

例子

```shell
pixi upgrade numpy
```

# remove

```shell
pixi remove numpy
# 移除 pypi 包
pixi add numpy --pypi
```

# search

查询 conda 包

```shell
pixi search numpy
```

# lock

`pixi lock` 命令用于生成或更新项目的锁定文件（`pixi.lock`），确保依赖环境的可复现性和一致性。它的核心作用和机制如下：

作用与原理

- **锁定环境**：`pixi.lock` 文件会详细记录项目所有依赖（包括 conda 包和 PyPI 包）的具体版本、下载渠道、平台信息等，使得无论在哪台机器上用 Pixi 安装环境，都能得到完全一致的依赖树和包版本[2](https://pixi.sh/latest/features/lockfile/)。
- **与配置文件同步**：`pixi.lock` 总是与 `pixi.toml` 或 `pyproject.toml` 等依赖清单保持同步。如果依赖清单有变动（如添加、删除或更改依赖），需要重新生成锁文件以反映最新的依赖状态[2](https://pixi.sh/latest/features/lockfile/)。
- **不可手动编辑**：锁文件是机器生成的，虽然是人类可读的，但不建议手动修改，推荐纳入版本控制（如 git）以便团队协作和环境回溯[2](https://pixi.sh/latest/features/lockfile/)。

典型用法

- **生成或更新锁文件**

  ```shell
  pixi lock
  ```

  该命令会解析依赖清单，解析所有依赖的兼容版本，并生成（或更新）`pixi.lock` 文件。

- **常见场景**

  - 在 CI/CD 流程中，确保环境可复现。
  - 团队协作时，保证每个人的开发环境一致。
  - 依赖发生变更后，及时更新锁文件。

- **与其他命令的关系**

  - 许多 Pixi 命令（如 `pixi install`, `pixi add`, `pixi remove` 等）在需要时会自动生成或更新锁文件，无需每次手动运行 `pixi lock`[2](https://pixi.sh/latest/features/lockfile/)。
  - 但在某些场景下（如只想刷新 lockfile 而不安装环境），可以单独运行 `pixi lock`。

锁文件内容

- 记录所有环境（如 default、test 等）的完整包列表及其精确版本。
- 记录包的来源渠道、平台、哈希等详细信息，确保可追溯和安全[2](https://pixi.sh/latest/features/lockfile/)。
- 版本号与 Pixi 兼容，确保不同 Pixi 版本间的兼容性。

# clean

```shell
pixi clean        # 清理虚拟环境
pixi clean cache  # 清理 pixi 缓存
```

# reinstall

重新安装环境

# shell

进入虚拟环境

`exit` 命令用来退出虚拟环境

# task

- **常见操作**：

  - **添加任务**

    ```shell
    pixi task add <task-name> "<command>" [--cwd <directory>]
    ```

    例如：

    ```shell
    pixi task add test "pytest"
    pixi task add build "python setup.py build" --cwd src
    ```

    这会在 `pixi.toml` 的 `[tasks]` 段落下添加对应任务定义[3](https://prefix.dev/docs/pixi/advanced/advanced_tasks)。

  - **任务别名与依赖**
    可以让一个任务依赖于其他任务。例如：

    ```shell
    [tasks]
    fmt = "ruff"
    lint = "pylint"
    style = [{ task = "fmt" }, { task = "lint" }]
    ```

    这样 `pixi run style` 会依次执行 `fmt` 和 `lint`[1](https://pixi.sh/latest/workspace/advanced_tasks/),[2](https://prefix-dev.github.io/pixi/v0.30.0/features/advanced_tasks/)。

  - **高级配置**

    - 支持设置工作目录（cwd）、环境变量（env）、输入输出缓存（inputs/outputs）、任务依赖（depends-on）、参数（args）等[2](https://prefix-dev.github.io/pixi/v0.30.0/features/advanced_tasks/)[3](https://prefix.dev/docs/pixi/advanced/advanced_tasks)。
    - 任务可以定义参数，实现更灵活的自动化脚本[1](https://pixi.sh/latest/workspace/advanced_tasks/),[2](https://prefix-dev.github.io/pixi/v0.30.0/features/advanced_tasks/)。

# run

**用途**：用于执行在 `pixi.toml`（或 `pyproject.toml`）中定义的任务。
**语法**：

```shell
pixi run <task-name> [args...]
```

**用法说明**：

- **执行单个任务**

  ```shell
  pixi run test
  ```

  会在 Pixi 环境下运行 `pytest`[8](https://holoviews.org/developer_guide/index.html),[10](https://contrib.scikit-learn.org/project-template/quick_start.html)。

- **带参数执行任务**
  如果任务定义了参数，可以在命令行传递：

  ```shell
  [tasks.greet]
  args = ["name"]
  cmd = "echo Hello, {{ name }}!"
  ```

  执行：

  ```shell
  pixi run greet Alice
  ```

  输出：`Hello, Alice!`[1](https://pixi.sh/latest/workspace/advanced_tasks/),[2](https://prefix-dev.github.io/pixi/v0.30.0/features/advanced_tasks/)

- **执行依赖任务或任务别名**
  如果任务依赖其他任务（如 `style` 依赖 `fmt` 和 `lint`），只需：

  ```shell
  pixi run style
  ```

  Pixi 会自动按依赖顺序执行[1](https://pixi.sh/latest/workspace/advanced_tasks/),[2](https://prefix-dev.github.io/pixi/v0.30.0/features/advanced_tasks/)。

- **指定环境执行**
  如果项目有多环境，可以用：

  ```shell
  pixi run --environment <env> <task-name>
  ```

  例如：

  ```shell
  pixi run --environment test-core test-unit
  ```

  在 `test-core` 环境下运行 `test-unit` 任务[9](https://hvplot.holoviz.org/developer_guide.html)。

- **设置环境变量**
  任务可在定义时设置环境变量，也可在命令行临时覆盖：

  ```shell
  [tasks.echo]
  cmd = "echo $ARGUMENT"
  env = { ARGUMENT = "hello" }
  ```

  命令行：

  ```shell
  ARGUMENT=world pixi run echo
  ```

  输出：`world`[1](https://pixi.sh/latest/workspace/advanced_tasks/),[2](https://prefix-dev.github.io/pixi/v0.30.0/features/advanced_tasks/)

# tree

显示包依赖

# build

`pixi build` 是 Pixi 提供的用于构建项目包（如 conda 包或 Python 包）的命令，目的是将你的项目打包成可分发的格式（如 `.conda` 文件），方便分发、复用或上传到包仓库。它是 Pixi 环境与多语言生态（Python、C/C++、Rust 等）深度集成的核心功能之一。

主要功能和作用

- **构建 conda 包或 Python 包**：`pixi build` 会读取你的 `pixi.toml` 或 `pyproject.toml` 配置，根据其中的 `[package]`、`[package.build]`、`[package.host-dependencies]` 等段落，自动调用合适的构建后端（如 `pixi-build-python`）完成打包[2](https://pixi.sh/latest/build/python/),[3](https://prefix-dev.github.io/pixi/dev/build/getting_started/)。
- **生成可分发的包文件**：构建完成后，会在本地生成 `.conda` 包文件，可以用于本地安装、上传到 conda 仓库或在其他项目中复用[3](https://prefix-dev.github.io/pixi/dev/build/getting_started/)。
- **多语言支持**：不仅支持 Python，还可以用于 C/C++、Rust、R 等多种语言的包构建，统一管理依赖和打包流程[2](https://pixi.sh/latest/build/python/),[3](https://prefix-dev.github.io/pixi/dev/build/getting_started/)。
- **自动处理依赖**：构建时会自动安装 `host-dependencies`（构建时依赖）、`build-dependencies`（编译依赖）和 `run-dependencies`（运行时依赖），确保包的完整性[5](https://prefix.dev/docs/pixi/configuration)。

# info

显示信息

# help

```shell
> pixi help
Pixi [version 0.48.0] - Developer Workflow and Environment Management for Multi-Platform, Language-Agnostic Workspaces.

Pixi is a versatile developer workflow tool designed to streamline the management of your workspace's dependencies,
tasks, and environments.
Built on top of the Conda ecosystem, Pixi offers seamless integration with the PyPI ecosystem.

Basic Usage:
    Initialize pixi for a workspace:
    $ pixi init
    $ pixi add python numpy pytest

    Run a task:
    $ pixi task add test 'pytest -s'
    $ pixi run test

Found a Bug or Have a Feature Request?
Open an issue at: https://github.com/prefix-dev/pixi/issues

Need Help?
Ask a question on the Prefix Discord server: https://discord.gg/kKV8ZxyzY4

For more information, see the documentation at: https://pixi.sh

Usage: pixi.exe [OPTIONS] <COMMAND>

Commands:
  add          Adds dependencies to the workspace [aliases: a]
  auth         Login to prefix.dev or anaconda.org servers to access private channels
  build        Workspace configuration
  clean        Cleanup the environments
  completion   Generates a completion script for a shell
  config       Configuration management
  exec         Run a command and install it in a temporary environment [aliases: x]
  global       Subcommand for global package management actions [aliases: g]
  info         Information about the system, workspace and environments for the current machine
  init         Creates a new workspace
  install      Install an environment, both updating the lockfile and installing the environment [aliases: i]
  list         List workspace's packages [aliases: ls]
  lock         Solve environment and update the lock file without installing the environments
  reinstall    Re-install an environment, both updating the lockfile and re-installing the environment
  remove       Removes dependencies from the workspace [aliases: rm]
  run          Runs task in the pixi environment [aliases: r]
  search       Search a conda package
  self-update  Update pixi to the latest version or a specific version
  shell        Start a shell in a pixi environment, run `exit` to leave the shell [aliases: s]
  shell-hook   Print the pixi environment activation script
  task         Interact with tasks in the workspace
  tree         Show a tree of workspace dependencies [aliases: t]
  update       The `update` command checks if there are newer versions of the dependencies and updates the `pixi.lock`
               file and environments accordingly
  upgrade      Checks if there are newer versions of the dependencies and upgrades them in the lockfile and manifest
               file
  upload       Upload a conda package
  workspace    Modify the workspace configuration file through the command line
  help         Print this message or the help of the given subcommand(s)

Options:
  -V, --version  Print version

Global Options:
  -h, --help           Display help information
  -v, --verbose...     Increase logging verbosity (-v for warnings, -vv for info, -vvv for debug, -vvvv for trace)
  -q, --quiet...       Decrease logging verbosity (quiet mode)
      --color <COLOR>  Whether the log needs to be colored [env: PIXI_COLOR=] [default: auto] [possible values: always,
                       never, auto]
      --no-progress    Hide all progress bars, always turned on if stderr is not a terminal [env: PIXI_NO_PROGRESS=]
```

