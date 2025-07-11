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
- **UV** 和 **Pixi** 都没有查询包是否可以更新的功能，类似 `pip list --outdate`

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

# [Python](https://docs.astral.sh/uv/getting-started/features/)

Installing and managing Python itself.

- `uv python install`: Install Python versions.
- `uv python list`: View available Python versions.
- `uv python find`: Find an installed Python version.
- `uv python pin`: Pin the current project to use a specific Python version.
- `uv python uninstall`: Uninstall a Python version.
- etc

## [Install python]([Installing and managing Python | uv](https://docs.astral.sh/uv/guides/install-python/))

To install the latest Python version:

```shell
uv python install
```

## [Installing a specific version](https://docs.astral.sh/uv/guides/install-python/#installing-a-specific-version)

To install a specific Python version:

```shell
uv python install 3.13
uv python install cpython-3.13.5-windows-x86_64-none
```

To install multiple Python versions:

```shell
uv python install 3.12 3.13
```

To install a version that satisfies constraints:

```shell
uv python install '>=3.8,<3.10'
```

To install an alternative Python implementation, e.g., PyPy:

```shell
uv python install pypy@3.10
```

See the [`python install`](https://docs.astral.sh/uv/concepts/python-versions/#installing-a-python-version) documentation for more details.

## [Viewing available Python versions](https://docs.astral.sh/uv/concepts/python-versions/#viewing-available-python-versions)

View available Python versions.

```shell
uv python list
```

## [Viewing Python installations](https://docs.astral.sh/uv/guides/install-python/#viewing-python-installations)

```shell
uv python find
uv python find --system
```

## [Pin Python version](https://docs.astral.sh/uv/reference/cli/#uv-python-pin)

Pin the current project to use a specific Python version.

```shell
uv python pin
```

eg:

```shell
uv python pin 3.13
uv python pin cpython-3.13.5-windows-x86_64-none
```

## [Upgrading Python versions](https://docs.astral.sh/uv/guides/install-python/#upgrading-python-versions)

Important

Support for upgrading Python patch versions is in *preview*. This means the behavior is experimental and subject to change.

To upgrade a Python version to the latest supported patch release:

```shell
uv python upgrade 3.12
```

To upgrade all uv-managed Python versions:

```shell
uv python upgrade
```

See the [`python upgrade`](https://docs.astral.sh/uv/concepts/python-versions/#upgrading-python-versions) documentation for more details.

## [Reinstalling Python](https://docs.astral.sh/uv/guides/install-python/#reinstalling-python)

To reinstall uv-managed Python versions, use `--reinstall`, e.g.:

```shell
uv python install --reinstall
```

This will reinstall all previously installed Python versions. Improvements are constantly being added to the Python distributions, so reinstalling may resolve bugs even if the Python version does not change.

## [Uninstall Python](https://docs.astral.sh/uv/reference/cli/#uv-python-uninstall)

Uninstall Python versions

```shell
uv python uninstall
uv python uninstall 3.13
uv python uninstall cpython-3.13.5-windows-x86_64-none
```

## [Show Python installation dir](https://docs.astral.sh/uv/reference/cli/#uv-python-dir)

Show the uv Python installation directory.

```shell
uv python dir
```

# init

Create a new project

```shell
uv init [PATH]

uv init .  # 初始化当前文件夹
```

# [Configuration files](https://docs.astral.sh/uv/concepts/configuration-files/#configuration-files)

uv supports persistent configuration files at both the project- and user-level.

Specifically, uv will search for a `pyproject.toml` or `uv.toml` file in the current directory, or in the nearest parent directory.

> Note
>
> For `tool` commands, which operate at the user level, local configuration files will be ignored. Instead, uv will exclusively read from user-level configuration (e.g., `~/.config/uv/uv.toml`) and system-level configuration (e.g., `/etc/uv/uv.toml`).

In workspaces, uv will begin its search at the workspace root, ignoring any configuration defined in workspace members. Since the workspace is locked as a single unit, configuration is shared across all members.

If a `pyproject.toml` file is found, uv will read configuration from the `[tool.uv]` table. For example, to set a persistent index URL, add the following to a `pyproject.toml`:

```toml
# pyproject.toml

[[tool.uv.index]]
url = "https://test.pypi.org/simple"
default = true
```

(If there is no such table, the `pyproject.toml` file will be ignored, and uv will continue searching in the directory hierarchy.)

uv will also search for `uv.toml` files, which follow an identical structure, but omit the `[tool.uv]` prefix. For example:

```toml
# uv.toml

[[index]]
url = "https://test.pypi.org/simple"
default = true
```

> Note
>
> `uv.toml` files take precedence over `pyproject.toml` files, so if both `uv.toml` and `pyproject.toml` files are present in a directory, configuration will be read from `uv.toml`, and `[tool.uv]` section in the accompanying `pyproject.toml` will be ignored.

uv will also discover user-level configuration at `~/.config/uv/uv.toml` (or `$XDG_CONFIG_HOME/uv/uv.toml`) on macOS and Linux, or `%APPDATA%\uv\uv.toml` on Windows; and system-level configuration at `/etc/uv/uv.toml` (or `$XDG_CONFIG_DIRS/uv/uv.toml`) on macOS and Linux, or `%SYSTEMDRIVE%\ProgramData\uv\uv.toml` on Windows.

User-and system-level configuration must use the `uv.toml` format, rather than the `pyproject.toml` format, as a `pyproject.toml` is intended to define a Python *project*.

If project-, user-, and system-level configuration files are found, the settings will be merged, with project-level configuration taking precedence over the user-level configuration, and user-level configuration taking precedence over the system-level configuration. (If multiple system-level configuration files are found, e.g., at both `/etc/uv/uv.toml` and `$XDG_CONFIG_DIRS/uv/uv.toml`, only the first-discovered file will be used, with XDG taking priority.)

For example, if a string, number, or boolean is present in both the project- and user-level configuration tables, the project-level value will be used, and the user-level value will be ignored. If an array is present in both tables, the arrays will be concatenated, with the project-level settings appearing earlier in the merged array.

Settings provided via environment variables take precedence over persistent configuration, and settings provided via the command line take precedence over both.

uv accepts a `--no-config` command-line argument which, when provided, disables the discovery of any persistent configuration.

uv also accepts a `--config-file` command-line argument, which accepts a path to a `uv.toml` to use as the configuration file. When provided, this file will be used in place of *any* discovered configuration files (e.g., user-level configuration will be ignored).

## [Settings](https://docs.astral.sh/uv/concepts/configuration-files/#settings)

See the [settings reference](https://docs.astral.sh/uv/reference/settings/) for an enumeration of the available settings.

## [`.env`](https://docs.astral.sh/uv/concepts/configuration-files/#env)

`uv run` can load environment variables from dotenv files (e.g., `.env`, `.env.local`, `.env.development`), powered by the [`dotenvy`](https://github.com/allan2/dotenvy) crate.

To load a `.env` file from a dedicated location, set the `UV_ENV_FILE` environment variable, or pass the `--env-file` flag to `uv run`.

For example, to load environment variables from a `.env` file in the current working directory:

```sh
echo "MY_VAR='Hello, world!'" > .env
uv run --env-file .env -- python -c 'import os; print(os.getenv("MY_VAR"))'
```

The `--env-file` flag can be provided multiple times, with subsequent files overriding values defined in previous files. To provide multiple files via the `UV_ENV_FILE` environment variable, separate the paths with a space (e.g., `UV_ENV_FILE="/path/to/file1 /path/to/file2"`).

To disable dotenv loading (e.g., to override `UV_ENV_FILE` or the `--env-file` command-line argument), set the `UV_NO_ENV_FILE` environment variable to `1`, or pass the`--no-env-file` flag to `uv run`.

If the same variable is defined in the environment and in a `.env` file, the value from the environment will take precedence.

## [Configuring the pip interface](https://docs.astral.sh/uv/concepts/configuration-files/#configuring-the-pip-interface)

A dedicated [`[tool.uv.pip\]`](https://docs.astral.sh/uv/reference/settings/#pip) section is provided for configuring *just* the `uv pip` command line interface. Settings in this section will not apply to `uv` commands outside the `uv pip` namespace. However, many of the settings in this section have corollaries in the top-level namespace which *do* apply to the `uv pip` interface unless they are overridden by a value in the `uv.pip` section.

The `uv.pip` settings are designed to adhere closely to pip's interface and are declared separately to retain compatibility while allowing the global settings to use alternate designs (e.g., `--no-build`).

As an example, setting the `index-url` under `[tool.uv.pip]`, as in the following `pyproject.toml`, would only affect the `uv pip` subcommands (e.g., `uv pip install`, but not `uv sync`, `uv lock`, or `uv run`):

```toml
# pyproject.toml

[tool.uv.pip]
index-url = "https://test.pypi.org/simple"
```



## 设置镜像源

对于项目或全局的永久性配置，推荐使用配置文件。`uv` 支持在项目根目录下的 `pyproject.toml` 或 `uv.toml` 文件中进行配置。

### 项目级别配置

在项目的 `pyproject.toml` 文件中添加 `[tool.uv]` 或 `[tool.uv.pip]` 表。这种方式的好处是配置随项目走，方便团队协作和保持环境一致性。

**a) 使用 `[tool.uv.pip]`（兼容 pip 配置）**

这种方式的结构与 `pip` 的配置类似，简单直观。

Ini, TOML

```toml
# pyproject.toml

[tool.uv.pip]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple/"
extra-index-url = [
  "https://mirrors.aliyun.com/pypi/simple/",
  "https://pypi.mirrors.ustc.edu.cn/simple/"
]
```

**b) 使用 `[[tool.uv.index]]`（uv 推荐方式）**

这是 `uv` 提供的更灵活、功能更丰富的配置方式，允许为每个源指定名称等属性。

- `default = true` 表示将该源设为默认的主索引。
- 索引的优先级按照它们在文件中的定义顺序排列。

Ini, TOML

```toml
# pyproject.toml

[tool.uv]
# 将清华源设置为默认主索引
[[tool.uv.index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple/"
default = true
name = "tsinghua"

# 添加阿里源作为额外索引
[[tool.uv.index]]
url = "https://mirrors.aliyun.com/pypi/simple/"
name = "aliyun"
```

### 用户级别全局配置

如果您希望为当前用户的所有项目都应用相同的配置，可以在用户主目录下的配置文件中进行设置。

- **Linux/macOS**: `~/.config/uv/uv.toml`
- **Windows**: `%APPDATA%\uv\uv.toml`

在 `uv.toml` 文件中写入配置，其语法与 `pyproject.toml` 中的 `[tool.uv]` 部分类似，但不需要 `[tool.uv]` 前缀。

```toml
# 将清华源设置为默认主索引
[[index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple/"
default = true

# 添加阿里源
[[index]]
url = "https://mirrors.aliyun.com/pypi/simple/"

```

# sync

Update the project's environment

同步项目环境。

- 用法：`uv sync`
- 功能：根据锁文件（`uv.lock`）将依赖安装到虚拟环境，确保环境与锁文件一致。默认会移除未在依赖列表中的多余包，实现“精确同步”。可用 `--inexact` 保留多余包[2](https://docs.astral.sh/uv/concepts/projects/sync/),[10](https://docs.astral.sh/uv/reference/cli/)。

# add

## 基础

Add dependencies to the project

添加依赖。

- 用法：`uv add [包名]`
- 功能：将指定依赖添加到 `pyproject.toml`，并自动解析依赖、更新锁文件和虚拟环境。支持添加普通依赖、开发依赖（`--dev`）、可选依赖（`--optional`）、指定版本、指定源（如 Git、路径、本地文件）等[3](https://docs.astral.sh/uv/concepts/projects/dependencies/),[10](https://docs.astral.sh/uv/reference/cli/),[11](https://gihyo.jp/article/2024/09/monthly-python-2409),[12](https://www.datacamp.com/tutorial/python-uv)。

```shell
uv add numpy
uv add pandas==2.2.0
```

## requirements.txt

If you're migrating from a `requirements.txt` file, you can use `uv add` with the `-r` flag to add all dependencies from the file:

```sh
# Add all dependencies from `requirements.txt`.
uv add -r requirements.txt -c constraints.txt
```

## [Changing dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/#changing-dependencies)

To change an existing dependency, e.g., to use a different constraint for `httpx`:

```sh
uv add "httpx>0.1.0"
```

> In this example, we are changing the constraints for the dependency in the `pyproject.toml`. The locked version of the dependency will only change if necessary to satisfy the new constraints. To force the package version to update to the latest within the constraints, use `--upgrade-package <name>`, e.g.:
>
> ```sh
> uv add "httpx>0.1.0" --upgrade-package httpx
> ```
>
> See the [lockfile](https://docs.astral.sh/uv/concepts/projects/sync/#upgrading-locked-package-versions) documentation for more details on upgrading packages.

## [Dependency sources](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-sources)

The `tool.uv.sources` table extends the standard dependency tables with alternative dependency sources, which are used during development.

Dependency sources add support for common patterns that are not supported by the `project.dependencies` standard, like editable installations and relative paths. For example, to install `foo` from a directory relative to the project root:

**pyproject.toml**

```
[project]
name = "example"
version = "0.1.0"
dependencies = ["foo"]

[tool.uv.sources]
foo = { path = "./packages/foo" }
```

The following dependency sources are supported by uv:

- [Index](https://docs.astral.sh/uv/concepts/projects/dependencies/#index): A package resolved from a specific package index.
- [Git](https://docs.astral.sh/uv/concepts/projects/dependencies/#git): A Git repository.
- [URL](https://docs.astral.sh/uv/concepts/projects/dependencies/#url): A remote wheel or source distribution.
- [Path](https://docs.astral.sh/uv/concepts/projects/dependencies/#path): A local wheel, source distribution, or project directory.
- [Workspace](https://docs.astral.sh/uv/concepts/projects/dependencies/#workspace-member): A member of the current workspace.

> **Important**
>
> Sources are only respected by uv. If another tool is used, only the definitions in the standard project tables will be used. If another tool is being used for development, any metadata provided in the source table will need to be re-specified in the other tool's format.

### [Index](https://docs.astral.sh/uv/concepts/projects/dependencies/#index)

To add Python package from a specific index, use the `--index` option:

```sh
uv add torch torchvision torchaudio --index pytorch-cu128=https://download.pytorch.org/whl/cu128
```

uv will store the index in `[[tool.uv.index]]` and add a `[tool.uv.sources]` entry:

**pyproject.toml**

```toml
[project]
dependencies = [
  "torch",
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cu128"},
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
```

Using an `index` source *pins* a package to the given index — it will not be downloaded from other indexes.

When defining an index, an `explicit` flag can be included to indicate that the index should *only* be used for packages that explicitly specify it in `tool.uv.sources`. If `explicit` is not set, other packages may be resolved from the index, if not found elsewhere.

**pyproject.toml**

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

### [Git](https://docs.astral.sh/uv/concepts/projects/dependencies/#git)

### [URL](https://docs.astral.sh/uv/concepts/projects/dependencies/#url)

### [Path](https://docs.astral.sh/uv/concepts/projects/dependencies/#path)

### [Workspace member](https://docs.astral.sh/uv/concepts/projects/dependencies/#workspace-member)

### [Platform-specific sources](https://docs.astral.sh/uv/concepts/projects/dependencies/#platform-specific-sources)

### [Multiple sources](https://docs.astral.sh/uv/concepts/projects/dependencies/#multiple-sources)

### [Disabling sources](https://docs.astral.sh/uv/concepts/projects/dependencies/#disabling-sources)

## [Optional dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/#optional-dependencies)

## [Pytorch](https://docs.astral.sh/uv/guides/integration/pytorch/)

### 看上面的Index

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
    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchaudio = [
    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
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

```sh
uv remove numpy
```

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

## [Checking if the lockfile is up-to-date](https://docs.astral.sh/uv/concepts/projects/sync/#checking-if-the-lockfile-is-up-to-date)

When considering if the lockfile is up-to-date, uv will check if it matches the project metadata. For example, if you add a dependency to your `pyproject.toml`, the lockfile will be considered outdated. Similarly, if you change the version constraints for a dependency such that the locked version is excluded, the lockfile will be considered outdated. However, if you change the version constraints such that the existing locked version is still included, the lockfile will still be considered up-to-date.

You can check if the lockfile is up-to-date by passing the `--check` flag to `uv lock`:

```
uv lock --check
```

This is equivalent to the `--locked` flag for other commands.

## [Upgrading locked package versions](https://docs.astral.sh/uv/concepts/projects/sync/#upgrading-locked-package-versions)

With an existing `uv.lock` file, uv will prefer the previously locked versions of packages when running `uv sync` and `uv lock`. Package versions will only change if the project's dependency constraints exclude the previous, locked version.

To upgrade all packages:

```sh
uv lock --upgrade
```

To upgrade a single package to the latest version, while retaining the locked versions of all other packages:

```sh
uv lock --upgrade-package <package>
```

To upgrade a single package to a specific version:

```sh
uv lock --upgrade-package <package>==<version>
```

In all cases, upgrades are limited to the project's dependency constraints. For example, if the project defines an upper bound for a package then an upgrade will not go beyond that version.

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

# [Running scripts](https://docs.astral.sh/uv/guides/scripts/#running-scripts)

A Python script is a file intended for standalone execution, e.g., with `python <script>.py`. Using uv to execute scripts ensures that script dependencies are managed without manually managing environments.

> Note
>
> If you are not familiar with Python environments: every Python installation has an environment that packages can be installed in. Typically, creating [*virtual* environments](https://docs.python.org/3/library/venv.html) is recommended to isolate packages required by each script. uv automatically manages virtual environments for you and prefers a [declarative](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies) approach to dependencies.

## [Running a script without dependencies](https://docs.astral.sh/uv/guides/scripts/#running-a-script-without-dependencies)

If your script has no dependencies, you can execute it with `uv run`:

**example.py**

```python
print("Hello world")
```

run

```sh
uv run example.py
# Hello world
```

Similarly, if your script depends on a module in the standard library, there's nothing more to do:

**example.py**

```python
import os

print(os.path.expanduser("~"))
```

run

```sh
uv run example.py
# /Users/astral
```

Arguments may be provided to the script:

**example.py**

```python
import sys

print(" ".join(sys.argv[1:]))
```

run

```python
uv run example.py test
# test

uv run example.py hello world!
# hello world!
```

Additionally, your script can be read directly from stdin:

```sh
echo 'print("hello world!")' | uv run -
```

Or, if your shell supports [here-documents](https://en.wikipedia.org/wiki/Here_document):

```sh
uv run - <<EOF
print("hello world!")
EOF
```

Note that if you use `uv run` in a *project*, i.e., a directory with a `pyproject.toml`, it will install the current project before running the script. If your script does not depend on the project, use the `--no-project` flag to skip this:

```sh
# Note: the `--no-project` flag must be provided _before_ the script name.
uv run --no-project example.py
```

See the [projects guide](https://docs.astral.sh/uv/guides/projects/) for more details on working in projects.

## [Running a script with dependencies](https://docs.astral.sh/uv/guides/scripts/#running-a-script-with-dependencies)

When your script requires other packages, they must be installed into the environment that the script runs in. uv prefers to create these environments on-demand instead of using a long-lived virtual environment with manually managed dependencies. This requires explicit declaration of dependencies that are required for the script. Generally, it's recommended to use a [project](https://docs.astral.sh/uv/guides/projects/) or [inline metadata](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies) to declare dependencies, but uv supports requesting dependencies per invocation as well.

For example, the following script requires `rich`.

**example.py**

```python
import time
from rich.progress import track

for i in track(range(20), description="For example:"):
    time.sleep(0.05)
```

If executed without specifying a dependency, this script will fail:

```sh
uv run --no-project example.py
# Trackback (most recent call last):
# ...
```

Request the dependency using the `--with` option:

```sh
uv run --with rich example.py
# For example ...
```

Constraints can be added to the requested dependency if specific versions are needed:

```sh
uv run --with 'rich>12,<13' example.py
```

Multiple dependencies can be requested by repeating with `--with` option.

Note that if `uv run` is used in a *project*, these dependencies will be included *in addition* to the project's dependencies. To opt-out of this behavior, use the `--no-project` flag.

## [Creating a Python script](https://docs.astral.sh/uv/guides/scripts/#creating-a-python-script)

Python recently added a standard format for [inline script metadata](https://packaging.python.org/en/latest/specifications/inline-script-metadata/#inline-script-metadata). It allows for selecting Python versions and defining dependencies. Use `uv init --script` to initialize scripts with the inline metadata:

```sh
uv init --script example.py --python 3.12
```

## [Declaring script dependencies](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies)

The inline metadata format allows the dependencies for a script to be declared in the script itself.

uv supports adding and updating inline script metadata for you. Use `uv add --script` to declare the dependencies for the script:

```sh
uv add --script example.py 'requests<3' 'rich'
```

This will add a `script` section at the top of the script declaring the dependencies using TOML:

**example.py**

```python
# /// script
# dependencies = [
#   "requests<3",
#   "rich",
# ]
# ///

import requests
from rich.pretty import pprint

resp = requests.get("https://peps.python.org/api/peps.json")
data = resp.json()
pprint([(k, v["title"]) for k, v in data.items()][:10])
```

uv will automatically create an environment with the dependencies necessary to run the script, e.g.:

```sh
uv run example.py
# ...
```

> Important
>
> When using inline script metadata, even if `uv run` is [used in a *project*](https://docs.astral.sh/uv/concepts/projects/run/), the project's dependencies will be ignored. The `--no-project` flag is not required.

uv also respects Python version requirements:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

# Use some syntax added in Python 3.12
type Point = tuple[float, float]
print(Point)
```

> Note
>
> The `dependencies` field must be provided even if empty.

`uv run` will search for and use the required Python version. The Python version will download if it is not installed — see the documentation on [Python versions](https://docs.astral.sh/uv/concepts/python-versions/) for more details.

## [Using a shebang to create an executable file](https://docs.astral.sh/uv/guides/scripts/#using-a-shebang-to-create-an-executable-file)

A shebang can be added to make a script executable without using `uv run` — this makes it easy to run scripts that are on your `PATH` or in the current folder.

For example, create a file called `greet` with the following contents

**greet**

```sh
#!/usr/bin/env -S uv run --script

print("Hello, world!")
```

Ensure that your script is executable, e.g., with `chmod +x greet`, then run the script:

```sh
./greet
# Hello, world!
```

Declaration of dependencies is also supported in this context, for example:

**example**

```python
#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///

import httpx

print(httpx.get("https://example.com"))
```

## [Using alternative package indexes](https://docs.astral.sh/uv/guides/scripts/#using-alternative-package-indexes)

If you wish to use an alternative [package index](https://docs.astral.sh/uv/concepts/indexes/) to resolve dependencies, you can provide the index with the `--index` option:

```sh
uv add --index "https://example.com/simple" --script example.py 'requests<3' 'rich'
```

This will include the package data in the inline metadata:

```toml
# [[tool.uv.index]]
# url = "https://example.com/simple"
```

If you require authentication to access the package index, then please refer to the [package index](https://docs.astral.sh/uv/concepts/indexes/) documentation.

## [Locking dependencies](https://docs.astral.sh/uv/guides/scripts/#locking-dependencies)

uv supports locking dependencies for PEP 723 scripts using the `uv.lock` file format. Unlike with projects, scripts must be explicitly locked using `uv lock`:

```sh
uv lock --script example.py
```

Running `uv lock --script` will create a `.lock` file adjacent to the script (e.g., `example.py.lock`).

Once locked, subsequent operations like `uv run --script`, `uv add --script`, `uv export --script`, and `uv tree --script` will reuse the locked dependencies, updating the lockfile if necessary.

If no such lockfile is present, commands like `uv export --script` will still function as expected, but will not create a lockfile.

## [Improving reproducibility](https://docs.astral.sh/uv/guides/scripts/#improving-reproducibility)

In addition to locking dependencies, uv supports an `exclude-newer` field in the `tool.uv` section of inline script metadata to limit uv to only considering distributions released before a specific date. This is useful for improving the reproducibility of your script when run at a later point in time.

The date must be specified as an [RFC 3339](https://www.rfc-editor.org/rfc/rfc3339.html) timestamp (e.g., `2006-12-02T02:07:43Z`).

**example.py**

```python
# /// script
# dependencies = [
#   "requests",
# ]
# [tool.uv]
# exclude-newer = "2023-10-16T00:00:00Z"
# ///

import requests

print(requests.__version__)
```

## [Using different Python versions](https://docs.astral.sh/uv/guides/scripts/#using-different-python-versions)

uv allows arbitrary Python versions to be requested on each script invocation, for example:

**example.py**

```python
import sys

print(".".join(map(str, sys.version_info[:3])))
```

```python
# Use the default Python version, may differ on your machine
uv run example.py
# 3.12.6
```

```python
# Use a specific Python version
uv run --python 3.10 example.py
# 3.10.15
```

See the [Python version request](https://docs.astral.sh/uv/concepts/python-versions/#requesting-a-version) documentation for more details on requesting Python versions.

## [Using GUI scripts](https://docs.astral.sh/uv/guides/scripts/#using-gui-scripts)

On Windows `uv` will run your script ending with `.pyw` extension using `pythonw`:

**example.pyw**

```python
from tkinter import Tk, ttk

root = Tk()
root.title("uv")
frm = ttk.Frame(root, padding=10)
frm.grid()
ttk.Label(frm, text="Hello World").grid(column=0, row=0)
root.mainloop()
```

```sh
uv run example.pyw
# 会出现窗口
```

Similarly, it works with dependencies as well:

**example_pyqt.pyw**

```python
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout

app = QApplication(sys.argv)
widget = QWidget()
grid = QGridLayout()

text_label = QLabel()
text_label.setText("Hello World!")
grid.addWidget(text_label)

widget.setLayout(grid)
widget.setGeometry(100, 100, 200, 50)
widget.setWindowTitle("uv")
widget.show()
sys.exit(app.exec_())
```

```sh
uv run --with PyQt5 example_pyqt.pyw
# 会出现窗口
```

# [Using tools](https://docs.astral.sh/uv/guides/tools/#using-tools)

Many Python packages provide applications that can be used as tools. uv has specialized support for easily invoking and installing tools.

Running and installing tools published to Python package indexes, e.g., `ruff` or `black`.

- `uvx` / `uv tool run`: Run a tool in a temporary environment.
- `uv tool install`: Install a tool user-wide.
- `uv tool uninstall`: Uninstall a tool.
- `uv tool list`: List installed tools.
- `uv tool update-shell`: Update the shell to include tool executables.

## [Running tools](https://docs.astral.sh/uv/guides/tools/#running-tools)

The `uvx` command invokes a tool without installing it.

For example, to run `ruff`:

```sh
uvx ruff
```

> Note
>
> This is exactly equivalent to:
>
> ```sh
> uv tool run ruff
> ```
>
> `uvx` is provided as an alias for convenience.

Arguments can be provided after the tool name:

```sh
uvx pycowsay hello from uv
```

Tools are installed into temporary, isolated environments when using `uvx`.

> Note
>
> If you are running a tool in a [*project*](https://docs.astral.sh/uv/concepts/projects/) and the tool requires that your project is installed, e.g., when using `pytest` or `mypy`, you'll want to use [`uv run`](https://docs.astral.sh/uv/guides/projects/#running-commands) instead of `uvx`. Otherwise, the tool will be run in a virtual environment that is isolated from your project.
>
> If your project has a flat structure, e.g., instead of using a `src` directory for modules, the project itself does not need to be installed and `uvx` is fine. In this case, using `uv run` is only beneficial if you want to pin the version of the tool in the project's dependencies.

## [Commands with different package names](https://docs.astral.sh/uv/guides/tools/#commands-with-different-package-names)

When `uvx ruff` is invoked, uv installs the `ruff` package which provides the `ruff` command. However, sometimes the package and command names differ.

The `--from` option can be used to invoke a command from a specific package, e.g., `http` which is provided by `httpie`:

```sh
uvx --from httpie http
```

## [Requesting specific versions](https://docs.astral.sh/uv/guides/tools/#requesting-specific-versions)

To run a tool at a specific version, use `command@<version>`:

```sh
uvx ruff@0.3.0 check
```

To run a tool at the latest version, use `command@latest`:

```sh
uvx ruff@latest check
```

The `--from` option can also be used to specify package versions, as above:

```sh
uvx --from 'ruff==0.3.0' ruff check
```

Or, to constrain to a range of versions:

```sh
uvx --from 'ruff>0.2.0,<0.3.0' ruff check
```

Note the `@` syntax cannot be used for anything other than an exact version.

## [Requesting extras](https://docs.astral.sh/uv/guides/tools/#requesting-extras)

The `--from` option can be used to run a tool with extras:

```sh
uvx --from 'mypy[faster-cache,reports]' mypy --xml-report mypy_report
```

This can also be combined with version selection:

```sh
uvx --from 'mypy[faster-cache,reports]==1.13.0' mypy --xml-report mypy_report
```

## [Requesting different sources](https://docs.astral.sh/uv/guides/tools/#requesting-different-sources)

The `--from` option can also be used to install from alternative sources.

For example, to pull from git:

```sh
uvx --from git+https://github.com/httpie/cli httpie
```

You can also pull the latest commit from a specific named branch:

```sh
uvx --from git+https://github.com/httpie/cli@master httpie
```

Or pull a specific tag:

```sh
uvx --from git+https://github.com/httpie/cli@3.2.4 httpie
```

Or even a specific commit:

```sh
uvx --from git+https://github.com/httpie/cli@2843b87 httpie
```

## [Commands with plugins](https://docs.astral.sh/uv/guides/tools/#commands-with-plugins)

Additional dependencies can be included, e.g., to include `mkdocs-material` when running `mkdocs`:

```sh
uvx --with mkdocs-material mkdocs --help
```

## [Installing tools](https://docs.astral.sh/uv/guides/tools/#installing-tools)

If a tool is used often, it is useful to install it to a persistent environment and add it to the `PATH` instead of invoking `uvx` repeatedly.

> Tip
>
> `uvx` is a convenient alias for `uv tool run`. All of the other commands for interacting with tools require the full `uv tool` prefix.

To install `ruff`:

```sh
uv tool install ruff
```

When a tool is installed, its executables are placed in a `bin` directory in the `PATH` which allows the tool to be run without uv. If it's not on the `PATH`, a warning will be displayed and `uv tool update-shell` can be used to add it to the `PATH`.

After installing `ruff`, it should be available:

```sh
ruff --version
```

Unlike `uv pip install`, installing a tool does not make its modules available in the current environment. For example, the following command will fail:

```sh
python -c "import ruff"
```

This isolation is important for reducing interactions and conflicts between dependencies of tools, scripts, and projects.

Unlike `uvx`, `uv tool install` operates on a *package* and will install all executables provided by the tool.

For example, the following will install the `http`, `https`, and `httpie` executables:

```sh
uv tool install httpie
```

Additionally, package versions can be included without `--from`:

```sh
uv tool install 'httpie>0.1.0'
```

And, similarly, for package sources:

```sh
uv tool install git+https://github.com/httpie/cli
```

As with `uvx`, installations can include additional packages:

```sh
uv tool install mkdocs --with mkdocs-material
```

## [Upgrading tools](https://docs.astral.sh/uv/guides/tools/#upgrading-tools)

To upgrade a tool, use `uv tool upgrade`:

```sh
uv tool upgrade ruff
```

Tool upgrades will respect the version constraints provided when installing the tool. For example, `uv tool install ruff >=0.3,<0.4` followed by `uv tool upgrade ruff` will upgrade Ruff to the latest version in the range `>=0.3,<0.4`.

To instead replace the version constraints, re-install the tool with `uv tool install`:

```sh
uv tool install ruff>=0.4
```

To instead upgrade all tools:

```sh
uv tool upgrade --all
```

## [Requesting Python versions](https://docs.astral.sh/uv/guides/tools/#requesting-python-versions)

By default, uv will use your default Python interpreter (the first it finds) when running, installing, or upgrading tools. You can specify the Python interpreter to use with the `--python` option.

For example, to request a specific Python version when running a tool:

```
uvx --python 3.10 ruff
```

Or, when installing a tool:

```
uv tool install --python 3.10 ruff
```

Or, when upgrading a tool:

```
uv tool upgrade --python 3.10 ruff
```

For more details on requesting Python versions, see the [Python version](https://docs.astral.sh/uv/concepts/python-versions/#requesting-a-version) concept page.

## [Legacy Windows Scripts](https://docs.astral.sh/uv/guides/tools/#legacy-windows-scripts)

Tools also support running [legacy setuptools scripts](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#scripts). These scripts are available via `$(uv tool dir)\<tool-name>\Scripts` when installed.

Currently only legacy scripts with the `.ps1`, `.cmd`, and `.bat` extensions are supported.

For example, below is an example running a Command Prompt script.

```
uv tool run --from nuitka==2.6.7 nuitka.cmd --version
```

In addition, you don't need to specify the extension. `uvx` will automatically look for files ending in `.ps1`, `.cmd`, and `.bat` in that order of execution on your behalf.

```
uv tool run --from nuitka==2.6.7 nuitka --version
```

# [Tools](https://docs.astral.sh/uv/concepts/tools/#tools)

Tools are Python packages that provide command-line interfaces.

> Note
>
> See the [tools guide](https://docs.astral.sh/uv/guides/tools/) for an introduction to working with the tools interface — this document discusses details of tool management.

## [The `uv tool` interface](https://docs.astral.sh/uv/concepts/tools/#the-uv-tool-interface)

uv includes a dedicated interface for interacting with tools. Tools can be invoked without installation using `uv tool run`, in which case their dependencies are installed in a temporary virtual environment isolated from the current project.

Because it is very common to run tools without installing them, a `uvx` alias is provided for `uv tool run` — the two commands are exactly equivalent. For brevity, the documentation will mostly refer to `uvx` instead of `uv tool run`.

Tools can also be installed with `uv tool install`, in which case their executables are [available on the `PATH`](https://docs.astral.sh/uv/concepts/tools/#the-path) — an isolated virtual environment is still used, but it is not removed when the command completes.

## [Execution vs installation](https://docs.astral.sh/uv/concepts/tools/#execution-vs-installation)

In most cases, executing a tool with `uvx` is more appropriate than installing the tool. Installing the tool is useful if you need the tool to be available to other programs on your system, e.g., if some script you do not control requires the tool, or if you are in a Docker image and want to make the tool available to users.

## [Tool environments](https://docs.astral.sh/uv/concepts/tools/#tool-environments)

When running a tool with `uvx`, a virtual environment is stored in the uv cache directory and is treated as disposable, i.e., if you run `uv cache clean` the environment will be deleted. The environment is only cached to reduce the overhead of repeated invocations. If the environment is removed, a new one will be created automatically.

When installing a tool with `uv tool install`, a virtual environment is created in the uv tools directory. The environment will not be removed unless the tool is uninstalled. If the environment is manually deleted, the tool will fail to run.

## [Tool versions](https://docs.astral.sh/uv/concepts/tools/#tool-versions)

Unless a specific version is requested, `uv tool install` will install the latest available of the requested tool. `uvx` will use the latest available version of the requested tool *on the first invocation*. After that, `uvx` will use the cached version of the tool unless a different version is requested, the cache is pruned, or the cache is refreshed.

For example, to run a specific version of Ruff:

```sh
uvx ruff@0.6.0 --version
# ruff 0.6.0
```

A subsequent invocation of `uvx` will use the latest, not the cached, version.

```sh
uvx ruff --version
# ruff 0.6.2
```

But, if a new version of Ruff was released, it would not be used unless the cache was refreshed.

To request the latest version of Ruff and refresh the cache, use the `@latest` suffix:

```sh
uvx ruff@latest --version
# ruff 0.6.2
```

Once a tool is installed with `uv tool install`, `uvx` will use the installed version by default.

For example, after installing an older version of Ruff:

```sh
uv tool install ruff==0.5.0
```

The version of `ruff` and `uvx ruff` is the same:

```sh
ruff --version
# ruff 0.5.0
uvx ruff --version
# ruff 0.5.0
```

However, you can ignore the installed version by requesting the latest version explicitly, e.g.:

```sh
uvx ruff@latest --version
# 0.6.2
```

Or, by using the `--isolated` flag, which will avoid refreshing the cache but ignore the installed version:

```sh
uvx --isolated ruff --version
# 0.6.2
```

`uv tool install` will also respect the `{package}@{version}` and `{package}@latest` specifiers, as in:

```sh
uv tool install ruff@latest
uv tool install ruff@0.6.0
```

## [Tools directory](https://docs.astral.sh/uv/concepts/tools/#tools-directory)

By default, the uv tools directory is named `tools` and is in the uv application state directory, e.g., `~/.local/share/uv/tools`. The location may be customized with the `UV_TOOL_DIR` environment variable.

To display the path to the tool installation directory:

```sh
uv tool dir
```

Tool environments are placed in a directory with the same name as the tool package, e.g., `.../tools/<name>`.

> Important
>
> Tool environments are *not* intended to be mutated directly. It is strongly recommended never to mutate a tool environment manually, e.g., with a `pip` operation.

## [Upgrading tools](https://docs.astral.sh/uv/concepts/tools/#upgrading-tools)

Tool environments may be upgraded via `uv tool upgrade`, or re-created entirely via subsequent `uv tool install` operations.

To upgrade all packages in a tool environment

```sh
uv tool upgrade black
```

To upgrade a single package in a tool environment:

```sh
uv tool upgrade black --upgrade-package click
```

Tool upgrades will respect the version constraints provided when installing the tool. For example, `uv tool install black >=23,<24` followed by `uv tool upgrade black` will upgrade Black to the latest version in the range `>=23,<24`.

To instead replace the version constraints, reinstall the tool with `uv tool install`:

```sh
uv tool install black>=24
```

Similarly, tool upgrades will retain the settings provided when installing the tool. For example, `uv tool install black --prerelease allow` followed by `uv tool upgrade black` will retain the `--prerelease allow` setting.

> Note
>
> Tool upgrades will reinstall the tool executables, even if they have not changed.

To reinstall packages during upgrade, use the `--reinstall` and `--reinstall-package` options.

To reinstall all packages in a tool environment

```sh
uv tool upgrade black --reinstall
```

To reinstall a single package in a tool environment:

```sh
uv tool upgrade black --reinstall-package click
```

## [Including additional dependencies](https://docs.astral.sh/uv/concepts/tools/#including-additional-dependencies)

Additional packages can be included during tool execution:

```
uvx --with <extra-package> <tool>
```

And, during tool installation:

```
uv tool install --with <extra-package> <tool-package>
```

The `--with` option can be provided multiple times to include additional packages.

The `--with` option supports package specifications, so a specific version can be requested:

```
uvx --with <extra-package>==<version> <tool-package>
```

If the requested version conflicts with the requirements of the tool package, package resolution will fail and the command will error.

# [The pip interface](https://docs.astral.sh/uv/pip/#the-pip-interface)

## [Managing packages](https://docs.astral.sh/uv/pip/packages/#managing-packages)

### [Installing a package](https://docs.astral.sh/uv/pip/packages/#installing-a-package)

To install a package into the virtual environment, e.g., Flask:

```sh
uv pip install flask
```

To install a package with optional dependencies enabled, e.g., Flask with the "dotenv" extra:

```sh
uv pip install "flask[dotenv]"
```

To install multiple packages, e.g., Flask and Ruff:

```sh
uv pip install flask ruff
```

To install a package with a constraint, e.g., Ruff v0.2.0 or newer:

```sh
uv pip install 'ruff>=0.2.0'
```

To install a package at a specific version, e.g., Ruff v0.3.0:

```sh
uv pip install 'ruff==0.3.0'
```

To install a package from the disk:

```sh
uv pip install "ruff @ ./projects/ruff"
```

To install a package from GitHub:

```sh
uv pip install "git+https://github.com/astral-sh/ruff"
```

To install a package from GitHub at a specific reference:

```sh
# Install a tag
uv pip install "git+https://github.com/astral-sh/ruff@v0.2.0"

# Install a commit
uv pip install "git+https://github.com/astral-sh/ruff@1fadefa67b26508cc59cf38e6130bde2243c929d"

# Install a branch
uv pip install "git+https://github.com/astral-sh/ruff@main"
```

See the [Git authentication](https://docs.astral.sh/uv/concepts/authentication/#git-authentication) documentation for installation from a private repository.

### [Editable packages](https://docs.astral.sh/uv/pip/packages/#editable-packages)

Editable packages do not need to be reinstalled for changes to their source code to be active.

To install the current project as an editable package

```sh
uv pip install -e .
```

To install a project in another directory as an editable package:

```sh
uv pip install -e "ruff @ ./project/ruff"
```

### [Installing packages from files](https://docs.astral.sh/uv/pip/packages/#installing-packages-from-files)

Multiple packages can be installed at once from standard file formats.

Install from a `requirements.txt` file:

```sh
uv pip install -r requirements.txt
```

See the [`uv pip compile`](https://docs.astral.sh/uv/pip/compile/) documentation for more information on `requirements.txt` files.

Install from a `pyproject.toml` file:

```sh
uv pip install -r pyproject.toml
```

Install from a `pyproject.toml` file with optional dependencies enabled, e.g., the "foo" extra:

```sh
uv pip install -r pyproject.toml --extra foo
```

Install from a `pyproject.toml` file with all optional dependencies enabled:

```sh
uv pip install -r pyproject.toml --all-extras
```

To install dependency groups in the current project directory's `pyproject.toml`, for example the group `foo`:

```sh
uv pip install --group foo
```

To specify the project directory where groups should be sourced from:

```sh
uv pip install --project some/path/ --group foo --group bar
```

Alternatively, you can specify a path to a `pyproject.toml` for each group:

```sh
uv pip install --group some/path/pyproject.toml:foo --group other/pyproject.toml:bar
```

Note

As in pip, `--group` flags do not apply to other sources specified with flags like `-r` or -e`. For instance,`uv pip install -r some/path/pyproject.toml --group foo`sources`foo`from`./pyproject.toml`and **not**`some/path/pyproject.toml`.

### [Uninstalling a package](https://docs.astral.sh/uv/pip/packages/#uninstalling-a-package)

To uninstall a package, e.g., Flask:

```sh
uv pip uninstall flask
```

To uninstall multiple packages, e.g., Flask and Ruff:

```sh
uv pip uninstall flask ruff
```

## [Inspecting environments](https://docs.astral.sh/uv/pip/inspection/#inspecting-environments)

### [Listing installed packages](https://docs.astral.sh/uv/pip/inspection/#listing-installed-packages)

To list all the packages in the environment:

```sh
uv pip list
```

To list the packages in a JSON format:

```sh
uv pip list --format json
```

To list all the packages in the environment in a `requirements.txt` format:

```sh
uv pip freeze
```

### [Inspecting a package](https://docs.astral.sh/uv/pip/inspection/#inspecting-a-package)

To show information about an installed package, e.g., `numpy`:

```sh
uv pip show numpy
```

Multiple packages can be inspected at once.

### [Verifying an environment](https://docs.astral.sh/uv/pip/inspection/#verifying-an-environment)

It is possible to install packages with conflicting requirements into an environment if installed in multiple steps.

To check for conflicts or missing dependencies in the environment:

```sh
uv pip check
```

# venv

Create a virtual environment

- 用法：`uv venv [目录] [--seed]`

- 例子：
  - `uv venv .venv` 在当前目录创建虚拟环境
  
  uv supports creating virtual environments, e.g., to create a virtual environment at `.venv`:
  
  ```
  uv venv
  ```
  
  A specific name or path can be specified, e.g., to create a virtual environment at `my-name`:
  
  ```
  uv venv my-name
  ```
  
  A Python version can be requested, e.g., to create a virtual environment with Python 3.11:
  
  ```
  uv venv --python 3.11
  ```
  
  Note this requires the requested Python version to be available on the system. However, if unavailable, uv will download Python for you. See the [Python version](https://docs.astral.sh/uv/concepts/python-versions/) documentation for more details.

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

