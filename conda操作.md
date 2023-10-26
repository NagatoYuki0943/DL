# 下载

[miniconda](https://docs.conda.io/en/latest/miniconda.html)

[miniconda](https://repo.anaconda.com/miniconda/)

[anaconda](https://www.anaconda.com/products/distribution)

# 环境变量(anaconda和miniconda)

```shell
{conda}
{conda}\Scripts
{conda}\Library\bin
```

# 查看帮助

```shell
> conda --help
usage: conda-script.py [-h] [-V] command ...

conda is a tool for managing and deploying applications, environments and packages.

Options:

positional arguments:
  command
    clean        Remove unused packages and caches.
    compare      Compare packages between conda environments.
    config       Modify configuration values in .condarc. This is modeled after the git config command. Writes to the user
                 .condarc file (C:\Users\Frostbite\.condarc) by default.
    create       Create a new conda environment from a list of specified packages.
    help         Displays a list of available conda commands and their help strings.
    info         Display information about current conda install.
    init         Initialize conda for shell interaction. [Experimental]
    install      Installs a list of packages into a specified conda environment.
    list         List linked packages in a conda environment.
    package      Low-level conda package utility. (EXPERIMENTAL)
    remove       Remove a list of packages from a specified conda environment.
    uninstall    Alias for conda remove.
    run          Run an executable in a conda environment.
    search       Search for packages and display associated information. The input is a MatchSpec, a query language for
                 conda packages. See examples below.
    update       Updates conda packages to the latest compatible version.
    upgrade      Alias for conda update.

optional arguments:
  -h, --help     Show this help message and exit.
  -V, --version  Show the conda version number and exit.

conda commands available from other packages:
  content-trust
  env
```

# 查看config

```shell
conda config --show
```

# 配置

## 配置文件

> `~/.condarc` 保存 代理,是否进入base环境等

```shell
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
auto_activate_base: false
ssl_verify: false
```

> `~/.conda/environments.txt`  存放虚拟环境

## 关闭自动进入base环境

```shell
conda config --set auto_activate_base false
```

## 关闭ssl验证

```shell
conda config --set ssl_verify no
```

## 代理

> https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

### 设置代理

> 先设置free在设置main，这样main在上面

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
```

> 设置启动设置好的国内镜像源

```shell
conda config --set show_channel_urls yes
```

> 查看是否添加上了源

```shell
conda config --show	# channel默认是defaults,修改之后就不一样了
```

### 删除代理

```
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --remove channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
```



# 环境管理

## 初始化

```shell
conda init
conda init powershell # powershell下面有详细介绍
conda init zsh
conda init fish
```



## 进入环境

```shell
conda activate 环境名
```



## 退出环境

```shell
conda deactivate
```



## 创建虚拟环境 **conda create -n 环境名 python=3.9**

```shell
conda create -n/--name=env_name python=3.10
```

> 建完成后目录文件夹会生成在Anaconda3/Miniconda3的安装目录envs下边
>
> linux会创建在 home/xxx/.conda/envs/ 目录下



## 删除虚拟环境 **conda remove -n 环境名 --all**

```shell
conda remove -n/--name=env_name --all
```

## 列出所有可用环境 **conda env list**

```shell
conda env list
conda info --envs
```



## 从旧的环境克隆出一个新环境

```shell
conda create -n 环境名 --clone oldname
```

## 导出环境配置(进入虚拟环境中使用)

```shell
conda env export > environment.yml
```

## 用导出的配置生成一个新环境

```shell
conda env create -f environment.yml
```

## 删除缓存

```shell
conda clean -a
conda clean -p	# 让conda决定哪些包是安全的删除
```



> 或者删除 Anaconda/Miniconda 目录下的 pkgs 文件夹



# 包管理

## 搜索某包

```shell
conda search packagename
```

## 安装某包

```shell
conda install packagename
conda install packagename=1.13			# 指定版本
conda install -n envname packagename 	# 为某环境安装某包
conda install --use-local pkg.tar.bz2 	# 安装离线包
```

## 更新某包

```shell
conda update --all						# 更新环境中全部的包
conda update packagename
conda update -n envname packagename 	# 更新某特定环境某包
```

## 删除某包

```shell
conda remove packagename
conda remove -n envname packagename 	# 删除某环境某包
```

## 列举当前环境下的所有包

```shell
conda list
conda list -n packagename  # 列举某个特定名称包
```

## conda更新

```shell
conda update -n base -c defaults conda
conda upgrade	# Alias for conda update.
conda update anaconda
```

# 给powershell添加conda

```shell
# powershell使用
set-ExecutionPolicy RemoteSigned 选择 Y
conda init powershell
conda config --set auto_activate_base false
```

> 配置文件位置
>
> `%userprofile%\Documents\WindowsPowerShell\profile.ps1`
>
> `%userprofile%\Documents\PowerShell\profile.ps1`

## 给powershell添加conda路径问题

### 问题1(已解决)

> pycharm默认的终端是powershell，进入的就是你的环境，即使没有提示，不过添加conda路径之后默认进入base，要手动进入你的环境中
>
> 解决办法：设置conda不自动激活base环境	`conda config --set auto_activate_base false`

### 问题2(已解决,和pycharm改法相同了):

> vscode右键运行默认是powershell，在没有添加conda路径之前，不能进入conda环境，右键命令是 `& conda run -n ai --no-capture-output --live-stream python e:/ai/test.py`, 可以正常运行，而在进入conda环境之后，这个命令会报错，因为 `run -n ai` 不能在 ai 环境中运行，要在base中运行，所以要回到base才能用(cmd也要这样)，所以不建议添加conda到powershell
>
> (old)解决办法： 在vscode设置中搜索python.terminal.activateEnvironment，取消勾选就不会自动进入虚拟环境了(PS:上面的命令在linux上即使进入虚拟环境也不报错)
>
> (new)最新版vscode运行python文件和pycharm相同了,`D:\Anaconda3\envs\ai\python.exe E:/ai/test.py`,没问题了