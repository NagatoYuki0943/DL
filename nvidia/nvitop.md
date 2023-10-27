# nvitop nvidia 显卡监控工具

https://github.com/XuehaiPan/nvitop

`pip install nvitop`

# Run as a resource monitor:

```sh
# Monitor mode (when the display mode is omitted, `NVITOP_MONITOR_MODE` will be used)
$ nvitop  # or use `python3 -m nvitop`

# Automatically configure the display mode according to the terminal size
$ nvitop -m auto     # shortcut: `a` key

# Arbitrarily display as `full` mode
$ nvitop -m full     # shortcut: `f` key

# Arbitrarily display as `compact` mode
$ nvitop -m compact  # shortcut: `c` key

# Specify query devices (by integer indices)
$ nvitop -o 0 1  # only show <GPU 0> and <GPU 1>

# Only show devices in `CUDA_VISIBLE_DEVICES` (by integer indices or UUID strings)
$ nvitop -ov

# Only show GPU processes with the compute context (type: 'C' or 'C+G')
$ nvitop -c

# Use ASCII characters only
$ nvitop -U  # useful for terminals without Unicode support

# For light terminals
$ nvitop --light

# For spectrum-like bar charts (requires the terminal supports 256-color)
$ nvitop --colorful
```