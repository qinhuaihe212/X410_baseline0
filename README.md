# X410 Baseline 0：单用户 PUSCH 有线回环

这个目录实现一次最小的单用户、单层、单通道 5G NR PUSCH 传输块实验。Sionna 2.2 的传统发射机完成 TB CRC、LDPC、速率匹配、扰码、QPSK、DMRS 映射和 OFDM 调制；传统接收机完成 OFDM 解调、DMRS-LS 信道估计、LMMSE 检测及 TB 译码，并输出 CRC。X410 通过 UHD Python API **实际发射一个有限长度的 burst，同时采集原始复数 IQ**。软件随后寻找前导、估计小频偏与复增益，截取 PUSCH 时域采样并交给传统接收机。Sionna 接收机的接口见[官方文档](https://nvlabs.github.io/sionna/v2.2.0/phy/api/nr/sionna.phy.nr.PUSCHReceiver.html)。

`run.py` 的 `selftest` 模式不连接硬件，只生成已知延时、复增益、频偏和噪声的采样，检验从前导定位到 TB CRC 的软件流程。`hardware` 模式只连接**一台 X410**，TX/RX 共用同一个 `MultiUSRP` 对象和设备时间。代码没有实现随机接入、调度、HARQ、真实 UE/gNB 协议栈或盲搜索任意未知用户；前导是为实验加在 PUSCH slot 前面的非 NR 训练序列。

## 依赖

- Python 3.11+，`pip install -r requirements.txt`；建议在 Linux 上使用与 UHD Python 绑定兼容的环境。`sionna-no-rt==2.2.0` 会安装 PyTorch 等 Python 依赖。
- `hardware` 还需要可导入的 `uhd` Python 绑定、X410 兼容的主机 UHD 及设备镜像。`uhd` 通常随 UHD 安装，不包含在本目录的 pip requirements 中。先用 `uhd_find_devices`、`uhd_usrp_probe --args="addr=<X410地址>"` 检查连接和 ZBX 射频板卡。
- X410 的主时钟和可用采样率受 FPGA 镜像约束。脚本从 Sionna 资源栅格计算采样率，设置后回读 TX/RX 实际值；如不一致直接报错，不隐式重采样。参考 [X4x0 设备手册](https://files.ettus.com/manual/page_usrp_x4xx.html)。

## 射频连接

将所选通道的 **TX/RX0 → 固定衰减器与线缆 → RX1**。默认通道为 0。ZBX 的端口名可通过 UHD 枚举；脚本也会核查。ZBX 官方资料给出的最大输出功率可达 23 dBm，接收端正常工作最大输入为 0 dBm；本示例要求输入实测链路衰减至少 40 dB，默认 TX 增益 0 dB、数字峰值幅度 0.1。仍需按实际设备、频率与线缆核算接收功率，避免过载。[ZBX 官方手册](https://files.ettus.com/manual/page_zbx.html)

不要把 TX/RX0 与 RX1 直接用一根无衰减线连接，也不要把两个脚本分别打开同一台 X410。本目录的程序在一个进程中完成收发。

## 运行

在仓库根目录：

```bash
python X410_Baseline0/run.py --mode selftest --output X410_Baseline0/selftest_001
```

自检应输出 `PUSCH CRC=True, bit errors=0/...`。硬件运行示例：

```bash
python X410_Baseline0/run.py --mode hardware \
  --device 'addr=192.168.10.2' \
  --channel 0 --tx-antenna TX/RX0 --rx-antenna RX1 \
  --freq 2.4e9 --tx-gain 0 --rx-gain 0 \
  --attenuation-db 40 \
  --output X410_Baseline0/capture_001
```

把设备 IP、端口、频率和**实测**衰减量改成现场值。输出目录必须不存在。`--prbs` 默认 4、`--mcs` 默认 4（MCS 表 1 的 QPSK）；改变资源配置后程序会重新计算采样率，并核对 X410 的设置结果。

## 输出与判读

| 文件 | 含义 |
|---|---|
| `tx_burst.npy` | 包括静默区、双段训练前导、PUSCH slot 和尾部静默区的实际发送 IQ |
| `tx_tb_bits.npy` | 发射 TB 信息比特真值，仅用于评估 |
| `rx_raw_iq.npy` | UHD 接收的完整原始 IQ，未对齐或补偿 |
| `rx_aligned_pusch.npy` | 前导同步、频偏与复增益补偿后的 PUSCH 时域 IQ |
| `rx_tb_bits.npy` | 传统接收机译码得到的信息比特 |
| `report.json` | 请求与回读硬件参数、同步位置、频偏、噪声估计、CRC、BER |

先看 `report.json` 中的 `sync.correlation`、`rx_peak_magnitude` 和 `result.tb_crc_ok`。CRC 成功并且误比特数为 0，才表示本次**已知 TB 的有线回环**闭环成功；程序显示 `hardware_tested: true` 只代表执行了 UHD 硬件路径，并不等于 CRC 已通过。接收采样若溢出、超时或短读，程序会报错，不会把旧缓冲区数据当作有效 IQ。

## 当前边界

- 同一 X410 的定时发射和接收并不消除模拟射频链路的群时延、IQ 失衡或滤波影响。前导用于粗同步与单一复增益校正；复杂多径、较大采样时钟偏差和严重频偏需要额外算法。
- 噪声方差取前导前的静默区，再按估计增益归一化，是最小实验的近似值；可靠 BER/BLER 曲线仍需独立校准并运行大量 TB。
- 程序不会把采集结果冒充已经完成的 X410 实测。本仓库提交的是可运行的硬件路径和自检；只有现场运行并检查 `report.json` 才能确认设备链路。
