# sw_simulation.py 零基础解读

**可以用这段脚本做传统 PUSCH 的频域基带收发仿真。** 它用 Sionna 实现单用户、单层、单天线端口的编码、调制、信道估计和译码，在电脑里把一个传输块从发射端送到接收端。实际射频收发需要时域波形、设备采集和同步，这些环节位于同目录的 `hw_simulation.py` 路径。本文的接口行为以 `requirements.txt` 指定的 Sionna 2.2.0 为准。 [发射机 API](https://nvlabs.github.io/sionna/v2.2.0/phy/api/nr/sionna.phy.nr.PUSCHTransmitter.html) · [接收机 API](https://nvlabs.github.io/sionna/v2.2.0/phy/api/nr/sionna.phy.nr.PUSCHReceiver.html)

## 先认识实验里的数据

**PUSCH** 是 5G NR 的物理上行共享信道。这里模拟终端向基站发送的一小段上行数据。**TB**（传输块）是物理层一次处理的一组信息比特，也就是一串 0 和 1。脚本随机生成一个 TB，最后把译码得到的比特与原始比特逐位对照。

**资源栅格**是一张时频表。一个轴是 OFDM 符号，可以理解成一段一段的发送时间。另一个轴是子载波，可以理解成并排工作的频率位置。一个时间位置和一个频率位置组成一个资源元素（RE）。频域信号的每个 RE 由一个复数表示，数据格装入调制符号，另有一些格子保留给接收机事先知道的参考信号 DMRS 或留空。

```text
随机原始比特 b
   │ 编码、QPSK、映射到栅格，同时放入 DMRS
   ▼
发送栅格 x ── 加入随机噪声 ──► 接收栅格 y
                                      │ 估计信道、检测、译码
                                      ▼
原始比特 b ─────── 逐位比较 ────── 译码比特 b_hat
                       │
                       ▼
                      BER
```

## 按执行顺序读代码

| 行号 | 代码 | 这一步在做什么 |
| --- | --- | --- |
| 2–15 | 导入模块，设置 `seed = 42` | Sionna 提供收发机、噪声和 BER 计算。固定种子有助于在相同环境复现随机比特和噪声。脚本本身没有直接调用 `np` 或 `torch`。 |
| 17–25 | `PUSCHConfig()`、`num_layers = 1`、`num_antenna_ports = 1` | 配置一条并行数据流和一个发射天线端口。一个配置对象代表一个发射端。载波栅格宽 4 个 PRB，每个 PRB 有 12 个连续子载波，所以栅格一共 48 个子载波，相邻子载波相隔 15 kHz。 |
| 27–29 | `mcs_table = 1`、`mcs_index = 0` | MCS 是调制与编码方案。这里查表选到低目标码率的 QPSK。QPSK 用四种复数符号表示编码后的四种两比特组合，低码率为纠错留出较多冗余。 |
| 31–36 | `tx`、`rx`、`channel` | 建好发射机、接收机和加噪模块。接收机通过 `tx` 获取资源分配、导频模式和编码等配置。 |
| 38–42 | `x, b = tx(1)` | 生成一次传输。默认随机生成原始信息比特 `b`，再生成频域发送栅格 `x`。 |
| 44–45 | `y = channel(x, no)` | AWGN（加性白高斯噪声）直接给每个复数格加噪。复噪声方差是 `0.01`，实部和虚部的方差各为 `0.005`。这个数用线性单位表示。 |
| 47–51 | `b_hat = rx(y, no)`、`compute_ber(b, b_hat)` | 接收机用带噪栅格与实验提供的噪声方差恢复比特。拿到结果后，仿真程序用发送真值统计错误比例。 |
| 53–60 | 打印长度、形状、BER；调用 `tx.resource_grid.show()` | 打印本次传输的数据尺寸和结果，再创建资源分配图。图上展示数据、导频等格子的用途。 |

上述频域默认值、输入输出形状和噪声方差定义可在对应版本的接口和源码中核对。 [发射机 API](https://nvlabs.github.io/sionna/v2.2.0/phy/api/nr/sionna.phy.nr.PUSCHTransmitter.html) · [接收机 API](https://nvlabs.github.io/sionna/v2.2.0/phy/api/nr/sionna.phy.nr.PUSCHReceiver.html) · [AWGN 源码](https://raw.githubusercontent.com/NVlabs/sionna/v2.2.0/src/sionna/phy/channel/awgn.py)

### 发射机如何把 0 和 1 放进栅格

`tx(1)` 里，Sionna 先给 TB 加 CRC 校验位，供接收端判断译码结果是否通过校验。LDPC 编码加入纠错冗余，速率匹配把编码比特调整到能装进本次分配的资源，扰码用收发双方可重建的序列打散比特。随后，QPSK 把每两个编码比特变成一个复数符号。Sionna 将符号映射到数据格，并在预定位置放入 DMRS。 [发射机 API](https://nvlabs.github.io/sionna/v2.2.0/phy/api/nr/sionna.phy.nr.PUSCHTransmitter.html) · [TBEncoder API](https://nvlabs.github.io/sionna/v2.2.0/phy/api/nr/sionna.phy.nr.TBEncoder.html)

### 接收机如何从栅格取回 0 和 1

`rx(y, no)` 先比较收到的 DMRS 和事先知道的 DMRS，用 LS（最小二乘）估计信道，再把导频位置上的估计插值到数据位置。默认的 LMMSE（线性最小均方误差）检测利用估计的信道与噪声信息恢复数据符号，并给出每个比特倾向于 0 或 1 的软信息。TB 译码器对这些软信息做解扰和 LDPC 译码，给出最终的 `b_hat`。这些都是传统通信算法。PyTorch 在这里承担数值计算。 [接收机源码](https://raw.githubusercontent.com/NVlabs/sionna/v2.2.0/src/sionna/phy/nr/pusch_receiver.py)

虽然 AWGN 只模拟加噪，信号本身相当于通过增益为 1 的信道，但接收机仍会按默认流程从带噪导频估计信道。`no` 是仿真已知的噪声强度，`b` 只在末尾计算 BER 时使用。

## 读懂输出

| 输出 | 怎么读 |
| --- | --- |
| `TB size` | 一个 TB 的原始信息比特数。这个值由资源分配、导频开销和 MCS 等配置共同决定。 |
| `b.shape` 和 `b_hat.shape` | 形状为 `[1, 1, TB size]`，分别对应一组实验、一个发射端和该 TB 的比特。 |
| `x.shape` | 按 2.2.0 默认 OFDM 符号分配推得 `[1, 1, 1, 14, 48]`，对应批大小、发射端数、发射天线端口数、OFDM 符号数、子载波数。 |
| `BER` | 错误比特数除以比较的比特总数。比如 100 个比特错 2 个，BER 就是 0.02。若本次打印 0，表示这个 TB 逐位比对无误。 |
| 资源分配图 | 看横纵轴上的 OFDM 符号与子载波索引，再按图例找数据格和导频格。脚本调用绘图方法，窗口显示取决于运行环境的绘图后端。 |

上表的形状根据接口和默认配置推导。 [PUSCH 配置源码](https://raw.githubusercontent.com/NVlabs/sionna/v2.2.0/src/sionna/phy/nr/pusch_config.py) · [发射机 API](https://nvlabs.github.io/sionna/v2.2.0/phy/api/nr/sionna.phy.nr.PUSCHTransmitter.html)

脚本只发送一个 TB、使用一个噪声设置。要测某一噪声强度下的典型 BER，需要累计许多独立 TB 的错误数和比特数。脚本使用了 `PUSCHReceiver(tx)` 的默认返回值，只打印 BER。若想同时查看 CRC 是否通过，可将构造改成 `PUSCHReceiver(tx, return_tb_crc_status=True)`，再用 `b_hat, crc_ok = rx(y, no)` 接收两个结果。BER 需要发送真值，CRC 可由接收端根据译码结果自行检查。 [接收机 API](https://nvlabs.github.io/sionna/v2.2.0/phy/api/nr/sionna.phy.nr.PUSCHReceiver.html)

## 这段脚本覆盖到哪里

- **频域链路仿真**从随机信息比特出发，经过传统 PUSCH 编码、调制和资源映射，在频域加噪后完成传统接收。实验假定收发参数、栅格边界和噪声方差已知，信道模型只有 AWGN。
- **时域与硬件链路**要通过 IFFT 把频域格子变成时域 IQ 采样并添加循环前缀，接收侧在完成定位和同步后再变回栅格。当前脚本使用 `output_domain="freq"` 和 `input_domain="freq"` 的默认值，因此没有执行这两次转换，也没有打开射频设备。同目录的 [hw_simulation.py](hw_simulation.py) 明确选择时域接口，有软件回环和 X410 有线回环两条路径。 [发射机 API](https://nvlabs.github.io/sionna/v2.2.0/phy/api/nr/sionna.phy.nr.PUSCHTransmitter.html) · [接收机 API](https://nvlabs.github.io/sionna/v2.2.0/phy/api/nr/sionna.phy.nr.PUSCHReceiver.html)
- **完整的 5G 上行通信**还需要终端和基站的协议流程，例如接入和调度。本实验的资源分配由 Python 直接填写。

## 运行方式

在工作区根目录，使用已激活的、兼容仓库依赖的独立 Python 环境运行：

```bash
python -m pip install -r X410_baseline0/requirements.txt
python X410_baseline0/sw_simulation.py
```

当前工作区的默认 Python 环境装有 Sionna 0.19.2。尝试运行时，在旧依赖 Dr.Jit 初始化阶段因目录只读而中止。本次未取得 Sionna 2.2.0 下的实测 BER。
