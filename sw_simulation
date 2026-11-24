
import numpy as np
import torch
import sionna.phy

from sionna.phy.nr import (
    PUSCHConfig,
    PUSCHTransmitter,
    PUSCHReceiver
)
from sionna.phy.channel import AWGN
from sionna.phy.utils import compute_ber

# 1. 固定随机种子
sionna.phy.config.seed = 42

# 2. 配置单用户、单层、单发射天线 PUSCH
config = PUSCHConfig()

config.num_layers = 1
config.num_antenna_ports = 1

# 4 PRB，48 个子载波
config.carrier.n_size_grid = 4
config.carrier.subcarrier_spacing = 15

# QPSK，低码率 MCS
config.tb.mcs_table = 1
config.tb.mcs_index = 0

# 3. 创建发射机和接收机
tx = PUSCHTransmitter(config)
rx = PUSCHReceiver(tx)

# 4. 创建 AWGN 信道
channel = AWGN()

# 5. 发射一个 TB
batch_size = 1
no = 0.01

x, b = tx(batch_size)

# 6. 信号经过 AWGN 信道
y = channel(x, no)

# 7. 接收机恢复 TB
b_hat = rx(y, no)

# 8. 计算误码率
ber = compute_ber(b, b_hat)

print("TB size:", config.tb_size)
print("原始比特 shape:", b.shape)
print("发射资源网格 shape:", x.shape)
print("接收比特 shape:", b_hat.shape)
print("BER:", ber.item())

# 9. 可视化资源分配
tx.resource_grid.show()
