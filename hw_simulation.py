"""Single-user NR PUSCH B0 over one X410 ZBX channel, or a software self-test.

The PUSCH transmitter/receiver are Sionna's conventional blocks. A short
non-NR preamble is added only to locate and equalize the captured slot.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from sionna.phy.nr import PUSCHConfig, PUSCHReceiver, PUSCHTransmitter


def make_link(prbs: int, mcs: int):
    """One UE, one layer, QPSK, DMRS-LS estimation and LMMSE detection."""
    cfg = PUSCHConfig()
    cfg.carrier.n_size_grid = prbs
    cfg.num_antenna_ports = 1
    cfg.num_layers = 1
    cfg.tb.mcs_table = 1
    cfg.tb.mcs_index = mcs
    cfg.dmrs.dmrs_port_set = [0]
    cfg.dmrs.additional_position = 0
    cfg.check_config()
    if int(cfg.tb.num_bits_per_symbol) != 2:
        raise ValueError("Choose a QPSK MCS in table 1 (index 0..9)")
    tx = PUSCHTransmitter(cfg, output_domain="time")
    rx = PUSCHReceiver(tx, input_domain="time", l_min=0,
                       return_tb_crc_status=True)
    grid = tx.resource_grid
    rate = float(grid.fft_size * grid.subcarrier_spacing)
    return cfg, tx, rx, rate


def make_burst(tx, amplitude: float):
    """Create known TB bits, a PUSCH slot and a repeated BPSK sync preamble."""
    with torch.no_grad():
        x, bits = tx(1)
    pusch = np.asarray(x.detach().cpu().numpy()[0, 0, 0], dtype=np.complex64)
    truth = np.asarray(bits.detach().cpu().numpy()[0, 0], dtype=np.uint8)
    rng = np.random.default_rng(410)
    half = rng.choice(np.array([-1, 1], dtype=np.int8), size=127).astype(np.complex64)
    preamble = np.tile(half, 2)
    # Scale the entire useful burst to keep peak baseband magnitude below amplitude.
    scale = amplitude / max(1.0, float(np.max(np.abs(pusch))))
    guard = np.zeros(512, dtype=np.complex64)
    burst = np.concatenate((guard, scale * preamble, scale * pusch, guard))
    return np.ascontiguousarray(burst), preamble, pusch, truth, scale


def synchronize(capture: np.ndarray, preamble: np.ndarray, slot_len: int,
                rate: float):
    """Find the preamble, estimate CFO and one complex channel gain."""
    capture = np.asarray(capture, dtype=np.complex64).reshape(-1)
    length = len(preamble)
    if len(capture) < length + slot_len:
        raise ValueError("Capture is shorter than preamble plus PUSCH slot")
    corr = np.correlate(capture, preamble, mode="valid")
    energy = np.convolve(np.abs(capture) ** 2,
                         np.ones(length, dtype=np.float32), mode="valid")
    score = np.abs(corr) / np.sqrt(np.maximum(energy * length, 1e-20))
    score[len(capture) - length - slot_len + 1:] = 0
    start = int(np.argmax(score))
    metric = float(score[start])
    if metric < 0.50:
        raise RuntimeError(f"Preamble not found (normalized correlation {metric:.3f})")
    half = length // 2
    observed = capture[start:start + length]
    cfo = float(np.angle(np.vdot(observed[:half], observed[half:]))
                * rate / (2 * np.pi * half))
    segment = capture[start:start + length + slot_len]
    n = np.arange(len(segment), dtype=np.float64)
    corrected = segment * np.exp(-2j * np.pi * cfo * n / rate)
    gain = np.vdot(preamble, corrected[:length]) / np.vdot(preamble, preamble)
    if abs(gain) < 1e-5:
        raise RuntimeError("Estimated channel gain is too small")
    aligned = np.asarray(corrected[length:] / gain, dtype=np.complex64)
    # Guard immediately before the preamble gives a noise-only estimate. This
    # is a first-order estimate and must be checked for a real RF setup.
    guard = capture[max(0, start - 256):start]
    if len(guard) < 64:
        raise RuntimeError("Insufficient preamble guard samples for noise estimate")
    noise = float(np.mean(np.abs(guard) ** 2) / abs(gain) ** 2)
    noise = max(noise, 1e-6)
    return aligned, {"preamble_start_sample": start,
                     "correlation": metric, "cfo_hz": cfo,
                     "channel_gain_real": float(np.real(gain)),
                     "channel_gain_imag": float(np.imag(gain)),
                     "noise_variance_time": noise}


def decode(rx, aligned: np.ndarray, noise: float, truth: np.ndarray):
    y = torch.from_numpy(np.ascontiguousarray(aligned)).reshape(1, 1, 1, -1)
    with torch.no_grad():
        decoded, crc = rx(y, noise)
    bits = decoded.detach().cpu().numpy().reshape(-1).astype(np.uint8)
    if bits.shape != truth.shape:
        raise RuntimeError(f"Decoded TB shape {bits.shape} != sent {truth.shape}")
    errors = int(np.count_nonzero(bits != truth))
    return bits, {"tb_crc_ok": bool(crc.detach().cpu().numpy().reshape(-1)[0]),
                  "bit_errors": errors, "tb_bits": int(truth.size),
                  "ber": errors / int(truth.size)}


def software_capture(burst: np.ndarray, rate: float):
    """Deterministic delay, channel gain, CFO and AWGN for a hardware-free check."""
    rng = np.random.default_rng(17)
    padded = np.pad(burst, (173, 300)).astype(np.complex64)
    n = np.arange(len(padded))
    signal = 0.7 * np.exp(0.3j) * padded * np.exp(2j * np.pi * 150 * n / rate)
    noise = (rng.standard_normal(len(padded)) +
             1j * rng.standard_normal(len(padded))) * 2e-4
    return np.asarray(signal + noise, dtype=np.complex64)


def x410_capture(args, burst: np.ndarray, rate: float):
    """Timed full-duplex transmit and finite receive using one UHD MultiUSRP."""
    try:
        import uhd
    except ImportError as exc:
        raise RuntimeError("UHD Python bindings are required for --mode hardware") from exc

    usrp = uhd.usrp.MultiUSRP(args.device)
    if "X410" not in usrp.get_pp_string():
        raise RuntimeError("The selected UHD device does not identify as an X410")
    tx_ants = list(usrp.get_tx_antennas(args.channel))
    rx_ants = list(usrp.get_rx_antennas(args.channel))
    if args.tx_antenna not in tx_ants or args.rx_antenna not in rx_ants:
        raise RuntimeError(f"Antenna unavailable: TX {tx_ants}, RX {rx_ants}")
    usrp.set_tx_antenna(args.tx_antenna, args.channel)
    usrp.set_rx_antenna(args.rx_antenna, args.channel)
    usrp.set_tx_gain(args.tx_gain, args.channel)
    usrp.set_rx_gain(args.rx_gain, args.channel)
    usrp.set_tx_freq(args.freq, args.channel)
    usrp.set_rx_freq(args.freq, args.channel)
    usrp.set_tx_rate(rate, args.channel)
    usrp.set_rx_rate(rate, args.channel)
    actual = {"device": args.device, "device_info": usrp.get_pp_string(),
              "channel": args.channel, "tx_antenna": usrp.get_tx_antenna(args.channel),
              "rx_antenna": usrp.get_rx_antenna(args.channel),
              "tx_gain_db": float(usrp.get_tx_gain(args.channel)),
              "rx_gain_db": float(usrp.get_rx_gain(args.channel)),
              "tx_freq_hz": float(usrp.get_tx_freq(args.channel)),
              "rx_freq_hz": float(usrp.get_rx_freq(args.channel)),
              "tx_rate_hz": float(usrp.get_tx_rate(args.channel)),
              "rx_rate_hz": float(usrp.get_rx_rate(args.channel)),
              "cable_attenuation_db": args.attenuation_db}
    for key in ("tx_rate_hz", "rx_rate_hz"):
        if abs(actual[key] - rate) > 1.0:
            raise RuntimeError(f"{key}={actual[key]} differs from PUSCH rate {rate}; resampling is required")
    for key in ("tx_freq_hz", "rx_freq_hz"):
        if abs(actual[key] - args.freq) > 1.0:
            raise RuntimeError(f"{key}={actual[key]} differs from requested frequency {args.freq}")

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [args.channel]
    tx_stream = usrp.get_tx_stream(stream_args)
    rx_stream = usrp.get_rx_stream(stream_args)
    # RX starts first. TX begins 50 ms later, using the same device time base.
    lead = int(round(0.05 * rate))
    total = lead + len(burst) + int(round(0.05 * rate))
    start_time = usrp.get_time_now().get_real_secs() + 0.5
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    cmd.num_samps = total
    cmd.stream_now = False
    cmd.time_spec = uhd.types.TimeSpec(start_time)
    rx_stream.issue_stream_cmd(cmd)

    def receive():
        out = np.empty(total, dtype=np.complex64)
        buf = np.empty((1, rx_stream.get_max_num_samps()), dtype=np.complex64)
        md = uhd.types.RXMetadata()
        count = 0
        while count < total:
            got = rx_stream.recv(buf, md, timeout=2.0)
            if md.error_code != uhd.types.RXMetadataErrorCode.none:
                raise RuntimeError(f"UHD receive error after {count} samples: {md.strerror()}")
            if got <= 0:
                raise RuntimeError(f"UHD receive returned no samples after {count}")
            take = min(int(got), total - count)
            out[count:count + take] = buf[0, :take]
            count += take
        return out

    def transmit():
        md = uhd.types.TXMetadata()
        md.start_of_burst = True
        md.end_of_burst = False
        md.has_time_spec = True
        md.time_spec = uhd.types.TimeSpec(start_time + 0.05)
        offset = 0
        chunk = tx_stream.get_max_num_samps()
        while offset < len(burst):
            samples = np.ascontiguousarray(burst[offset:offset + chunk].reshape(1, -1))
            sent = int(tx_stream.send(samples, md, timeout=2.0))
            if sent <= 0:
                raise RuntimeError(f"UHD transmit timeout after {offset} samples")
            offset += sent
            md.start_of_burst = False
            md.has_time_spec = False
        md.end_of_burst = True
        tx_stream.send(np.empty((1, 0), dtype=np.complex64), md, timeout=2.0)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(receive)
        transmit()
        capture = future.result()
    actual["rx_start_device_time_s"] = start_time
    actual["tx_start_device_time_s"] = start_time + 0.05
    actual["rx_samples"] = int(total)
    return capture, actual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("selftest", "hardware"), default="selftest")
    p.add_argument("--output", type=Path, default=Path("capture_b0"))
    p.add_argument("--prbs", type=int, default=4)
    p.add_argument("--mcs", type=int, default=4)
    p.add_argument("--amplitude", type=float, default=0.1)
    p.add_argument("--device", help='X410 UHD address, e.g. "addr=192.168.10.2"')
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--tx-antenna", default="TX/RX0")
    p.add_argument("--rx-antenna", default="RX1")
    p.add_argument("--freq", type=float, default=2.4e9)
    p.add_argument("--tx-gain", type=float, default=0.0)
    p.add_argument("--rx-gain", type=float, default=0.0)
    p.add_argument("--attenuation-db", type=float,
                   help="Measured cable attenuation including pads; required for hardware mode")
    args = p.parse_args()
    if args.prbs < 1 or not 0 <= args.mcs <= 9 or not 0 < args.amplitude <= 0.5:
        p.error("Use positive PRBs, QPSK MCS 0..9, and amplitude in (0, 0.5]")
    if args.mode == "hardware":
        if not args.device or args.attenuation_db is None or args.attenuation_db < 40:
            p.error("Hardware mode requires --device and at least 40 dB measured cable attenuation")
    torch.manual_seed(410)
    cfg, tx, rx, rate = make_link(args.prbs, args.mcs)
    burst, preamble, pusch, truth, scale = make_burst(tx, args.amplitude)
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / "tx_burst.npy", burst)
    np.save(args.output / "tx_tb_bits.npy", truth)
    if args.mode == "hardware":
        capture, actual = x410_capture(args, burst, rate)
    else:
        capture, actual = software_capture(burst, rate), {"synthetic": True}
    np.save(args.output / "rx_raw_iq.npy", capture)
    report = {"mode": args.mode, "hardware_tested": args.mode == "hardware",
              "prbs": args.prbs, "mcs": args.mcs, "tb_size_bits": int(cfg.tb_size),
              "sample_rate_hz": rate, "tx_scale": scale,
              "slot_samples": int(len(pusch)), "tx_burst_samples": int(len(burst)),
              "hardware": actual, "rx_peak_magnitude": float(np.max(np.abs(capture)))}
    try:
        aligned, sync = synchronize(capture, preamble, len(pusch), rate)
        np.save(args.output / "rx_aligned_pusch.npy", aligned)
        decoded, result = decode(rx, aligned, sync["noise_variance_time"], truth)
        np.save(args.output / "rx_tb_bits.npy", decoded)
        report.update({"sync": sync, "result": result})
        print(f"PUSCH CRC={result['tb_crc_ok']}, bit errors={result['bit_errors']}/{result['tb_bits']}")
        if args.mode == "selftest" and (not result["tb_crc_ok"] or result["bit_errors"]):
            raise RuntimeError("Software self-test did not decode the known TB correctly")
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Capture and report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
