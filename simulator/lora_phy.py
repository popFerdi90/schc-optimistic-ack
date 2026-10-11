"""
LoRa/LoRaWAN physical-layer model: Time-on-Air, duty-cycle and energy.

Time-on-Air follows the Semtech LoRa Modem Designer's Guide (AN1200.13) and the
SX1272/76 datasheet; LoRaWAN framing overhead follows the LoRaWAN 1.0.4
specification (EU868 regional parameters).

All durations are in seconds, energies in joules (unless stated otherwise).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Radio constants
# ---------------------------------------------------------------------------

BW_DEFAULT = 125_000.0        # Hz, EU868 default bandwidth
N_PREAMBLE = 8                # LoRaWAN uplink preamble symbols
CR_DEFAULT = 1                # coding rate 4/5  -> CR = 1

# SX1276 current draw (datasheet, 3.3 V supply)
V_SUPPLY = 3.3                # V
I_TX_14DBM = 28.0e-3          # A, PA_BOOST at +14 dBm
I_RX = 11.5e-3                # A, receiver with LNA boost
I_SLEEP = 1.5e-6              # A

# LoRaWAN Class A receive windows (EU868)
RX1_DELAY = 1.0               # s after end of uplink
RX2_DELAY = 2.0               # s after end of uplink
RX_SYMB_TIMEOUT = 5           # symbols the node listens for a preamble

# LoRaWAN MAC framing overhead carried in the PHY payload:
#   MHDR(1) + FHDR(7, no FOpts) + FPort(1) + MIC(4)
LORAWAN_OVERHEAD_B = 13

# EU868 duty-cycle limits per sub-band
DUTY_CYCLE_G1 = 0.01          # 868.0-868.6 MHz  (uplink channels)  -> 1 %
DUTY_CYCLE_G3 = 0.10          # 869.4-869.65 MHz (RX2 downlink)     -> 10 %

# EU868 maximum application payload per data rate (repeater-compatible)
MAX_APP_PAYLOAD = {12: 51, 11: 51, 10: 51, 9: 115, 8: 222, 7: 222}


def low_data_rate_optimize(sf: int, bw: float = BW_DEFAULT) -> int:
    """LowDataRateOptimize is mandated when the symbol time exceeds 16 ms."""
    return 1 if (2 ** sf) / bw > 16e-3 else 0


def symbol_time(sf: int, bw: float = BW_DEFAULT) -> float:
    return (2 ** sf) / bw


def time_on_air(payload_b: int, sf: int, bw: float = BW_DEFAULT,
                cr: int = CR_DEFAULT, crc: int = 1, implicit_header: int = 0,
                n_preamble: int = N_PREAMBLE) -> float:
    """Time-on-Air of a LoRa frame whose PHY payload is ``payload_b`` bytes."""
    t_sym = symbol_time(sf, bw)
    t_preamble = (n_preamble + 4.25) * t_sym
    de = low_data_rate_optimize(sf, bw)
    num = 8 * payload_b - 4 * sf + 28 + 16 * crc - 20 * implicit_header
    den = 4 * (sf - 2 * de)
    n_payload = 8 + max(math.ceil(num / den) * (cr + 4), 0)
    return t_preamble + n_payload * t_sym


def toa_uplink(app_payload_b: int, sf: int, **kw) -> float:
    """ToA of an uplink carrying ``app_payload_b`` bytes of FRMPayload."""
    return time_on_air(app_payload_b + LORAWAN_OVERHEAD_B, sf, crc=1, **kw)


def toa_downlink(app_payload_b: int, sf: int, **kw) -> float:
    """ToA of a downlink; LoRaWAN downlinks carry no PHY CRC."""
    return time_on_air(app_payload_b + LORAWAN_OVERHEAD_B, sf, crc=0, **kw)


def time_off(toa: float, duty_cycle: float = DUTY_CYCLE_G1) -> float:
    """Mandatory silence after a transmission of duration ``toa``."""
    return toa * (1.0 / duty_cycle - 1.0)


# ---------------------------------------------------------------------------
# Energy accounting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnergyModel:
    v_supply: float = V_SUPPLY
    i_tx: float = I_TX_14DBM
    i_rx: float = I_RX
    i_sleep: float = I_SLEEP

    def e_tx(self, toa: float) -> float:
        return self.v_supply * self.i_tx * toa

    def e_rx(self, duration: float) -> float:
        return self.v_supply * self.i_rx * duration

    def e_rx_window_idle(self, sf_rx1: int, sf_rx2: int = 12) -> float:
        """Energy of a Class A pair of receive windows that stay empty.

        The node wakes for RX1, detects no preamble within RX_SYMB_TIMEOUT
        symbols, sleeps, then wakes again for RX2 and does the same.
        """
        listen1 = RX_SYMB_TIMEOUT * symbol_time(sf_rx1)
        listen2 = RX_SYMB_TIMEOUT * symbol_time(sf_rx2)
        return self.e_rx(listen1 + listen2)

    def e_rx_window_ack(self, ack_toa: float, sf_rx1: int,
                        on_rx2: bool = True) -> float:
        """Energy of a receive-window pair in which an ACK is actually received.

        When the ACK is served on RX2 (SF12, 869.525 MHz) the node has first
        paid for an empty RX1 listening period.
        """
        listen1 = RX_SYMB_TIMEOUT * symbol_time(sf_rx1)
        if on_rx2:
            return self.e_rx(listen1 + ack_toa)
        return self.e_rx(ack_toa)


DEFAULT_ENERGY = EnergyModel()
