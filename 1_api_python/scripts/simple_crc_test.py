"""
SIMPLE CRC TEST

What a "checksum" (CRC) is for: every packet the Pico sends ends with a
small number (the CRC) that is calculated from the rest of the packet's
bytes. If even one bit in the packet gets damaged, recalculating that
number on the receiving side gives a DIFFERENT result - so we can tell
the data is bad without knowing anything else about it.

This script:
  1. Connects to the Pico.
  2. Asks it to start sending data, and reads one real packet.
  3. Recalculates the checksum ourselves and checks it matches (PASS case).
  4. Damages a copy of that same packet and checks the checksum again,
     to show it now correctly does NOT match (FAIL case).

Run this from the 1_api_python folder:
    python scripts/simple_crc_test.py
"""
from time import sleep
from api.mcu_api import DeviceAPI, Commands
from api.src._helper import build_crc_excluding_endframe


# ---- Step 1: connect to the Pico ----
print("Connecting to the Pico...")
dut = DeviceAPI(com_name="AUTOCOM")
print("Connected.\n")

# Ask the Pico how its data packets are laid out (how many bytes, etc.)
info = dut.get_daq_characteristics()
packet_size = info.num_bytes_total
print(f"Each data packet from the Pico is {packet_size} bytes long.")

# Speed up sampling a bit so we don't have to wait long for a full packet.
# (This next line and the two marked "behind the scenes" below use some
# internal parts of the library that aren't meant for everyday use - they're
# just the simplest way to grab the raw bytes for this test.)
dut._update_daq_sampling_rate(50)


# ---- Step 2: start the Pico sending data, and read one packet ----
print("\nStarting data acquisition...")
dut._write_without_feedback(Commands.START_DAQ)   # behind the scenes: sends the "start" command
sleep(0.5)  # give it a moment to start sending

raw_device = dut._DeviceAPI__device   # behind the scenes: the raw USB connection
packet = b""
while len(packet) < packet_size:
    packet += raw_device.read(packet_size - len(packet))

dut._write_without_feedback(Commands.STOP_DAQ)   # tell the Pico to stop sending
dut.close()

print(f"\nCaptured one packet ({len(packet)} bytes):")
print(packet.hex(' '))


# ---- Step 3: PASS case - check this real, untouched packet ----
# The last 3 bytes of every packet are: CRC low byte, CRC high byte, then
# a fixed "end of packet" marker byte.
crc_sent_by_pico = packet[-3] + (packet[-2] << 8)
crc_we_calculate = build_crc_excluding_endframe(packet)

print(f"\nCRC the Pico sent with the packet: {hex(crc_sent_by_pico)}")
print(f"CRC we calculate ourselves:        {hex(crc_we_calculate)}")

if crc_we_calculate == crc_sent_by_pico:
    print("PASS - the checksums match, so this packet is correct.")
else:
    print("FAIL - the checksums do not match (unexpected for a real packet!)")


# ---- Step 4: FAIL case - damage a copy of the packet and check again ----
damaged_packet = bytearray(packet)
damaged_packet[10] = damaged_packet[10] ^ 1   # flip one bit, somewhere in the data
damaged_packet = bytes(damaged_packet)

print(f"\nDamaged a copy of the packet by changing one bit in byte 10:")
print(f"  before: {packet[10]:#04x}")
print(f"  after:  {damaged_packet[10]:#04x}")

crc_we_calculate_damaged = build_crc_excluding_endframe(damaged_packet)
print(f"\nCRC the Pico originally sent:                {hex(crc_sent_by_pico)}")
print(f"CRC we calculate on the DAMAGED packet:      {hex(crc_we_calculate_damaged)}")

if crc_we_calculate_damaged != crc_sent_by_pico:
    print("GOOD - the checksums no longer match, so the damage was correctly caught.")
else:
    print("PROBLEM - the damaged packet still matched. That should not happen.")
