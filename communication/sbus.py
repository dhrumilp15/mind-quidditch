#!/usr/bin/env python
import time
import serial
from ctypes import c_uint8, c_ubyte

# Constants 

RC_CHANNEL_MIN = 990
RC_CHANNEL_MAX = 2010

SBUS_MIN_OFFSET = 173
SBUS_MID_OFFSET = 992
SBUS_MAX_OFFSET = 1811
SBUS_CHANNEL_NUMBER = 16
SBUS_PACKET_LENGTH = 25
SBUS_FRAME_HEADER = 0x0f
SBUS_FRAME_FOOTER = 0x00
SBUS_FRAME_FOOTER_V2 = 0x04
SBUS_STATE_FAILSAFE = 0x08
SBUS_STATE_SIGNALLOSS = 0x04
SBUS_UPDATE_RATE = 15 * 1e-3 # ms

# def arduinomap(x : int, in_min: int, in_max: int, out_min: int, out_max: int) -> int:
def arduinomap(x, in_min, in_max, out_min, out_max):
    '''
        Map given data from a given range to a desired range - effectively normalizes data
        :return: Mapped data
    '''
    return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

# def sbusPreparePacket(packet: list, channels : list, isSignalLoss : bool, isFailSafe: bool) -> list:
def sbusPreparePacket(packet, channels, isSignalLoss, isFailSafe):
    '''
        Prepares the packet to be sent to the flight controller using the given packet
        :return: The final packet as a list
    '''
    output = [0] * SBUS_CHANNEL_NUMBER
    output[0] = 0x00

    for i in range(SBUS_CHANNEL_NUMBER):
        output[i] = arduinomap(channels[i], RC_CHANNEL_MIN, RC_CHANNEL_MAX, SBUS_MIN_OFFSET, SBUS_MAX_OFFSET)
    
    statebyte = 0x00
    if isSignalLoss:
        statebyte |= SBUS_STATE_SIGNALLOSS
    
    if isFailSafe:
        statebyte |= SBUS_STATE_FAILSAFE
    packet[0] = SBUS_FRAME_HEADER

    packet[1] = (output[0] & 0x07FF)
    packet[2] = ((output[0] & 0x07FF) >> 8 | (output[1] & 0x07FF) << 3)
    packet[3] = ((output[1] & 0x07FF) >> 5 | (output[2] & 0x07FF) << 6)
    packet[4] = ((output[2] & 0x07FF) >> 2)
    packet[5] = ((output[2] & 0x07FF) >> 10 | (output[3] & 0x07FF) << 1)
    packet[6] = ((output[3] & 0x07FF) >> 7 | (output[4] & 0x07FF) << 4)
    packet[7] = ((output[4] & 0x07FF) >> 4 | (output[5] & 0x07FF) << 7)
    packet[8] = ((output[5] & 0x07FF) >> 1)
    packet[9] = ((output[5] & 0x07FF) >> 9 | (output[6] & 0x07FF) << 2)
    packet[10] = ((output[6] & 0x07FF) >> 6 | (output[7] & 0x07FF) << 5)
    packet[11] = ((output[7] & 0x07FF) >> 3)
    packet[12] = ((output[8] & 0x07FF))
    packet[13] = ((output[8] & 0x07FF) >> 8 | (output[9] & 0x07FF) << 3)
    packet[14] = ((output[9] & 0x07FF) >> 5 | (output[10] & 0x07FF) << 6)
    packet[15] = ((output[10] & 0x07FF) >> 2)
    packet[16] = ((output[10] & 0x07FF) >> 10 | (output[11] & 0x07FF) << 1)
    packet[17] = ((output[11] & 0x07FF) >> 7 | (output[12] & 0x07FF) << 4)
    packet[18] = ((output[12] & 0x07FF) >> 4 | (output[13] & 0x07FF) << 7)
    packet[19] = ((output[13] & 0x07FF) >> 1)
    packet[20] = ((output[13] & 0x07FF) >> 9 | (output[14] & 0x07FF) << 2)
    packet[21] = ((output[14] & 0x07FF) >> 6 | (output[15] & 0x07FF) << 5)
    packet[22] = ((output[15] & 0x07FF) >> 3)

    packet[23] = statebyte
    packet[24] = SBUS_FRAME_FOOTER_V2
    return packet
    # return bytearray(packet)
    # return list(map(lambda byte: bin(byte)[2:].zfill(8),packet))

# Initialize packet
sbuspacket = [None] * SBUS_PACKET_LENGTH
rcChannels = [1500] * SBUS_CHANNEL_NUMBER
rcChannels[2] = 1000
rcChannels[4] = 1200
sbustime = 0

Serial = serial.Serial(
        port = '/dev/serial0', # Switch to AMA0 if needed
        # baudrate = 9600
        # For flight controller:
        baudrate = 100000, 
        parity=serial.PARITY_EVEN,
        stopbits=serial.STOPBITS_TWO,
        bytesize=serial.EIGHTBITS
)

def convtobyte(packet):
    return list(map(lambda x: c_ubyte(x).value, packet))

while True:
    currtime = time.time()
    
    rcChannels[2] += 2
    if rcChannels[2] > 2000:
        rcChannels[2] = 1000
    time.sleep(100 * 1e-3)
    # rcChannels[2] += 2
    if currtime > sbustime:
        sbuspacket = sbusPreparePacket(sbuspacket, rcChannels, False, False)
        final = convtobyte(sbuspacket)
        print("Any greater than 255? {}".format(any(item > 255 for item in final)))
        print(bytearray(final))
        Serial.write(bytearray(final))
        sbustime = currtime + SBUS_UPDATE_RATE