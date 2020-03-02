#!/usr/bin/env python
import time
import serial

Serial = serial.Serial(
        # port='/dev/ttyS0', #    Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        port = '/dev/ttyS0',
        baudrate = 115200, # Needed for flight controller
        # Want SERIAL8E2
        parity=serial.PARITY_EVEN,
        stopbits=serial.STOPBITS_TWO,
        bytesize=serial.EIGHTBITS
)

while 1:
    x=Serial.readline()
    print(x)