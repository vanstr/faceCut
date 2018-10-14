import sys
import serial

ser = serial.Serial('COM4')

if ser.isOpen():
    ser.close()
ser.open()
ser.isOpen()

ser.write("F\n")
