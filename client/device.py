import RPi.GPIO as GPIO
import time


def SG90():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    servopin = 4
    GPIO.setup(servopin, GPIO.OUT)
    p = GPIO.PWM(servopin, 50)
    p.start(0)
    time.sleep(0.5)
    p.ChangeDutyCycle(2.5)
    time.sleep(5)
    p.ChangeDutyCycle(7.5)
    time.sleep(0.5)
    GPIO.cleanup()

