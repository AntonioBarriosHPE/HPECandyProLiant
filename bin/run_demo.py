#!/usr/bin/python

import sys
import time
import subprocess
import os

while True:
    data = input()
    print("DEMOSTARTER:%s" % data)
    if (data == "CPU+CPU"):
        os.system("run_demo_targets.bat CPU CPU")
    if (data == "VPU+VPU"):
        #subprocess.Popen(["run_demo_targets.bat", "MYRIAD", "MYRIAD"])
        os.system("run_demo_targets.bat MYRIAD MYRIAD")
    if (data == "HDDL+HDDL"):
        #subprocess.call(["run_demo_targets.bat", "MYRIAD", "MYRIAD"], env=e)
        #subprocess.call(["C:\\Users\\iot\\Desktop\\demo_v8\\bin\\websocketd", "--port=8080", "--staticdir", ".", "C:\\Users\\iot\\Desktop\\demo_v8\\bin\\candy_demo.exe", "-i", "cam", "-d", "HDDL", "-d_em", "HDDL"])
        print("STARTING HDDL VIA START_INFERENCE.BAT")
        os.system("start_inference.bat HDDL")        
    time.sleep(1)