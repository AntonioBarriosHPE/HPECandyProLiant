rem OpenVINO 2018.5.456
rem call "C:\Intel\computer_vision_sdk\bin\setupvars.bat"

rem OpenVINO 2019.2.275
call "C:\Program Files (x86)\IntelSWTools\openvino_2019.2.275\bin\setupvars.bat"
call "C:\Program Files (x86)\IntelSWTools\openvino_2019.2.275\inference_engine\external\hddl\bin\hddldaemon.exe"
websocketd.exe --port=8080 candy_demo.exe -i cam -d %1 -d_em %1
