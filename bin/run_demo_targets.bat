rem OpenVINO 2018.5.456
rem call "C:\Intel\computer_vision_sdk\bin\setupvars.bat"

rem OpenVINO 2019.2.275
rem call "C:\Program Files (x86)\IntelSWTools\openvino_2019.2.275\bin\setupvars.bat"

websocketd.exe --port=8080 candy_demo.exe -i cam -d %1 -d_em %2
