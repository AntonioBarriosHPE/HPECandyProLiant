call "C:\Program Files (x86)\IntelSWTools\openvino_2019.2.275\bin\setupvars.bat"
rem call "C:\Program Files (x86)\IntelSWTools\openvino_2019.2.275\inference_engine\external\hddl\bin\hddldaemon.exe"
cd c:\Users\HPE\Desktop\demo_v8\bin
start websocketd --port=9000 --staticdir . python run_demo.py 
timeout 1

"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --start-fullscreen http://127.0.0.1:9000
rem "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" http://127.0.0.1:9000