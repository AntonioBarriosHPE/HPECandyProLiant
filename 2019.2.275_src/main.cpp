// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine interactive_face_detection demo application
* \file interactive_face_detection_demo/main.cpp
* \example interactive_face_detection_demo/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>
#include <list>
#include <set>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include "interactive_face_detection.hpp"
#include "detectors.hpp"
#include "face.hpp"
#include "visualizer.hpp"

#include <ie_iextension.h>
#include <ext_list.hpp>

/*  HPE  */
#include "deps/mcp2221_dll_um.h"

#pragma comment(lib, "deps/mcp2221_dll_um_x64.lib")  //Link MCP2221 library, applicable for visual studio only
// Note that you should add the base path to the Linker additional libraries folder and not include deps/ on the end

#include "base64.h"
#include <iostream>
#include <thread>
//#include <boost/beast/core.hpp>
//#include <boost/beast/websocket.hpp>
//#include <boost/asio/ip/tcp.hpp>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>

//using tcp = boost::asio::ip::tcp;               // from <boost/asio/ip/tcp.hpp>
//namespace websocket = boost::beast::websocket;  // from <boost/beast/websocket.hpp>

// Global variables for web options
double tempalpha = 0.95; // 0.05;
float candy_duration = 2000;
auto lineColor = cv::Scalar(100, 100, 100);
double id_window_size = 1;  // 0.20
float detect_left = 100; //  -1;
float detect_right = 500; //  -1;
size_t gwidth = 0;
size_t gheight = 0;
double text_scale = 0.5;
bool statson = true;
auto statsColor = cv::Scalar(0, 255, 0);
bool showEmotionBar = false;

float out_width = 640;
float out_height = 480;
float face_width = 300;
float face_height = 300;

bool candyOn = false;
bool candyGiven = false;

std::mutex msg_mutex;
std::string msg;

void read()
{
	size_t pos = 0;
	std::string key;
	std::string value;
	std::string::size_type sz;

	while (true) {
		std::string sin;
		//std::cin >> sin;
		std::getline(std::cin, sin);
		// The mutex causes this to not work well or take too long to get a lock so ignoring... YOLO
		//std::lock_guard<std::mutex> lock{ msg_mutex };
		msg = sin;
		//std::cout << "read_thread " << msg << std::endl;
		//std::this_thread::sleep_for(std::chrono::milliseconds(1));

		// Parsing for options
		pos = msg.find(":");
		key = msg.substr(0, pos);
		value = msg.substr(pos + 1, msg.length());
		if (key == "alpha") {
			std::cout << "SETTING ALPHA to " << value << std::endl;
			::tempalpha = std::stod(value, &sz);
		}
		if (key == "duration") {
			std::cout << "SETTING DURATION to " << value << std::endl;
			::candy_duration = std::stod(value, &sz);
		}
		if (key == "emotionbar") {
			std::cout << "SETTING EMOTIONBAR to " << value << std::endl;
			if (value == "true") {
				::showEmotionBar = true;
			}
			else {
				::showEmotionBar = false;
			}
		}
		if (key == "linecolor") {
			std::cout << "SETTING LINE COLOR to " << value << std::endl;
			if (value == "Blue") {
				::lineColor = cv::Scalar(255, 0, 0);
			}
			if (value == "Green") {
				::lineColor = cv::Scalar(0, 255, 0);
			}
			if (value == "Red") {
				::lineColor = cv::Scalar(0, 0, 255);
			}
			if (value == "Black") {
				::lineColor = cv::Scalar(0, 0, 0);
			}
			if (value == "White") {
				::lineColor = cv::Scalar(255, 255, 255);
			}
			if (value == "Purple") {
				::lineColor = cv::Scalar(255, 0, 255);
			}
			if (value == "Teal") {
				::lineColor = cv::Scalar(255, 255, 0);
			}
			if (value == "Yellow") {
				::lineColor = cv::Scalar(0, 255, 255);
			}
			if (value == "Gray") {
				::lineColor = cv::Scalar(100, 100, 100);
			}
		}
		if (key == "statscolor") {
			std::cout << "SETTING STATS COLOR to " << value << std::endl;

			::statson = true;
			if (value == "Blue") {
				::statsColor = cv::Scalar(255, 0, 0);
			}
			if (value == "Green") {
				::statsColor = cv::Scalar(0, 255, 0);
			}
			if (value == "Red") {
				::statsColor = cv::Scalar(0, 0, 255);
			}
			if (value == "Black") {
				::statsColor = cv::Scalar(0, 0, 0);
			}
			if (value == "White") {
				::statsColor = cv::Scalar(255, 255, 255);
			}
			if (value == "Purple") {
				::statsColor = cv::Scalar(255, 0, 255);
			}
			if (value == "Teal") {
				::statsColor = cv::Scalar(255, 255, 0);
			}
			if (value == "Yellow") {
				::statsColor = cv::Scalar(0, 255, 255);
			}
			if (value == "Gray") {
				::statsColor = cv::Scalar(100, 100, 100);
			}
			if (value == "Off") {
				::statson = false;
			}
		}
		if (key == "windowsize") {
			std::cout << "SETTING ID WINDOW SIZE to " << value << std::endl;
			::id_window_size = std::stod(value, &sz);
			// middle = ::gwidth / 2
			// default is 0.2 or 20% of the window 
			// divide the window size by 2...multiply that * width and subtract from left and add to right
			detect_left = (::gwidth / 2) - ((::id_window_size / 2) * ::gwidth);
			detect_right = (::gwidth / 2) + ((::id_window_size / 2) * ::gwidth);
			//detect_left = (::gwidth / 5) * 2;
			//detect_right = (::gwidth / 5) * 3;
		}
		if (key == "candy") {
			if (value == "on") {
				printf("RECEIVED CANDY ON\n");
				::candyOn = true;
			}
			if (value == "off") {
				printf("RECEIVED CANDY OFF\n");
				::candyOn = false;
			}
		}

	}
}



// Websocket handler portion
// Echoes back all received WebSocket messages
/*
void
do_session(tcp::socket& socket)
{
	try
	{
		// Construct the stream by moving in the socket
		websocket::stream<tcp::socket> ws{ std::move(socket) };

		// Accept the websocket handshake
		ws.accept();

		for (;;)
		{
			// This buffer will hold the incoming message
			boost::beast::multi_buffer buffer;

			// Read a message
			ws.read(buffer);

			// Echo the message back
			ws.text(ws.got_text());
			ws.write(buffer.data());
		}
	}
	catch (boost::system::system_error const& se)
	{
		// This indicates that the session was closed
		if (se.code() != websocket::error::closed)
			std::cerr << "Error: " << se.code().message() << std::endl;
	}
	catch (std::exception const& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
	}
}
*/

//Global variables
void* handle;

void ExitFunc()
{

	printf("Closing any MCP2221 connections\n");
	_sleep(10);
	//Mcp2221_Reset(handle);

	//Close all devices at exit
	Mcp2221_CloseAll();
}


/*  END HPE  */


using namespace InferenceEngine;


bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

	/* HPE ADD */
	if (FLAGS_d == "CPU") {
		printf("FDEVICE:CPU\n");
		printf("Overriding model information for CPU (FP32)\n");
		FLAGS_m = "models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml";
		//FLAGS_m_em = "C:/Intel/computer_vision_sdk_2018.5.456/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml";
		//FLAGS_m_hp = "C:\Intel\computer_vision_sdk_2018.5.456\deployment_tools\intel_models\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml";
	}
	if (FLAGS_d == "MYRIAD") {
		printf("FDEVICE:VPU\n");
		printf("Overriding model information for non-CPU (FP16)\n");
		FLAGS_m = "models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml";
		//FLAGS_m_em = "C:/Intel/computer_vision_sdk_2018.5.456/deployment_tools/intel_models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml";
		//FLAGS_m_hp = "C:\Intel\computer_vision_sdk_2018.5.456\deployment_tools\intel_models\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001.xml";
	}
	if (FLAGS_d == "HDDL") {
		printf("FDEVICE:HDDL\n");
		printf("Overriding model information for non-CPU (FP16)\n");
		FLAGS_m = "models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml";
		//FLAGS_m_em = "C:/Intel/computer_vision_sdk_2018.5.456/deployment_tools/intel_models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml";
		//FLAGS_m_hp = "C:\Intel\computer_vision_sdk_2018.5.456\deployment_tools\intel_models\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001.xml";
	}
	if (FLAGS_d_em == "CPU") {
		printf("EDEVICE:CPU\n");
		printf("Overriding model information for CPU (FP32)\n");
		//FLAGS_m = "C:/Intel/computer_vision_sdk_2018.5.456/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml";
		FLAGS_m_em = "models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml";
		//FLAGS_m_hp = "C:\Intel\computer_vision_sdk_2018.5.456\deployment_tools\intel_models\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml";
	}
	if (FLAGS_d_em == "MYRIAD") {
		printf("EDEVICE:VPU\n");
		printf("Overriding model information for non-CPU (FP16)\n");
		//FLAGS_m = "C:/Intel/computer_vision_sdk_2018.5.456/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml";
		FLAGS_m_em = "models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml";
		//FLAGS_m_hp = "C:\Intel\computer_vision_sdk_2018.5.456\deployment_tools\intel_models\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001.xml";
	}
	if (FLAGS_d_em == "HDDL") {
		printf("EDEVICE:HDDL\n");
		printf("Overriding model information for non-CPU (FP16)\n");
		//FLAGS_m = "C:/Intel/computer_vision_sdk_2018.5.456/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml";
		FLAGS_m_em = "models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml";
		//FLAGS_m_hp = "C:\Intel\computer_vision_sdk_2018.5.456\deployment_tools\intel_models\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001.xml";
	}
	//printf("FLAGS: Candy duration is set to %d ms\n", FLAGS_candy_duration);
	/* END HPE */

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_n_ag < 1) {
        throw std::logic_error("Parameter -n_ag cannot be 0");
    }

    if (FLAGS_n_hp < 1) {
        throw std::logic_error("Parameter -n_hp cannot be 0");
    }

    // no need to wait for a key press from a user if an output image/video file is not shown.
    FLAGS_no_wait |= FLAGS_no_show;

    return true;
}

int main(int argc, char *argv[]) {

	/* HPE ADD */
	unsigned char txdata[2];
	int error = 0;

	wchar_t LibVer[6];
	wchar_t MfrDescriptor[30];
	wchar_t ProdDescrip[30];
	bool i2cpresent = false;
	unsigned char pinFunc[4] = { MCP2221_GPFUNC_IO, MCP2221_GPFUNC_IO, MCP2221_GP_DAC, MCP2221_GPFUNC_IO };  //Set GP0, GP1, GP3 as digital IO and GP2 as DAC
	unsigned char pinDir[4] = { MCP2221_GPDIR_OUTPUT, MCP2221_GPDIR_OUTPUT, NO_CHANGE, MCP2221_GPDIR_OUTPUT };  //configure GP0, GP1, GP3 as digital output
	unsigned char OutValues[4] = { 0, 0, NO_CHANGE, 0 };   //set initial values to 0's
	unsigned char PowerAttrib;
	unsigned char DacVal = 31;
	unsigned char DacRefValue = 0;
	unsigned int ReqCurrent;
	unsigned int PID = 0xDD;
	unsigned int VID = 0x4D8;
	unsigned int NumOfDev = 0;

	std::thread reader(read);

	// not sure we need this...
	std::setbuf(stdout, NULL);

	/*  END HPE */

    try {

		/* HPE ADD */
		int ver = 0;
		//int error = 0;
		int flag = 0;

		// Websocket Initialization
		/*
		printf("Starting websocket server\n");
		try
		{
			auto const address = boost::asio::ip::make_address("127.0.0.1");
			auto const port = static_cast<unsigned short>(8080);

			// The io_context is required for all I/O
			boost::asio::io_context ioc{ 1 };

			// The acceptor receives incoming connections
			tcp::acceptor acceptor{ ioc, {address, port} };
			for (;;)
			{
				// This will receive the new connection
				tcp::socket socket{ ioc };

				// Block until we get a connection
				acceptor.accept(socket);

				// Launch the session, transferring ownership of the socket
				std::thread{ std::bind(
					&do_session,
					std::move(socket)) }.detach();
			}
		}
		catch (const std::exception& e)
		{
			std::cerr << "Error: " << e.what() << std::endl;
			return EXIT_FAILURE;
		}
		*/

		// DIO Initialization
		if (1) { // DIO initialization
			printf("Starting DIO interface\n");

			atexit(ExitFunc); //Call exit function

			ver = Mcp2221_GetLibraryVersion(LibVer);  //Get DLL version
			if (ver == 0)
				printf("Library (DLL) version: %ls\n", LibVer);
			else
			{
				error = Mcp2221_GetLastError();
				printf("Version can't be found, version: %d, error: %d\n", ver, error);
			}
			//Get number of connected devices with this VID & PID
			Mcp2221_GetConnectedDevices(VID, PID, &NumOfDev);
			if (NumOfDev == 0)
			{
				printf("No MCP2221 devices connected\n");
				printf("Will not be able to control devices\n");
			}
			else {
				printf("Number of devices found: %d\n", NumOfDev);

				//Open first MCP2221 device discovered by index
				handle = Mcp2221_OpenByIndex(VID, PID, NumOfDev - 1);
				error = Mcp2221_GetLastError();
				if (error == NULL) {
					i2cpresent = true;
					printf("Connection successful\n");
				}
				else {
					printf("Error message is %s\n", error);
					i2cpresent = false;
					printf("DIO not running on this instance");
				}
				if (i2cpresent) {
					//Get manufacturer descriptor
					flag = Mcp2221_GetManufacturerDescriptor(handle, MfrDescriptor);
					if (flag == 0)
						printf("Manufacturer descriptor: %ls\n", MfrDescriptor);
					else
						printf("Error getting descriptor: %d\n", flag);

					//Get product descriptor
					flag = Mcp2221_GetProductDescriptor(handle, ProdDescrip);
					if (flag == 0)
						printf("Product descriptor: %ls\n", ProdDescrip);
					else
						printf("Error getting product descriptor: %d\n", flag);

					//Get power attributes
					flag = Mcp2221_GetUsbPowerAttributes(handle, &PowerAttrib, &ReqCurrent);
					if (flag == 0)
						printf("Power Attributes, %x\nRequested current units = %d\nRequested current(mA) = %d\n", PowerAttrib, ReqCurrent, ReqCurrent * 2);
					else
						printf("Error getting power attributes: %d\n", flag);


					printf("DIO: Set all outputs to OFF: \n");
					txdata[0] = 0x01;  // reg 0x01 is REG_OUT
					txdata[1] = 0xFF; // data
					error = Mcp2221_I2cWrite(handle, 2, 0x21, 0x01, txdata);
					if (error != 0)
						printf("DIO: Set outputs error. I2C error: %d\n", error);

					printf("DIO: Configuring outputs on all ports: \n");
					txdata[0] = 0x03;  // reg 0x03 is REG_DIRECTION 
					txdata[1] = 0x00;  // data 0x00 (0 = OUT, 1 = IN default)
					error = Mcp2221_I2cWrite(handle, 2, 0x21, 0x01, txdata);
					if (error != 0)
						printf("DIO: Configure outputs.  I2C error: %d\n", error);
				}
			}
		}


		/* END HPE */




        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validating of input arguments --------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }

		/* HPE ADD */

		const size_t mywidth = (size_t)cap.get(cv::CAP_PROP_FRAME_WIDTH);
		const size_t myheight = (size_t)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		::gwidth = mywidth;
		::gheight = myheight;
		printf("FRAME_WIDTH=%d\n", mywidth);
		printf("FRAME_HEIGHT=%d\n", myheight);

		// Setup vertical bars for where the face must be to do facial recognition
		::detect_left = 0; // (mywidth / 5) * 2;
		::detect_right = mywidth; // (mywidth / 5) * 3;

		// Set up scale of text based on width of image
		if (mywidth < 800) {
			::text_scale = 0.7;
		}
		if (mywidth >= 800) {
			::text_scale = 1.6;
		}

		/* END HPE */

        Timer timer;
        // read input (video) frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }

        const size_t width  = static_cast<size_t>(frame.cols);
        const size_t height = static_cast<size_t>(frame.rows);

        cv::VideoWriter videoWriter;
        if (!FLAGS_o.empty()) {
            videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('I', 'Y', 'U', 'V'), 25, cv::Size(width, height));
        }
        // ---------------------------------------------------------------------------------------------------
        // --------------------------- 1. Loading Inference Engine -----------------------------

        Core ie;

        std::set<std::string> loadedDevices;
        std::vector<std::pair<std::string, std::string>> cmdOptions = {
            {FLAGS_d, FLAGS_m},
            {FLAGS_d_ag, FLAGS_m_ag},
            {FLAGS_d_hp, FLAGS_m_hp},
            {FLAGS_d_em, FLAGS_m_em},
            {FLAGS_d_lm, FLAGS_m_lm}
        };
        FaceDetection faceDetector(FLAGS_m, FLAGS_d, 1, false, FLAGS_async, FLAGS_t, FLAGS_r,
                                   static_cast<float>(FLAGS_bb_enlarge_coef), static_cast<float>(FLAGS_dx_coef), static_cast<float>(FLAGS_dy_coef));
        AgeGenderDetection ageGenderDetector(FLAGS_m_ag, FLAGS_d_ag, FLAGS_n_ag, FLAGS_dyn_ag, FLAGS_async, FLAGS_r);
        HeadPoseDetection headPoseDetector(FLAGS_m_hp, FLAGS_d_hp, FLAGS_n_hp, FLAGS_dyn_hp, FLAGS_async, FLAGS_r);
        EmotionsDetection emotionsDetector(FLAGS_m_em, FLAGS_d_em, FLAGS_n_em, FLAGS_dyn_em, FLAGS_async, FLAGS_r);
        FacialLandmarksDetection facialLandmarksDetector(FLAGS_m_lm, FLAGS_d_lm, FLAGS_n_lm, FLAGS_dyn_lm, FLAGS_async, FLAGS_r);

        for (auto && option : cmdOptions) {
            auto deviceName = option.first;
            auto networkName = option.second;

            if (deviceName.empty() || networkName.empty()) {
                continue;
            }

            if (loadedDevices.find(deviceName) != loadedDevices.end()) {
                continue;
            }
            slog::info << "Loading device " << deviceName << slog::endl;
            std::cout << ie.GetVersions(deviceName) << std::endl;

            /** Loading extensions for the CPU device **/
            if ((deviceName.find("CPU") != std::string::npos)) {
                ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    ie.AddExtension(extension_ptr, "CPU");
                    slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
                }
            } else if (!FLAGS_c.empty()) {
                // Loading extensions for GPU
                ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
            }

            loadedDevices.insert(deviceName);
        }

        /** Per-layer metrics **/
        if (FLAGS_pc) {
            ie.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
        }
        // ---------------------------------------------------------------------------------------------------

        // --------------------------- 2. Reading IR models and loading them to plugins ----------------------
        // Disable dynamic batching for face detector as it processes one image at a time
        Load(faceDetector).into(ie, FLAGS_d, false);
        Load(ageGenderDetector).into(ie, FLAGS_d_ag, FLAGS_dyn_ag);
        Load(headPoseDetector).into(ie, FLAGS_d_hp, FLAGS_dyn_hp);
        Load(emotionsDetector).into(ie, FLAGS_d_em, FLAGS_dyn_em);
        Load(facialLandmarksDetector).into(ie, FLAGS_d_lm, FLAGS_dyn_lm);
        // ----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Doing inference -----------------------------------------------------
        // Starting inference & calculating performance
        slog::info << "Start inference " << slog::endl;

        bool isFaceAnalyticsEnabled = ageGenderDetector.enabled() || headPoseDetector.enabled() ||
                                      emotionsDetector.enabled() || facialLandmarksDetector.enabled();

        std::ostringstream out;
        size_t framesCounter = 0;
        bool frameReadStatus;
        bool isLastFrame;
        int delay = 1;
        double msrate = -1;
        cv::Mat prev_frame, next_frame;
        std::list<Face::Ptr> faces;
        size_t id = 0;

        if (FLAGS_fps > 0) {
            msrate = 1000.f / FLAGS_fps;
        }

        Visualizer::Ptr visualizer;
        if (!FLAGS_no_show || !FLAGS_o.empty()) {
            visualizer = std::make_shared<Visualizer>(cv::Size(width, height));
            if (!FLAGS_no_show_emotion_bar && emotionsDetector.enabled()) {
                visualizer->enableEmotionBar(emotionsDetector.emotionsVec);
            }
        }

		/* HPE ADD */

		//std::ostringstream demo_out;  // If we used this for the demo we should change name from out to demo_out or something different

		bool alreadyHappy = false;
		//bool candyOn = false;
		//bool candyGiven = false;
		//cv::Mat prev_frame, next_frame;

		auto happy_start = std::chrono::system_clock::now();
		auto happy_end = std::chrono::system_clock::now();
		auto happy_current = std::chrono::system_clock::now();

		auto candy_start = std::chrono::system_clock::now();
		auto candy_current = std::chrono::system_clock::now();
		std::chrono::duration<double> candy_elapsed;

		int mux = 1;
		int happycount = 0;
		int peoplecount = 0;

		std::cout << std::fixed << std::setprecision(3);

		/* END HPE */

        // Detecting all faces on the first frame and reading the next one
        faceDetector.enqueue(frame);
        faceDetector.submitRequest();

        prev_frame = frame.clone();

        // Reading the next frame
        frameReadStatus = cap.read(frame);

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press any key";
        }
        std::cout << std::endl;

        while (true) {
            timer.start("total");
            framesCounter++;
            isLastFrame = !frameReadStatus;

            // Retrieving face detection results for the previous frame
            faceDetector.wait();
            faceDetector.fetchResults();
            auto prev_detection_results = faceDetector.results;

            // No valid frame to infer if previous frame is the last
            if (!isLastFrame) {
                faceDetector.enqueue(frame);
                faceDetector.submitRequest();
            }

            // Filling inputs of face analytics networks
            for (auto &&face : prev_detection_results) {
                if (isFaceAnalyticsEnabled) {
                    auto clippedRect = face.location & cv::Rect(0, 0, width, height);
                    cv::Mat face = prev_frame(clippedRect);
                    ageGenderDetector.enqueue(face);
                    headPoseDetector.enqueue(face);
                    emotionsDetector.enqueue(face);
                    facialLandmarksDetector.enqueue(face);
                }
            }

            // Running Age/Gender Recognition, Head Pose Estimation, Emotions Recognition, and Facial Landmarks Estimation networks simultaneously
            if (isFaceAnalyticsEnabled) {
                ageGenderDetector.submitRequest();
                headPoseDetector.submitRequest();
                emotionsDetector.submitRequest();
                facialLandmarksDetector.submitRequest();
            }

            // Reading the next frame if the current one is not the last
            if (!isLastFrame) {
                frameReadStatus = cap.read(next_frame);
                if (FLAGS_loop_video && !frameReadStatus) {
                    if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
                        throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
                    }
                    frameReadStatus = cap.read(next_frame);
                }
            }

            if (isFaceAnalyticsEnabled) {
                ageGenderDetector.wait();
                headPoseDetector.wait();
                emotionsDetector.wait();
                facialLandmarksDetector.wait();
            }

            //  Postprocessing
            std::list<Face::Ptr> prev_faces;

            if (!FLAGS_no_smooth) {
                prev_faces.insert(prev_faces.begin(), faces.begin(), faces.end());
            }

            faces.clear();

			/* HPE ADD */
			int largest_face_area = 0;
			int current_face_area = 0;
			int biggest_face = -1;

			// Draw detection lines on frame
			cv::Mat tempframe;
			prev_frame.copyTo(tempframe);
			cv::line(tempframe, cv::Point(detect_left, 0), cv::Point(detect_left, height), lineColor, 2);
			cv::line(tempframe, cv::Point(detect_right, 0), cv::Point(detect_right, height), lineColor, 2);
			cv::addWeighted(tempframe, ::tempalpha, prev_frame, 1 - ::tempalpha, 0, prev_frame);

			//std::cout << "-------------------------" << std::endl;
			//std::cout << "NUMBER FACES DETECTED: " << prev_detection_results.size() << std::endl;
			// For every detected face, detect the largest face and only manage this one
			for (size_t i = 0; i < prev_detection_results.size(); i++) {
				auto& result = prev_detection_results[i];
				cv::Rect rect = result.location & cv::Rect(0, 0, width, height);

				Face::Ptr face;
				if (!FLAGS_no_smooth) {
					face = matchFace(rect, prev_faces);
					float intensity_mean = calcMean(prev_frame(rect));

					if ((face == nullptr) ||
						((face != nullptr) && ((std::abs(intensity_mean - face->_intensity_mean) / face->_intensity_mean) > 0.07f))) {
						face = std::make_shared<Face>(id++, rect);
					}
					else {
						prev_faces.remove(face);
					}

					face->_intensity_mean = intensity_mean;
					face->_location = rect;
				}
				else {
					face = std::make_shared<Face>(id++, rect);
				}

				/*  HPE ADD  */
				current_face_area = (result.location.width * result.location.height);
				if (current_face_area > largest_face_area) {
					largest_face_area = current_face_area;  // Support largest face...doesn't really work?
					

					int center_of_face = (result.location.x + (result.location.width / 2));

					if ((detect_left < center_of_face) && (center_of_face < detect_right)) {  // Must be in window
						if ((result.location.y > 0) && (result.location.y <= (::gheight - result.location.height)) && (result.location.x > 0) && (result.location.x <= (::gwidth - result.location.width))) {   // MIK MIK MIK
							// This roi workaround avoids clipping outside of the image
							cv::Rect roi = cv::Rect(result.location.x, result.location.y, result.location.width, result.location.height) & cv::Rect(0, 0, ::gwidth, ::gheight);
							//cv::Mat faceROI(prev_frame, cv::Rect(result.location.x, result.location.y, result.location.width, result.location.height));
							cv::Mat faceROI(prev_frame, roi);
							cv::Mat faceImage;
							faceROI.copyTo(faceImage);

							// Resize the face
							cv::resize(faceImage, faceImage, cv::Size(::face_width, ::face_height), cv::INTER_AREA);

							std::vector<uchar> buf_face;
							cv::imencode(".jpg", faceImage, buf_face);
							uchar* enc_msg_face = new uchar[buf_face.size()];
							for (int i = 0; i < buf_face.size(); i++) enc_msg_face[i] = buf_face[i];
							std::string encoded_frame_face = base64_encode(enc_msg_face, buf_face.size());

							std::cout << "FACE:" << encoded_frame_face << std::endl;  // Production

							biggest_face = i; 
							//std::cout << "CURRENT: " << current_face_area << "    LARGEST: " << largest_face_area << "IN WINDOW: TRUE" << std::endl;

						}
						else {
							//std::cout << "CURRENT: " << current_face_area << "    LARGEST: " << largest_face_area << "IN WINDOW: FALSE" << std::endl;
						}



					}

				}
			}

			//std::cout << "BIGGEST FACE INDEX: " << biggest_face << std::endl;
			//std::cout << "-------------------------" << std::endl;
            // For every detected face
            for (size_t i = 0; i < prev_detection_results.size(); i++) {

				auto& result = prev_detection_results[i];
				cv::Rect rect = result.location & cv::Rect(0, 0, width, height);

				Face::Ptr face;
				if (!FLAGS_no_smooth) {
					face = matchFace(rect, prev_faces);
					float intensity_mean = calcMean(prev_frame(rect));

					if ((face == nullptr) ||
						((face != nullptr) && ((std::abs(intensity_mean - face->_intensity_mean) / face->_intensity_mean) > 0.07f))) {
						face = std::make_shared<Face>(id++, rect);
					}
					else {
						prev_faces.remove(face);
					}

					face->_intensity_mean = intensity_mean;
					face->_location = rect;
				}
				else {
					face = std::make_shared<Face>(id++, rect);
				}


				if (i == biggest_face) {


					face->ageGenderEnable((ageGenderDetector.enabled() &&
						i < ageGenderDetector.maxBatch));
					if (face->isAgeGenderEnabled()) {
						AgeGenderDetection::Result ageGenderResult = ageGenderDetector[i];
						face->updateGender(ageGenderResult.maleProb);
						face->updateAge(ageGenderResult.age);
					}

					face->emotionsEnable((emotionsDetector.enabled() &&
						i < emotionsDetector.maxBatch));
					if (face->isEmotionsEnabled()) {
						face->updateEmotions(emotionsDetector[i]);
					}

					face->headPoseEnable((headPoseDetector.enabled() &&
						i < headPoseDetector.maxBatch));
					if (face->isHeadPoseEnabled()) {
						face->updateHeadPose(headPoseDetector[i]);
					}

					face->landmarksEnable((facialLandmarksDetector.enabled() &&
						i < facialLandmarksDetector.maxBatch));
					if (face->isLandmarksEnabled()) {
						face->updateLandmarks(facialLandmarksDetector[i]);
					}
					faces.push_back(face);
				}
				else {
					faces.remove(face);
				}
                //faces.push_back(face);
            }
			/* END HPE */

            //  Visualizing results
            if (!FLAGS_no_show || !FLAGS_o.empty()) {
                out.str("");
				/* HPE REMOVE
				out << "Total image throughput: " << std::fixed << std::setprecision(2)
					<< 1000.f / (timer["total"].getSmoothedDuration()) << " fps";
				cv::putText(prev_frame, out.str(), cv::Point2f(10, 45), cv::FONT_HERSHEY_TRIPLEX, 1.2,
							cv::Scalar(255, 0, 0), 2);
				 END HPE */

                // drawing faces
                visualizer->draw(prev_frame, faces);

				/*  HPE ADD  */

				// Resize the image
				cv::resize(prev_frame, prev_frame, cv::Size(::out_width, ::out_height), cv::INTER_AREA);

				std::vector<uchar> buf;
				cv::imencode(".jpg", prev_frame, buf);
				uchar* enc_msg = new uchar[buf.size()];
				for (int i = 0; i < buf.size(); i++) enc_msg[i] = buf[i];
				std::string encoded_frame = base64_encode(enc_msg, buf.size());

				std::cout << "FRAME:" << encoded_frame << std::endl;  // Production

				/*  END HPE */



				/* HPE REMOVE  
				if (!FLAGS_no_show) {
					cv::imshow("Detection results", prev_frame);
				}
				END HPE */ 
            }

			/* HPE ADD */
			// Handle all the candy timers

			if (::candyOn) {
				//std::cout << "CANDY!" << std::endl;
				//out4 << "!!CANDY!!";
				//::candyOn = true;
				if (!::candyGiven) {
					::candyGiven = true;
					if (i2cpresent) {
						candy_start = std::chrono::system_clock::now();
						printf("Sending I2C command to turn candy on\n");

						printf("DIO: Set all outputs to ON: \n");
						txdata[0] = 0x01;  // reg 0x01 is REG_OUT
						txdata[1] = 0x00; // data
						error = Mcp2221_I2cWrite(handle, 2, 0x21, 0x01, txdata);
						if (error != 0)
							printf("DIO: Set outputs error. I2C error: %d\n", error);

					}
				}
				// Handle candy dispenser turn off
				candy_current = std::chrono::system_clock::now();
				candy_elapsed = candy_current - candy_start;
				printf("CANDY_DURATION: %f    CANDY ELAPSED: %f\n", ::candy_duration, candy_elapsed.count());
				if (candy_elapsed.count() >= ::candy_duration) {
					// Stop the flow of candy
					printf("Sending I2C command to turn candy off\n");
					printf("DIO: Set all outputs to OFF: \n");
					txdata[0] = 0x01;  // reg 0x01 is REG_OUT
					txdata[1] = 0xFF; // data
					error = Mcp2221_I2cWrite(handle, 2, 0x21, 0x01, txdata);
					if (error != 0)
						printf("DIO: Set outputs error. I2C error: %d\n", error);
					::candyOn = false;
					::candyGiven = false;
				}
			}
			/* END HPE ADD */

            if (!FLAGS_o.empty()) {
                videoWriter.write(prev_frame);
            }

            prev_frame = frame;
            frame = next_frame;
            next_frame = cv::Mat();

            timer.finish("total");

            if (FLAGS_fps > 0) {
                delay = std::max(1, static_cast<int>(msrate - timer["total"].getLastCallDuration()));
            }

            // End of file (or a single frame file like an image). The last frame is displayed to let you check what is shown
            if (isLastFrame) {
                if (!FLAGS_no_wait) {
                    std::cout << "No more frames to process!" << std::endl;
                    cv::waitKey(0);
                }
                break;
            } else if (!FLAGS_no_show && -1 != cv::waitKey(delay)) {
                break;
            }
        }

        slog::info << "Number of processed frames: " << framesCounter << slog::endl;
        slog::info << "Total image throughput: " << framesCounter * (1000.f / timer["total"].getTotalDuration()) << " fps" << slog::endl;

        // Showing performance results
        if (FLAGS_pc) {
            faceDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d));
            ageGenderDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d_ag));
            headPoseDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d_hp));
            emotionsDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d_em));
            facialLandmarksDetector.printPerformanceCounts(getFullDeviceName(ie, FLAGS_d_lm));
        }
        // ---------------------------------------------------------------------------------------------------

        if (!FLAGS_o.empty()) {
            videoWriter.release();
        }

        // release input video stream
        cap.release();

        // close windows
        cv::destroyAllWindows();
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}