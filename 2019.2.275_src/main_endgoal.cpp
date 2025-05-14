// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/*

To run:
C:\Users\mikst\OneDrive\Desktop\demo>bin\websocketd.exe --port=8080  bin\candy_demo.exe -websocket

To Do:
1) Add counter for smiles (total people)
3) Log people who show up
5) Handle multiple people
6) Increase size of text on video so you can see easier in thumbnail
7) Fix scaling for different webcams?  Helps with testing in multiple places


*/

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


#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include "interactive_face_detection.hpp"
#include "detectors.hpp"

#include <ie_iextension.h>
#include <ext_list.hpp>

#include "deps/mcp2221_dll_um.h"

#pragma comment(lib, "deps/mcp2221_dll_um_x64.lib")  //Link MCP2221 library, applicable for visual studio only

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
auto lineColor = cv::Scalar(0, 255, 0);
double id_window_size = 0.20;
float detect_left = 100; //  -1;
float detect_right = 500; //  -1;
size_t gwidth = 0;
size_t gheight = 0;
double text_scale = 0.5;
bool statson = true;
auto statsColor = cv::Scalar(0, 255, 0);

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
			printf("SETTING ALPHA to %s\n", value);
			::tempalpha = std::stod(value, &sz);
		}
		if (key == "duration") {
			printf("SETTING DURATION to %s\n", value);
			::candy_duration = std::stod(value, &sz);
		}
		if (key == "linecolor") {
			printf("SETTING LINE COLOR to %s\n", value);
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
		}
		if (key == "statscolor") {
			printf("SETTING STATS COLOR to %s\n", value);
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
			if (value == "Off") {
				::statson = false;
			}
		}
		if (key == "windowsize") {
			printf("SETTING ID WINDOW SIZE to %s\n", value);
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
void *handle;

void ExitFunc()
{

	printf("Closing any MCP2221 connections\n");
	_sleep(10);
	//Mcp2221_Reset(handle);

	//Close all devices at exit
	Mcp2221_CloseAll();
}



using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
	// ---------------------------Parsing and validating input arguments--------------------------------------
	gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
	if (FLAGS_h) {
		showUsage();
		return false;
	}
	slog::info << "Parsing input parameters" << slog::endl;

	if (FLAGS_i.empty()) {
		throw std::logic_error("Parameter -i is not set");
	}
	if (FLAGS_d == "CPU") {
		printf("FDEVICE:CPU\n");
		printf("Overriding model information for CPU (FP32)\n");
		FLAGS_m = "C:/Intel/computer_vision_sdk_2018.5.456/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml";
		//FLAGS_m_em = "C:/Intel/computer_vision_sdk_2018.5.456/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml";
		//FLAGS_m_hp = "C:\Intel\computer_vision_sdk_2018.5.456\deployment_tools\intel_models\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml";
	}
	if (FLAGS_d == "MYRIAD") {
		printf("FDEVICE:VPU\n");
		printf("Overriding model information for non-CPU (FP16)\n");
		FLAGS_m = "C:/Intel/computer_vision_sdk_2018.5.456/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml";
		//FLAGS_m_em = "C:/Intel/computer_vision_sdk_2018.5.456/deployment_tools/intel_models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml";
		//FLAGS_m_hp = "C:\Intel\computer_vision_sdk_2018.5.456\deployment_tools\intel_models\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001.xml";
	}
	if (FLAGS_d_em == "CPU") {
		printf("EDEVICE:CPU\n");
		printf("Overriding model information for CPU (FP32)\n");
		//FLAGS_m = "C:/Intel/computer_vision_sdk_2018.5.456/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml";
		FLAGS_m_em = "C:/Intel/computer_vision_sdk_2018.5.456/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml";
		//FLAGS_m_hp = "C:\Intel\computer_vision_sdk_2018.5.456\deployment_tools\intel_models\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml";
	}
	if (FLAGS_d_em == "MYRIAD") {
		printf("EDEVICE:VPU\n");
		printf("Overriding model information for non-CPU (FP16)\n");
		//FLAGS_m = "C:/Intel/computer_vision_sdk_2018.5.456/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml";
		FLAGS_m_em = "C:/Intel/computer_vision_sdk_2018.5.456/deployment_tools/intel_models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml";
		//FLAGS_m_hp = "C:\Intel\computer_vision_sdk_2018.5.456\deployment_tools\intel_models\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001.xml";
	}

	//printf("Model input: %s\n", FLAGS_m.c_str());

	/*
	if (FLAGS_m.empty()) {
		throw std::logic_error("Parameter -m is not set");
	}
	*/
	printf("FLAGS: Candy duration is set to %d ms\n", FLAGS_candy_duration);

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

//void setLabel(cv::Mat& im, const std::string label, const cv::Point & or , double scale) // original
void setLabel(cv::Mat& im, const std::string label, const cv::Point & or , int justify) // updated
{
	// Justify, 0 = left, 1 = right

	int fontface = cv::FONT_HERSHEY_SIMPLEX;
	int baseline = 0;

	// Default values for 1080p stream...
	int thickness = 3;
	double scale = 1.6;

	if (::gwidth < 800) {
		// Override some values to fit better
		scale = 0.7;
		thickness = 1;
	}

	cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
	//cv::rectangle(im, or +cv::Point(0, baseline), or +cv::Point(text.width, -text.height), CV_RGB(0, 0, 0), cv::FILLED);
	if (justify == 1) { // right
		cv::putText(im, label, or +cv::Point(::gwidth - text.width, 0), fontface, scale, CV_RGB(255, 255, 0), thickness, 8);
	}
	else { // left and everything else?
		cv::putText(im, label, or , fontface, scale, CV_RGB(255, 255, 0), thickness, 8);
	}


	/*  // Example from the default Intel data...hard coded scale to 0.5, we use last argument value
	cv::putText(prev_frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5,
		cv::Scalar(255, 0, 0));
	*/
}

int main(int argc, char *argv[]) {

	unsigned char rxdata[2];
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

	try {
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


		std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

		// ------------------------------ Parsing and validating of input arguments --------------------------
		if (!ParseAndCheckCommandLine(argc, argv)) {
			return 0;
		}

		// Load in logo images
		//cv::Mat hpelogo = cv::imread("assets/hpe_logo.jpg"); // , cv::IMREAD_UNCHANGED);
		//cv::Mat intellogo = cv::imread("assets/intel_logo.png"); // , cv::IMREAD_UNCHANGED);


		slog::info << "Reading input" << slog::endl;
		cv::VideoCapture cap;
		const bool isCamera = FLAGS_i == "cam";
		if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
			throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
		}
		const size_t width = (size_t)cap.get(cv::CAP_PROP_FRAME_WIDTH);
		const size_t height = (size_t)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		::gwidth = width;
		::gheight = height;
		printf("FRAME_WIDTH=%d\n", width);
		printf("FRAME_HEIGHT=%d\n", height);

		// Setup vertical bars for where the face must be to do facial recognition
		::detect_left = (width / 5) * 2;
		::detect_right = (width / 5) * 3;

		// Set up scale of text based on width of image
		if (width < 800) {
			::text_scale = 0.7;
		}
		if (width >= 800) {
			::text_scale = 1.6;
		}

		// read input (video) frame
		cv::Mat frame;
		if (!cap.read(frame)) {
			throw std::logic_error("Failed to get frame from cv::VideoCapture");
		}
		// ---------------------------------------------------------------------------------------------------
		// --------------------------- 1. Loading plugin to the Inference Engine -----------------------------
		std::map<std::string, InferencePlugin> pluginsForDevices;
		std::vector<std::pair<std::string, std::string>> cmdOptions = {
			{FLAGS_d, FLAGS_m}, {FLAGS_d_ag, FLAGS_m_ag}, {FLAGS_d_hp, FLAGS_m_hp},
			{FLAGS_d_em, FLAGS_m_em}, {FLAGS_d_lm, FLAGS_m_lm}
		};
		FaceDetection faceDetector(FLAGS_m, FLAGS_d, 1, false, FLAGS_async, FLAGS_t, FLAGS_r);
		AgeGenderDetection ageGenderDetector(FLAGS_m_ag, FLAGS_d_ag, FLAGS_n_ag, FLAGS_dyn_ag, FLAGS_async);
		HeadPoseDetection headPoseDetector(FLAGS_m_hp, FLAGS_d_hp, FLAGS_n_hp, FLAGS_dyn_hp, FLAGS_async);
		EmotionsDetection emotionsDetector(FLAGS_m_em, FLAGS_d_em, FLAGS_n_em, FLAGS_dyn_em, FLAGS_async);
		FacialLandmarksDetection facialLandmarksDetector(FLAGS_m_lm, FLAGS_d_lm, FLAGS_n_lm, FLAGS_dyn_lm, FLAGS_async);

		for (auto && option : cmdOptions) {
			auto deviceName = option.first;
			auto networkName = option.second;

			if (deviceName == "" || networkName == "") {
				continue;
			}

			if (pluginsForDevices.find(deviceName) != pluginsForDevices.end()) {
				continue;
			}
			slog::info << "Loading plugin " << deviceName << slog::endl;
			InferencePlugin plugin = PluginDispatcher({ "../../../lib/intel64", "" }).getPluginByDevice(deviceName);

			/** Printing plugin version **/
			printPluginVersion(plugin, std::cout);

			/** Loading extensions for the CPU plugin **/
			if ((deviceName.find("CPU") != std::string::npos)) {
				plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

				if (!FLAGS_l.empty()) {
					// CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
					auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
					plugin.AddExtension(extension_ptr);
					slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
				}
			}
			else if (!FLAGS_c.empty()) {
				// Loading extensions for other plugins not CPU
				plugin.SetConfig({ {PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c} });
			}
			pluginsForDevices[deviceName] = plugin;
		}

		/** Per-layer metrics **/
		if (FLAGS_pc) {
			for (auto && plugin : pluginsForDevices) {
				plugin.second.SetConfig({ {PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES} });
			}
		}
		// ---------------------------------------------------------------------------------------------------

		// --------------------------- 2. Reading IR models and loading them to plugins ----------------------
		// Disable dynamic batching for face detector as it processes one image at a time
		Load(faceDetector).into(pluginsForDevices[FLAGS_d], false);
		Load(ageGenderDetector).into(pluginsForDevices[FLAGS_d_ag], FLAGS_dyn_ag);
		Load(headPoseDetector).into(pluginsForDevices[FLAGS_d_hp], FLAGS_dyn_hp);
		Load(emotionsDetector).into(pluginsForDevices[FLAGS_d_em], FLAGS_dyn_em);
		Load(facialLandmarksDetector).into(pluginsForDevices[FLAGS_d_lm], FLAGS_dyn_lm);
		// ----------------------------------------------------------------------------------------------------

		// --------------------------- 3. Doing inference -----------------------------------------------------
		// Starting inference & calculating performance
		slog::info << "Start inference " << slog::endl;
		if (!FLAGS_no_show) {
			std::cout << "Press any key to stop" << std::endl;
		}

		bool isFaceAnalyticsEnabled = ageGenderDetector.enabled() || headPoseDetector.enabled() ||
			emotionsDetector.enabled() || facialLandmarksDetector.enabled();

		Timer timer;
		timer.start("total");
		Timer emotiontimer;

		std::ostringstream out;
		//std::ostringstream out2;
		//std::ostringstream out3;
		//std::ostringstream out4;

		size_t framesCounter = 0;
		bool frameReadStatus;
		bool isLastFrame;
		bool alreadyHappy = false;
		//bool candyOn = false;
		//bool candyGiven = false;
		cv::Mat prev_frame, next_frame;

		auto happy_start = std::chrono::system_clock::now();
		auto happy_end = std::chrono::system_clock::now();
		auto happy_current = std::chrono::system_clock::now();
		std::chrono::duration<double> happy_elapsed;

		auto candy_start = std::chrono::system_clock::now();
		auto candy_current = std::chrono::system_clock::now();
		std::chrono::duration<double> candy_elapsed;

		float countdown;
		::candy_duration = (FLAGS_candy_duration / 1000);
		char szFileName[MAX_PATH];
		int mux = 1;
		int happycount = 0;
		int peoplecount = 0;

		std::cout << std::fixed << std::setprecision(3);

		// Detecting all faces on the first frame and reading the next one
		timer.start("detection");
		faceDetector.enqueue(frame);
		faceDetector.submitRequest();
		timer.finish("detection");

		prev_frame = frame.clone();

		// Reading the next frame
		timer.start("video frame decoding");
		frameReadStatus = cap.read(frame);
		timer.finish("video frame decoding");

		if (FLAGS_screen || FLAGS_fullscreen) {
			cv::namedWindow("Candy Demo", cv::WND_PROP_FULLSCREEN);
		}
		if (FLAGS_fullscreen) {
			cv::setWindowProperty("Candy Demo", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
		}

		while (true) {

			// The mutex causes this to not work well or take too long to get a lock so ignoring... YOLO
			//std::lock_guard<std::mutex> lock{ msg_mutex };
			if (msg.length() > 0) {
				std::cout << msg << std::endl;
				msg.clear();
			}

			framesCounter++;
			isLastFrame = !frameReadStatus;

			// Update file number for output "video"
			if (mux >= FLAGS_output_frames) {
				mux = 1;
			}
			else {
				++mux;
			}


			timer.start("detection");
			// Retrieving face detection results for the previous frame
			faceDetector.wait();
			faceDetector.fetchResults();
			auto prev_detection_results = faceDetector.results;

			// No valid frame to infer if previous frame is the last
			if (!isLastFrame) {
				faceDetector.enqueue(frame);
				faceDetector.submitRequest();
			}
			timer.finish("detection");

			timer.start("data preprocessing");
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
			timer.finish("data preprocessing");

			// Running Age/Gender Recognition, Head Pose Estimation, Emotions Recognition, and Facial Landmarks Estimation networks simultaneously
			timer.start("face analytics call");
			if (isFaceAnalyticsEnabled) {
				ageGenderDetector.submitRequest();
				headPoseDetector.submitRequest();
				emotionsDetector.submitRequest();
				facialLandmarksDetector.submitRequest();
			}
			timer.finish("face analytics call");

			// Reading the next frame if the current one is not the last
			if (!isLastFrame) {
				timer.start("video frame decoding");
				frameReadStatus = cap.read(next_frame);
				timer.finish("video frame decoding");
			}

			timer.start("face analytics wait");
			if (isFaceAnalyticsEnabled) {
				ageGenderDetector.wait();
				headPoseDetector.wait();
				emotionsDetector.wait();
				facialLandmarksDetector.wait();
			}
			timer.finish("face analytics wait");


			if ((char)cv::waitKey(1) == 0x6f) {
				std::cout << "Manual override for DIO off" << std::endl;
				printf("DIO: Set all outputs to OFF: \n");
				txdata[0] = 0x01;  // reg 0x01 is REG_OUT
				txdata[1] = 0xFF; // data
				error = Mcp2221_I2cWrite(handle, 2, 0x21, 0x01, txdata);
				if (error != 0)
					printf("DIO: Set outputs error. I2C error: %d\n", error);
			}
			/*
			else if ((char)cv::waitKey(1) == 0x6e) {
				std::cout << "Manual override for DIO on" << std::endl;
				printf("DIO: Set all outputs to ON: \n");
				txdata[0] = 0x01;  // reg 0x01 is REG_OUT
				txdata[1] = 0x00; // data
				error = Mcp2221_I2cWrite(handle, 2, 0x21, 0x01, txdata);
				if (error != 0)
					printf("DIO: Set outputs error. I2C error: %d\n", error);
			}*/

			// Visualizing results
			if (!FLAGS_no_show) {
				timer.start("visualization");
				out.str("");
				out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
					<< (timer["video frame decoding"].getSmoothedDuration() +
						timer["visualization"].getSmoothedDuration())
					<< " ms";
				if (::statson) {
					cv::putText(prev_frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, ::text_scale, // 0.5,
						statsColor);
				}

				out.str("");
				out << "Face detection time: " << std::fixed << std::setprecision(2)
					<< timer["detection"].getSmoothedDuration()
					<< " ms ("
					<< 1000.f / (timer["detection"].getSmoothedDuration())
					<< " fps)";
				if (::statson) {
					cv::putText(prev_frame, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_TRIPLEX, ::text_scale, //0.5,
						statsColor);
				}
				if (isFaceAnalyticsEnabled) {
					out.str("");
					out << "Face analytics "
						<< "time: " << std::fixed << std::setprecision(2)
						<< timer["face analytics call"].getSmoothedDuration() +
						timer["face analytics wait"].getSmoothedDuration()
						<< " ms ";
					if (!prev_detection_results.empty()) {
						out << "("
							<< 1000.f / (timer["face analytics call"].getSmoothedDuration() +
								timer["face analytics wait"].getSmoothedDuration())
							<< " fps)";
					}
					if (::statson) {
						cv::putText(prev_frame, out.str(), cv::Point2f(0, 65), cv::FONT_HERSHEY_TRIPLEX, ::text_scale, //0.5,
							statsColor);
					}
				}

				// For every detected face
				int i = 0;
				int largest_face_area = 0;
				int current_face_area = 0;

				for (auto &result : prev_detection_results) {
					cv::Rect rect = result.location;
					current_face_area = (result.location.width * result.location.height);
					if (current_face_area > largest_face_area) {
						largest_face_area = current_face_area;  // Support largest face...doesn't really work?

						int center_of_face = (result.location.x + (result.location.width / 2));

						if ((detect_left < center_of_face) && (center_of_face < detect_right)) {  // Must be in window

							// Print the bounding box for the face
						
							//printf("DEBUG_FACE:%d,%d,%d,%d\n", result.location.x, result.location.y, result.location.width, result.location.height);
							//Sleep(20);

							// Output the face as separate image
							if (FLAGS_websocket) {
								// Extract just the face
								// Do bounds checking...
								//if ((result.location.y > 0) && (result.location.y <= (::gheight - result.location.height))) {   // MIK MIK MIK
								if ((result.location.y > 0) && (result.location.y <= (::gheight - result.location.height)) && (result.location.x > 0) && (result.location.x <= (::gwidth - result.location.width))) {   // MIK MIK MIK
									cv::Mat faceROI(prev_frame, cv::Rect(result.location.x, result.location.y, result.location.width, result.location.height));
									cv::Mat faceImage;
									faceROI.copyTo(faceImage);

									// Resize the face
									cv::resize(faceImage, faceImage, cv::Size(::face_width, ::face_height), cv::INTER_AREA);

									std::vector<uchar> buf_face;
									cv::imencode(".jpg", faceImage, buf_face);
									uchar *enc_msg_face = new uchar[buf_face.size()];
									for (int i = 0; i < buf_face.size(); i++) enc_msg_face[i] = buf_face[i];
									std::string encoded_frame_face = base64_encode(enc_msg_face, buf_face.size());

									//std::string encoded_frame = base64_encode(prev_frame.data, prev_frame.cols * prev_frame.rows);
									// HPE logo example works fine with html...something is wrong with the output of the file I am sending...
									//std::string encoded_frame = "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";

									std::cout << "FACE:" << encoded_frame_face << std::endl;
									//printf("DIDN'T FAIL\n");
								}
								/*
								else {
									printf("CLIPPING DETECTED\n");
								}
								*/
							}


							out.str("");

							if (ageGenderDetector.enabled() && i < ageGenderDetector.maxBatch) {
								out << (ageGenderDetector[i].maleProb > 0.5 ? "M" : "F");
								out << std::fixed << std::setprecision(0) << "," << ageGenderDetector[i].age;
								if (FLAGS_r) {
									std::cout << "Predicted gender, age = " << out.str() << std::endl;
								}
							}
							else {
								out << (result.label < faceDetector.labels.size() ? faceDetector.labels[result.label] :
									std::string("label #") + std::to_string(result.label))
									<< ": " << std::fixed << std::setprecision(3) << result.confidence;
							}

							if (emotionsDetector.enabled() && i < emotionsDetector.maxBatch) {
								std::string emotion = emotionsDetector[i];
								if (FLAGS_r) {
									std::cout << "Predicted emotion = " << emotion << std::endl;
								}

								out << "," << emotion;
								//out2.str("");
								//out3.str("");
								//out4.str("");

								//out2 << "Current emotion: " << emotion;
								printf("EMOTION:%s\n", emotion);

								

								/*
								if ((emotion == "happy") && alreadyHappy) {
									happy_current = std::chrono::system_clock::now();
									happy_elapsed = happy_current - happy_start;
									//std::cout << "Happy for " << happy_elapsed.count() << std::endl;
									countdown = 3 - happy_elapsed.count();
									if (countdown <= 0) {
										countdown = 0.000;
										if (!candyGiven) {
											//std::cout << "CANDY!" << std::endl;
											//out4 << "!!CANDY!!";
											candyOn = true;
											candyGiven = true;
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
									}

									//out3 << "Happy Countdown: " << roundf(countdown * 1000) / 1000 << "s";
								}
								else if ((emotion == "happy") && !alreadyHappy) {
									alreadyHappy = true;
									happy_start = std::chrono::system_clock::now();
									//std::cout << "Started happy timer" << std::endl;
								}
								else if (emotion != "happy") {
									//std::cout << "Stop the happy timer" << std::endl;
									//emotiontimer.finish("happy");


									// Need to log this person if they were a happy person
									if (alreadyHappy) {
										happy_current = std::chrono::system_clock::now();
										happy_elapsed = happy_current - happy_start;
										//std::cout << "Happy for " << happy_elapsed.count() << std::endl;
										countdown = 3 - happy_elapsed.count();
										// Only log them if they met the criteria
										//if (countdown <= 0) {
											//sprintf(szFileName, "people/happy_%03d.png", happycount);
											//cv::imwrite(szFileName, prev_frame);
											//++happycount;
											// Log how long they were happy

										//}

									}
									// Clean up

									alreadyHappy = false;
									candyGiven = false;
									happy_end = std::chrono::system_clock::now();
								}
								*/
							}

							// Original
							//setLabel(prev_frame, out2.str(), cv::Point2f(0, height - 20), ::text_scale);  // Emotion
							//setLabel(prev_frame, out3.str(), cv::Point2f(width - 700, height - 20), ::text_scale);  // Happy count down
							// Updated
							//setLabel(prev_frame, out2.str(), cv::Point2f(0, height - 20), 0);  // Left justified Emotion
							//setLabel(prev_frame, out3.str(), cv::Point2f(0, height - 20), 1);  // Right justified Happy count down

							//setLabel(prev_frame, out4.str(), cv::Point2f((width / 2) - 100, (height / 2) - 100), 0.8);  // CANDY!

							//out2.str("");
							//out3.str("");
							//out4.str("");
							/*
							if (headPoseDetector.enabled() && i < headPoseDetector.maxBatch) {
								if (FLAGS_r) {
									std::cout << "Head pose results: yaw, pitch, roll = "
										<< headPoseDetector[i].angle_y << ";"
										<< headPoseDetector[i].angle_p << ";"
										<< headPoseDetector[i].angle_r << std::endl;
								}
								cv::Point3f center(rect.x + rect.width / 2, rect.y + rect.height / 2, 0);
								headPoseDetector.drawAxes(prev_frame, center, headPoseDetector[i], 50);
							}

							if (facialLandmarksDetector.enabled() && i < facialLandmarksDetector.maxBatch) {
								auto normed_landmarks = facialLandmarksDetector[i];
								auto n_lm = normed_landmarks.size();
								if (FLAGS_r)
									std::cout << "Normed Facial Landmarks coordinates (x, y):" << std::endl;
								for (auto i_lm = 0UL; i_lm < n_lm / 2; ++i_lm) {
									float normed_x = normed_landmarks[2 * i_lm];
									float normed_y = normed_landmarks[2 * i_lm + 1];

									if (FLAGS_r) {
										std::cout << normed_x << ", "
											<< normed_y << std::endl;
									}
									int x_lm = rect.x + rect.width * normed_x;
									int y_lm = rect.y + rect.height * normed_y;
									// Drawing facial landmarks on the frame
									cv::circle(prev_frame, cv::Point(x_lm, y_lm), 1 + static_cast<int>(0.012 * rect.width), cv::Scalar(0, 255, 255), -1);
								}
							}
							*/
							/*
							auto genderColor = (ageGenderDetector.enabled() && (i < ageGenderDetector.maxBatch)) ?
								((ageGenderDetector[i].maleProb < 0.5) ? cv::Scalar(147, 20, 255) : cv::Scalar(255, 0,
									0))
								: cv::Scalar(255, 0, 0);
							*/
							// Color for face we are tracking and draw rectangle
							auto genderColor = cv::Scalar(0, 255, 0);
							cv::rectangle(prev_frame, result.location, genderColor, 1);


						}
					}
					else {
						// Color for face we are NOT tracking and draw rectangle
						auto genderColor = cv::Scalar(100, 100, 100);
						cv::rectangle(prev_frame, result.location, genderColor, 1);
					}
					// Draw rectangle around face
					//cv::rectangle(prev_frame, result.location, genderColor, 1);






					i++;

				}

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

				// Draw detect lines
				cv::Mat tempframe;
				//double tempalpha = 0.05;
				prev_frame.copyTo(tempframe);
				cv::line(tempframe, cv::Point(detect_left, 0), cv::Point(detect_left, height), lineColor, 2);
				cv::line(tempframe, cv::Point(detect_right, 0), cv::Point(detect_right, height), lineColor, 2);
				cv::addWeighted(tempframe, ::tempalpha, prev_frame, 1 - ::tempalpha, 0, prev_frame);

				// Overlay other information before displaying
				//hpelogo.copyTo(prev_frame(cv::Rect(width - hpelogo.cols, 0, hpelogo.cols, hpelogo.rows)));

				// For file output
				if (FLAGS_img) {
					sprintf(szFileName, "output/output%03d.png", mux);
					cv::imwrite(szFileName, prev_frame);
					//std::string encoded_frame = base64_encode(prev_frame.data, prev_frame.cols * prev_frame.rows);
					//MIK
				}

				// For websocket output
				if (FLAGS_websocket) {

					// Resize the image
					cv::resize(prev_frame, prev_frame, cv::Size (::out_width, ::out_height), cv::INTER_AREA);

					std::vector<uchar> buf;
					cv::imencode(".jpg", prev_frame, buf);
					uchar *enc_msg = new uchar[buf.size()];
					for (int i = 0; i < buf.size(); i++) enc_msg[i] = buf[i];
					std::string encoded_frame = base64_encode(enc_msg, buf.size());

					//std::string encoded_frame = base64_encode(prev_frame.data, prev_frame.cols * prev_frame.rows);
					// HPE logo example works fine with html...something is wrong with the output of the file I am sending...
					//std::string encoded_frame = "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";

					std::cout << "FRAME:" << encoded_frame << std::endl;
				}

				// For on screen viewing
				if (FLAGS_screen || FLAGS_fullscreen) {
					cv::imshow("Candy Demo", prev_frame);
				}

				// For FFMPEG output (needs work)
				if (FLAGS_ffmpeg) {
					std::vector<uchar> buff;
					cv::imencode(".jpg", prev_frame, buff);
					for (auto i = buff.begin(); i != buff.end(); ++i) {
						std::cout << *i;
					}
				}


				timer.finish("visualization");
			}
			else if (FLAGS_r) {
				// For every detected face
				for (int i = 0; i < prev_detection_results.size(); i++) {
					if (ageGenderDetector.enabled() && i < ageGenderDetector.maxBatch) {
						out.str("");
						out << (ageGenderDetector[i].maleProb > 0.5 ? "M" : "F");
						out << std::fixed << std::setprecision(0) << "," << ageGenderDetector[i].age;
						std::cout << "Predicted gender, age = " << out.str() << std::endl;
					}

					if (emotionsDetector.enabled() && i < emotionsDetector.maxBatch) {
						std::cout << "Predicted emotion = " << emotionsDetector[i] << std::endl;
					}

					if (headPoseDetector.enabled() && i < headPoseDetector.maxBatch) {
						std::cout << "Head pose results: yaw, pitch, roll = "
							<< headPoseDetector[i].angle_y << ";"
							<< headPoseDetector[i].angle_p << ";"
							<< headPoseDetector[i].angle_r << std::endl;
					}

					if (facialLandmarksDetector.enabled() && i < facialLandmarksDetector.maxBatch) {
						auto normed_landmarks = facialLandmarksDetector[i];
						auto n_lm = normed_landmarks.size();
						std::cout << "Normed Facial Landmarks coordinates (x, y):" << std::endl;
						for (auto i_lm = 0UL; i_lm < n_lm / 2; ++i_lm) {
							float normed_x = normed_landmarks[2 * i_lm];
							float normed_y = normed_landmarks[2 * i_lm + 1];
							std::cout << normed_x << ", "
								<< normed_y << std::endl;
						}
					}
				}
			}

			// End of file (or a single frame file like an image). The last frame is displayed to let you check what is shown
			if (isLastFrame) {
				timer.finish("total");
				if (!FLAGS_no_wait) {
					std::cout << "No more frames to process. Press any key to exit" << std::endl;
					cv::waitKey(0);
					printf("crap2\n");
				}
				printf("here\n");
				break;
			}
			else if (!FLAGS_no_show && -1 != cv::waitKey(1)) {
				printf("crap1\n");
				timer.finish("total");
				break;
			}

			prev_frame = frame;
			frame = next_frame;
			next_frame = cv::Mat();
		}

		slog::info << "Number of processed frames: " << framesCounter << slog::endl;
		slog::info << "Total image throughput: " << framesCounter * (1000.f / timer["total"].getTotalDuration()) << " fps" << slog::endl;

		// Showing performance results
		if (FLAGS_pc) {
			faceDetector.printPerformanceCounts();
			ageGenderDetector.printPerformanceCounts();
			headPoseDetector.printPerformanceCounts();
			emotionsDetector.printPerformanceCounts();
			facialLandmarksDetector.printPerformanceCounts();
		}
		// ---------------------------------------------------------------------------------------------------
	}
	catch (const std::exception& error) {
		slog::err << error.what() << slog::endl;
		return 1;
	}
	catch (...) {
		slog::err << "Unknown/internal exception happened." << slog::endl;
		return 1;
	}

	printf("DIO: Closing - Set all outputs to OFF: \n");
	txdata[0] = 0x01;  // reg 0x01 is REG_OUT
	txdata[1] = 0xFF; // data
	error = Mcp2221_I2cWrite(handle, 2, 0x21, 0x01, txdata);
	if (error != 0)
		printf("DIO: Set outputs error. I2C error: %d\n", error);


	slog::info << "Execution successful" << slog::endl;
	return 0;
}

