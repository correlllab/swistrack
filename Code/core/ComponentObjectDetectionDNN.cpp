#include "ComponentObjectDetectionDNN.h"
#define THISCLASS ComponentObjectDetectionDNN

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <wx/filename.h>
#include "DisplayEditor.h"

static wxString ToWx(const std::string & s) {
	return wxString(s.c_str(), *wxConvCurrent);
}

THISCLASS::ComponentObjectDetectionDNN(SwisTrackCore * stc):
	Component(stc, wxT("ObjectDetectionDNN")),
	mDetector(), mThresholds(), mDetections(), mParticles(), mClassNames(), mDisplayScoreThreshold(0.5f),
	mLoadedModelKey(), mLoadWarnings(),
	mDisplayOutput(wxT("Output"), wxT("Object detections")) {

	// Data structure relations
	mCategory = &(mCore->mCategoryParticleDetection);
	AddDataStructureRead(&(mCore->mDataStructureImageColor));
	AddDataStructureWrite(&(mCore->mDataStructureParticles));
	AddDisplay(&mDisplayOutput);

	// Read the XML configuration file
	Initialize();
}

THISCLASS::~ComponentObjectDetectionDNN() {
}

void THISCLASS::OnStart() {
	OnReloadConfiguration();
}

void THISCLASS::OnReloadConfiguration() {
	if (! ObjectDetectorDNN::IsAvailable()) {
		AddError(wxT("This version of SwisTrack was built with an OpenCV library without the dnn module."));
		return;
	}

	// Thresholds (cheap, applied to the next frame)
	mThresholds.minScore = (float)std::min(std::max(GetConfigurationDouble(wxT("MinScore"), 0.1), 0.0), 1.0);
	mThresholds.nmsIoU = (float)std::min(std::max(GetConfigurationDouble(wxT("NMSThreshold"), 0.45), 0.0), 1.0);
	mThresholds.classAware = GetConfigurationBool(wxT("ClassAwareNMS"), true);
	mThresholds.maxDetections = std::max(GetConfigurationInt(wxT("MaxDetections"), 300), 0);
	mDisplayScoreThreshold = (float)GetConfigurationDouble(wxT("DisplayScoreThreshold"), 0.5);
	wxString classes = GetConfigurationString(wxT("Classes"), wxT(""));
	if (! ObjectDetectorDNN::ParseClassList(classes.ToStdString(), mThresholds.classes)) {
		AddError(wxT("Invalid class list '") + classes + wxT("': enter class numbers separated by commas, e.g. 0, 2."));
		return;
	}

	// Class names
	mClassNames.clear();
	wxString classNamesFile = GetConfigurationString(wxT("ClassNamesFile"), wxT(""));
	if (! classNamesFile.IsEmpty()) {
		wxFileName filename = mCore->GetProjectFileName(classNamesFile);
		if (! ObjectDetectorDNN::ReadClassNames(filename.GetFullPath().ToStdString(), mClassNames))
			AddWarning(wxT("Cannot read the class names file '") + filename.GetFullPath() + wxT("'."));
	}

	// Model configuration
	wxString modelFile = GetConfigurationString(wxT("ModelFile"), wxT(""));
	if (modelFile.IsEmpty()) {
		mDetector.Unload();
		mLoadedModelKey.Clear();
		AddError(wxT("No model file selected."));
		return;
	}
	wxFileName modelFileName = mCore->GetProjectFileName(modelFile);
	if (! modelFileName.IsOk()) {
		mDetector.Unload();
		mLoadedModelKey.Clear();
		AddError(wxT("Invalid model file name. To use a relative path, save the configuration first."));
		return;
	}
	if (! modelFileName.FileExists()) {
		mDetector.Unload();
		mLoadedModelKey.Clear();
		AddError(wxT("Model file '") + modelFileName.GetFullPath() + wxT("' not found."));
		return;
	}

	ObjectDetectorDNN::Config config;
	config.modelPath = modelFileName.GetFullPath().ToStdString();
	config.inputSize = cv::Size(GetConfigurationInt(wxT("InputWidth"), 640), GetConfigurationInt(wxT("InputHeight"), 640));

	wxString format = GetConfigurationString(wxT("Format"), wxT("auto"));
	config.format = ObjectDetectorDNN::FormatAuto;
	if (format == wxT("yolov8")) config.format = ObjectDetectorDNN::FormatYOLOv8;
	else if (format == wxT("yolov5")) config.format = ObjectDetectorDNN::FormatYOLOv5;
	else if (format == wxT("yolox")) config.format = ObjectDetectorDNN::FormatYOLOX;

	wxString colorOrder = GetConfigurationString(wxT("ColorOrder"), wxT("auto"));
	config.colorOrder = ObjectDetectorDNN::ColorAuto;
	if (colorOrder == wxT("rgb")) config.colorOrder = ObjectDetectorDNN::ColorRGB;
	else if (colorOrder == wxT("bgr")) config.colorOrder = ObjectDetectorDNN::ColorBGR;

	wxString pixelRange = GetConfigurationString(wxT("PixelRange"), wxT("auto"));
	config.pixelRange = ObjectDetectorDNN::RangeAuto;
	if (pixelRange == wxT("unit")) config.pixelRange = ObjectDetectorDNN::RangeUnit;
	else if (pixelRange == wxT("byte")) config.pixelRange = ObjectDetectorDNN::RangeByte;

	wxString backend = GetConfigurationString(wxT("Backend"), wxT("cpu"));
	config.backend = ObjectDetectorDNN::BackendCPU;
	if (backend == wxT("opencl")) config.backend = ObjectDetectorDNN::BackendOpenCL;
	else if (backend == wxT("opencl_fp16")) config.backend = ObjectDetectorDNN::BackendOpenCLFP16;
	else if (backend == wxT("cuda")) config.backend = ObjectDetectorDNN::BackendCUDA;
	else if (backend == wxT("cuda_fp16")) config.backend = ObjectDetectorDNN::BackendCUDAFP16;

	config.numClassesHint = (int)mClassNames.size();

	// (Re)load the model only if something relevant changed (including the model file itself)
	wxString modified = modelFileName.GetModificationTime().Format();
	wxString key = wxString::Format(wxT("%s|%s|%d|%d|%d|%d|%d|%d|%d"), modelFileName.GetFullPath(), modified,
		config.inputSize.width, config.inputSize.height, (int)config.format, (int)config.colorOrder, (int)config.pixelRange, (int)config.backend, config.numClassesHint);
	if (! mDetector.IsLoaded() || key != mLoadedModelKey) {
		mLoadedModelKey.Clear();
		mLoadWarnings.clear();
		std::string error;
		std::vector<std::string> warnings;
		bool ok = mDetector.Load(config, error, warnings);
		for (const std::string & warning : warnings) mLoadWarnings.push_back(ToWx(warning));
		if (! ok) {
			for (const wxString & warning : mLoadWarnings) AddWarning(warning);
			AddError(ToWx(error));
			return;
		}
		mLoadedModelKey = key;
	}

	for (const wxString & warning : mLoadWarnings) AddWarning(warning);
	AddInfo(wxString::Format(wxT("Model: %s, %d classes, %dx%d input, %s. First inference: %.1f ms."),
		ToWx(ObjectDetectorDNN::FormatName(mDetector.GetFormat())), mDetector.GetNumClasses(),
		config.inputSize.width, config.inputSize.height, ToWx(mDetector.GetBackendDescription()), mDetector.GetWarmupMs()));
}

void THISCLASS::OnStep() {
	// Get and check the input image (color image if available, otherwise the raw input)
	cv::Mat inputImage = mCore->mDataStructureImageColor.mImage;
	if (inputImage.empty()) inputImage = mCore->mDataStructureInput.mImage;
	if (inputImage.empty()) {
		AddError(wxT("No input image."));
		return;
	}
	if (! mDetector.IsLoaded()) {
		AddError(wxT("No model loaded."));
		return;
	}

	// Detect
	std::string error;
	if (! mDetector.Detect(inputImage, mThresholds, mDetections, error)) {
		AddError(ToWx(error));
		return;
	}
	mCore->mEventRecorder->Add(SwisTrackCoreEventRecorder::sType_StepLapTime, this);

	// One particle per detection
	mParticles.clear();
	mParticles.reserve(mDetections.size());
	for (const ObjectDetectorDNN::Detection & d : mDetections) {
		Particle particle;
		particle.mCenter = cv::Point2f(d.box.x + d.box.width / 2, d.box.y + d.box.height / 2);
		particle.mBoundingBox = d.box;
		particle.mArea = d.box.area();
		particle.mScore = d.score;
		particle.mOrientation = 0;
		particle.mFrameNumber = mCore->mDataStructureInput.mFrameNumber;
		particle.mFrameTimestamp = mCore->mDataStructureInput.mFrameTimestamp;
		mParticles.push_back(particle);
	}
	mCore->mDataStructureParticles.mParticles = &mParticles;

	// Let the Display know about our image
	DisplayEditor de(&mDisplayOutput);
	if (! de.IsActive()) return;
	de.SetParticles(&mParticles);

	cv::Mat outputImage;
	if (inputImage.channels() == 1) cv::cvtColor(inputImage, outputImage, cv::COLOR_GRAY2BGR);
	else if (inputImage.channels() == 4) cv::cvtColor(inputImage, outputImage, cv::COLOR_BGRA2BGR);
	else outputImage = inputImage.clone();
	DrawDetections(outputImage);
	de.SetMainImage(outputImage);
}

void THISCLASS::DrawDetections(cv::Mat & image) const {
	const int font = cv::FONT_HERSHEY_SIMPLEX;
	const double fontScale = 0.5;

	// Weakest first, so that strong detections end up on top
	for (auto it = mDetections.rbegin(); it != mDetections.rend(); ++it) {
		const ObjectDetectorDNN::Detection & d = *it;
		bool strong = (d.score >= mDisplayScoreThreshold);
		cv::Scalar color = strong ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
		cv::Rect box(cvRound(d.box.x), cvRound(d.box.y), cvRound(d.box.width), cvRound(d.box.height));
		cv::rectangle(image, box, color, strong ? 2 : 1);

		std::string label;
		if (d.classId >= 0 && d.classId < (int)mClassNames.size() && ! mClassNames[d.classId].empty()) label = mClassNames[d.classId];
		else label = "#" + std::to_string(d.classId);
		char score[16];
		std::snprintf(score, sizeof(score), " %.2f", d.score);
		label += score;

		int baseline = 0;
		cv::Size textSize = cv::getTextSize(label, font, fontScale, 1, &baseline);
		cv::Point origin(box.x, std::max(box.y, textSize.height + 2));
		cv::rectangle(image, cv::Rect(origin.x, origin.y - textSize.height - 2, textSize.width + 2, textSize.height + baseline + 2), color, cv::FILLED);
		cv::putText(image, label, cv::Point(origin.x + 1, origin.y - 1), font, fontScale, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
	}
}

void THISCLASS::OnStepCleanup() {
	mCore->mDataStructureParticles.mParticles = NULL;
}

void THISCLASS::OnStop() {
}
