#include "ComponentBackgroundModel.h"
#define THISCLASS ComponentBackgroundModel

#include <opencv2/imgproc.hpp>
#include "DisplayEditor.h"

THISCLASS::ComponentBackgroundModel(SwisTrackCore * stc):
	Component(stc, wxT("BackgroundModel")),
	mMethod(wxT("MOG2")), mUseColor(false), mHistory(500), mThreshold(16), mDetectShadows(true),
	mLearningRate(-1), mFreezeAfterFrames(0), mOpeningSize(3),
	mSubtractor(), mFramesLearned(0),
	mDisplayOutput(wxT("Output"), wxT("Foreground")),
	mDisplayBackground(wxT("Background"), wxT("Background model")) {

	// Data structure relations
	mCategory = &(mCore->mCategoryProcessingGray);
	AddDataStructureRead(&(mCore->mDataStructureImageGray));
	AddDataStructureRead(&(mCore->mDataStructureImageColor));
	AddDataStructureWrite(&(mCore->mDataStructureImageBinary));
	AddDisplay(&mDisplayOutput);
	AddDisplay(&mDisplayBackground);

	// Read the XML configuration file
	Initialize();
}

THISCLASS::~ComponentBackgroundModel() {
}

void THISCLASS::CreateModel() {
	if (mMethod == wxT("KNN")) {
		mSubtractor = cv::createBackgroundSubtractorKNN(mHistory, mThreshold, mDetectShadows);
	} else {
		mSubtractor = cv::createBackgroundSubtractorMOG2(mHistory, mThreshold, mDetectShadows);
	}
	mFramesLearned = 0;
}

void THISCLASS::OnStart() {
	OnReloadConfiguration();
	CreateModel();
}

void THISCLASS::OnReloadConfiguration() {
	wxString method = GetConfigurationString(wxT("Method"), wxT("MOG2"));
	bool useColor = GetConfigurationBool(wxT("UseColor"), false);
	int history = std::max(1, GetConfigurationInt(wxT("History"), 500));
	double threshold = GetConfigurationDouble(wxT("Threshold"), method == wxT("KNN") ? 400 : 16);
	bool detectShadows = GetConfigurationBool(wxT("DetectShadows"), true);

	// Model parameters that require a new model
	bool recreate = method != mMethod || useColor != mUseColor || history != mHistory || threshold != mThreshold || detectShadows != mDetectShadows;
	mMethod = method;
	mUseColor = useColor;
	mHistory = history;
	mThreshold = threshold;
	mDetectShadows = detectShadows;

	mLearningRate = GetConfigurationDouble(wxT("LearningRate"), -1);
	mFreezeAfterFrames = std::max(0, GetConfigurationInt(wxT("FreezeAfterFrames"), 0));
	mOpeningSize = std::max(0, GetConfigurationInt(wxT("OpeningSize"), 3));

	if (recreate && mSubtractor) CreateModel();
}

void THISCLASS::OnStep() {
	cv::Mat inputImage = mUseColor ? mCore->mDataStructureImageColor.mImage : mCore->mDataStructureImageGray.mImage;
	if (inputImage.empty()) {
		AddError(mUseColor ? wxT("No color input image.") : wxT("No grayscale input image."));
		return;
	}
	if (! mSubtractor) CreateModel();

	// Learn (unless frozen) and segment
	bool frozen = mFreezeAfterFrames > 0 && mFramesLearned >= mFreezeAfterFrames;
	cv::Mat mask;
	mSubtractor->apply(inputImage, mask, frozen ? 0.0 : mLearningRate);
	if (! frozen) mFramesLearned++;

	// Shadows are marked with 127: treat them as background
	cv::threshold(mask, mask, 200, 255, cv::THRESH_BINARY);
	if (mOpeningSize > 1) {
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(mOpeningSize, mOpeningSize));
		cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
	}
	mCore->mDataStructureImageBinary.mImage = mask;

	// Displays
	DisplayEditor de(&mDisplayOutput);
	if (de.IsActive()) de.SetMainImage(mask);

	DisplayEditor deb(&mDisplayBackground);
	if (deb.IsActive()) {
		cv::Mat background;
		mSubtractor->getBackgroundImage(background);
		if (! background.empty()) deb.SetMainImage(background);
	}
}

void THISCLASS::OnStepCleanup() {
}

void THISCLASS::OnStop() {
}
