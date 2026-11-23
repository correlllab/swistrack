#include "ComponentBlobDetectionSplitting.h"
#define THISCLASS ComponentBlobDetectionSplitting

#include <opencv2/imgproc.hpp>
#include "DisplayEditor.h"

THISCLASS::ComponentBlobDetectionSplitting(SwisTrackCore * stc):
	Component(stc, wxT("BlobDetectionSplitting")),
	mSplitter(), mParticles(),
	mDisplayOutput(wxT("Output"), wxT("Blobs")) {

	// Data structure relations
	mCategory = &(mCore->mCategoryParticleDetection);
	AddDataStructureRead(&(mCore->mDataStructureImageBinary));
	AddDataStructureWrite(&(mCore->mDataStructureParticles));
	AddDisplay(&mDisplayOutput);

	// Read the XML configuration file
	Initialize();
}

THISCLASS::~ComponentBlobDetectionSplitting() {
}

void THISCLASS::OnStart() {
	mSplitter.Reset();
	OnReloadConfiguration();
}

void THISCLASS::OnReloadConfiguration() {
	BlobSplitter::Parameters p;
	p.minArea = GetConfigurationInt(wxT("MinArea"), p.minArea);
	p.maxArea = GetConfigurationInt(wxT("MaxArea"), p.maxArea);
	p.objectArea = GetConfigurationDouble(wxT("ObjectArea"), p.objectArea);
	p.splitRatio = GetConfigurationDouble(wxT("SplitRatio"), p.splitRatio);
	p.maxObjectsPerBlob = std::max(1, GetConfigurationInt(wxT("MaxObjectsPerBlob"), p.maxObjectsPerBlob));
	p.maxNumber = std::max(0, GetConfigurationInt(wxT("MaxNumber"), p.maxNumber));
	p.splitScore = (float)GetConfigurationDouble(wxT("SplitScore"), p.splitScore);
	if (p.splitRatio < 1.0) AddWarning(wxT("A split ratio below 1 splits single objects."));
	if (p.maxArea > 0 && p.minArea > p.maxArea) AddError(wxT("The area selection is empty."));
	mSplitter.SetParameters(p);
}

void THISCLASS::OnStep() {
	cv::Mat inputImage = mCore->mDataStructureImageBinary.mImage;
	if (inputImage.empty()) {
		AddError(wxT("No input image."));
		return;
	}

	const int frameNumber = mCore->mDataStructureInput.mFrameNumber;
	const wxDateTime frameTimestamp = mCore->mDataStructureInput.mFrameTimestamp;

	mParticles.clear();
	for (auto & blob : mSplitter.Detect(inputImage)) {
		Particle particle;
		particle.mID = -1;
		particle.mIDCovariance = -1;
		particle.mCenter = blob.center;
		particle.mArea = blob.area;
		particle.mOrientation = blob.orientation;
		particle.mCompactness = 1.0 / blob.splitCount;
		particle.mBoundingBox = blob.boundingBox;
		particle.mScore = blob.score;
		particle.mFrameNumber = frameNumber;
		particle.mFrameTimestamp = frameTimestamp;
		mParticles.push_back(particle);
	}
	mCore->mDataStructureParticles.mParticles = &mParticles;

	// Display: blobs in gray, split detections highlighted
	DisplayEditor de(&mDisplayOutput);
	if (! de.IsActive()) return;

	cv::Mat image;
	cv::cvtColor(inputImage > 0, image, cv::COLOR_GRAY2BGR);
	image *= 0.5;
	for (auto & particle : mParticles) {
		cv::Scalar color = particle.mScore < 1.0f ? cv::Scalar(0, 160, 255) : cv::Scalar(0, 220, 0);
		cv::rectangle(image, particle.mBoundingBox, color, 1);
		cv::Point2f axis(std::cos(particle.mOrientation) * 8, std::sin(particle.mOrientation) * 8);
		cv::line(image, particle.mCenter - axis, particle.mCenter + axis, color, 1, cv::LINE_AA);
	}
	double objectArea = mSplitter.GetObjectArea();
	if (objectArea > 0)
		cv::putText(image, cv::format("object area %.0f px", objectArea), cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

	de.SetParticles(&mParticles);
	de.SetMainImage(image);
}

void THISCLASS::OnStepCleanup() {
}

void THISCLASS::OnStop() {
}
