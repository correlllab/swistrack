#include "ComponentMarkerDetection.h"
#define THISCLASS ComponentMarkerDetection

#include <string>
#include <opencv2/imgproc.hpp>
#include "DisplayEditor.h"

THISCLASS::ComponentMarkerDetection(SwisTrackCore * stc):
	Component(stc, wxT("MarkerDetection")),
	mDetector(), mConfigured(false), mMarkers(), mParticles(),
	mDisplayOutput(wxT("Output"), wxT("Markers")) {

	// Data structure relations
	mCategory = &(mCore->mCategoryParticleDetection);
	AddDataStructureRead(&(mCore->mDataStructureImageGray));
	AddDataStructureWrite(&(mCore->mDataStructureParticles));
	AddDisplay(&mDisplayOutput);

	// Read the XML configuration file
	Initialize();
}

THISCLASS::~ComponentMarkerDetection() {
}

void THISCLASS::OnStart() {
	OnReloadConfiguration();
}

void THISCLASS::OnReloadConfiguration() {
	MarkerDetector::Settings settings;
	settings.dictionary = std::string(GetConfigurationString(wxT("Dictionary"), wxT("APRILTAG_36h11")).mb_str(wxConvUTF8));
	settings.minMarkerPerimeterRate = GetConfigurationDouble(wxT("MinMarkerPerimeterRate"), 0.03);
	wxString refinement = GetConfigurationString(wxT("CornerRefinement"), wxT("Subpix"));
	if (! MarkerDetector::ParseCornerRefinement(std::string(refinement.mb_str(wxConvUTF8)), settings.cornerRefinement)) {
		AddError(wxT("Unknown corner refinement method '") + refinement + wxT("'."));
		settings.cornerRefinement = MarkerDetector::CornerRefineSubpix;
	}
	settings.filterByID = GetConfigurationBool(wxT("FilterByID"), false);
	settings.minID = GetConfigurationInt(wxT("MinID"), 0);
	settings.maxID = GetConfigurationInt(wxT("MaxID"), 1000);
	settings.keepRejected = GetConfigurationBool(wxT("ShowRejected"), false);

	std::string error;
	mConfigured = mDetector.Configure(settings, error);
	if (! mConfigured) {
		AddError(wxString::FromUTF8(error.c_str()));
	}
}

void THISCLASS::OnStep() {
	mParticles.clear();
	mMarkers.clear();
	mCore->mDataStructureParticles.mParticles = &mParticles;

	// Get and check input image
	cv::Mat inputImage = mCore->mDataStructureImageGray.mImage;
	if (inputImage.empty()) {
		AddError(wxT("No image available on the grayscale channel. You may want to add a 'Conversion to Grayscale' component."));
		return;
	}
	if (! mConfigured) {
		if (! MarkerDetector::IsAvailable()) {
			AddError(wxT("SwisTrack was built without ArUco support."));
		} else {
			AddError(wxT("Invalid marker detection settings."));
		}
		return;
	}

	// Detect the markers
	std::string error;
	if (! mDetector.Detect(inputImage, mMarkers, error)) {
		AddError(wxString::FromUTF8(error.c_str()));
		return;
	}

	// Convert them into particles
	for (const MarkerDetector::Marker & marker : mMarkers) {
		Particle particle;
		particle.mID = marker.id;
		particle.mIDCovariance = 0;
		particle.mCenter = marker.center;
		particle.mOrientation = marker.orientation;
		particle.mArea = marker.area;
		particle.mCompactness = 0;
		particle.mScore = 1.0f;
		particle.mBoundingBox = marker.boundingBox;
		particle.mFrameNumber = mCore->mDataStructureInput.mFrameNumber;
		particle.mFrameTimestamp = mCore->mDataStructureInput.mFrameTimestamp;
		mParticles.push_back(particle);
	}

	// Let the Display know about our image
	DisplayEditor de(&mDisplayOutput);
	if (! de.IsActive()) return;

	cv::Mat outputImage;
	cv::cvtColor(inputImage, outputImage, cv::COLOR_GRAY2BGR);
	MarkerDetector::Draw(outputImage, mMarkers, mDetector.Rejected());
	de.SetParticles(&mParticles);
	de.SetMainImage(outputImage);
}

void THISCLASS::OnStepCleanup() {
}

void THISCLASS::OnStop() {
}
