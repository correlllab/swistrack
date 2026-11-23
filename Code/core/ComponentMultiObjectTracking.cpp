#include "ComponentMultiObjectTracking.h"
#define THISCLASS ComponentMultiObjectTracking

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <set>
#include "DisplayEditor.h"

THISCLASS::ComponentMultiObjectTracking(SwisTrackCore * stc):
	Component(stc, wxT("MultiObjectTracking")),
	mTracker(), mMaxNumberOfPoints(100), mOutputFilteredPositions(true),
	mTracks(), mTentativeTracks(),
	mDisplayOutput(wxT("Output"), wxT("Tracking")) {

	// Data structure relations
	mCategory = &(mCore->mCategoryTracking);
	AddDataStructureRead(&(mCore->mDataStructureInput));
	AddDataStructureRead(&(mCore->mDataStructureParticles));
	AddDataStructureWrite(&(mCore->mDataStructureParticles));
	AddDataStructureWrite(&(mCore->mDataStructureTracks));
	AddDisplay(&mDisplayOutput);

	// Read the XML configuration file
	Initialize();
}

THISCLASS::~ComponentMultiObjectTracking() {
}

void THISCLASS::OnStart() {
	mTracker.Reset();
	mTracks.clear();
	mTentativeTracks.clear();
	OnReloadConfiguration();
}

void THISCLASS::OnReloadConfiguration() {
	MultiObjectTracker::Parameters p;
	p.maxDistance = GetConfigurationDouble(wxT("MaxDistance"), p.maxDistance);
	p.useMahalanobis = GetConfigurationBool(wxT("UseMahalanobis"), p.useMahalanobis);
	p.processNoise = GetConfigurationDouble(wxT("ProcessNoise"), p.processNoise);
	p.measurementNoise = GetConfigurationDouble(wxT("MeasurementNoise"), p.measurementNoise);
	p.initialVelocityStd = GetConfigurationDouble(wxT("InitialVelocityStd"), p.initialVelocityStd);
	p.minHits = GetConfigurationInt(wxT("MinHits"), p.minHits);
	p.maxAge = GetConfigurationInt(wxT("MaxAge"), p.maxAge);
	p.highScoreThreshold = (float)GetConfigurationDouble(wxT("HighScoreThreshold"), p.highScoreThreshold);
	p.lowScoreThreshold = (float)GetConfigurationDouble(wxT("LowScoreThreshold"), p.lowScoreThreshold);
	p.newTrackScoreThreshold = (float)GetConfigurationDouble(wxT("NewTrackScoreThreshold"), p.newTrackScoreThreshold);
	p.directionWeight = GetConfigurationDouble(wxT("DirectionWeight"), p.directionWeight);
	p.observationCentricRecovery = GetConfigurationBool(wxT("ObservationCentricRecovery"), p.observationCentricRecovery);
	p.observationCentricReupdate = GetConfigurationBool(wxT("ObservationCentricReupdate"), p.observationCentricReupdate);
	p.useDetectionIDs = GetConfigurationBool(wxT("UseParticleIDs"), p.useDetectionIDs);
	p.maxTracks = GetConfigurationInt(wxT("MaxNumberOfTracks"), p.maxTracks);
	p.fixedNumberOfObjects = GetConfigurationBool(wxT("FixedNumberOfObjects"), p.fixedNumberOfObjects);
	p.reidentificationDistance = GetConfigurationDouble(wxT("ReidentificationDistance"), p.reidentificationDistance);

	if (p.maxDistance <= 0) AddError(wxT("The max. distance must be positive."));
	if (p.measurementNoise <= 0) AddError(wxT("The measurement noise must be positive."));
	if (p.processNoise < 0) AddError(wxT("The process noise must not be negative."));
	if (p.fixedNumberOfObjects && p.maxTracks <= 0) AddWarning(wxT("Set the max. number of tracks to the number of objects when using a fixed number of objects."));
	if (p.lowScoreThreshold > p.highScoreThreshold) AddWarning(wxT("The low score threshold is above the high score threshold."));

	mTracker.SetParameters(p);
	mMaxNumberOfPoints = (unsigned int)std::max(1, GetConfigurationInt(wxT("MaxNumberOfPoints"), 100));
	mOutputFilteredPositions = GetConfigurationBool(wxT("OutputFilteredPositions"), true);
}

void THISCLASS::OnStep() {
	const int frameNumber = mCore->mDataStructureInput.mFrameNumber;
	DataStructureParticles::tParticleVector * particles = mCore->mDataStructureParticles.mParticles;
	if (! particles) {
		AddError(wxT("No particles."));
		return;
	}

	// Run the tracker
	std::vector<MultiObjectTracker::Detection> detections;
	detections.reserve(particles->size());
	for (auto & particle : *particles)
		detections.push_back(MultiObjectTracker::Detection(particle.mCenter, particle.mScore, particle.mID));
	std::vector<int> ids = mTracker.Update(frameNumber, detections);

	// Label the particles with their track ID
	for (size_t i = 0; i < particles->size(); i++)
		(*particles)[i].mID = ids[i];

	// Update the trajectories
	std::set<int> alive;
	for (auto & info : mTracker.GetTracks()) {
		alive.insert(info.id);

		Track * track;
		if (info.state == MultiObjectTracker::sTentative) {
			track = &mTentativeTracks[info.id];
		} else {
			auto it = mTracks.find(info.id);
			if (it == mTracks.end()) {
				// Newly confirmed: publish the trajectory collected while it was tentative
				Track & published = mTracks[info.id];
				auto tentative = mTentativeTracks.find(info.id);
				if (tentative != mTentativeTracks.end()) {
					published = tentative->second;
					mTentativeTracks.erase(tentative);
				}
				track = &published;
			} else {
				track = &it->second;
			}
		}

		track->mID = info.id;
		track->mVelocity = info.velocity;
		if (info.detectionIndex >= 0) {
			cv::Point2f point = mOutputFilteredPositions ? info.position : info.observation;
			track->AddPoint(point.x, point.y, frameNumber);
			track->Shorten(mMaxNumberOfPoints);
		}
	}

	// Remove deleted tracks
	for (auto it = mTracks.begin(); it != mTracks.end(); )
		it = alive.count(it->first) ? std::next(it) : mTracks.erase(it);
	for (auto it = mTentativeTracks.begin(); it != mTentativeTracks.end(); )
		it = alive.count(it->first) ? std::next(it) : mTentativeTracks.erase(it);

	// Publish the tracks
	mCore->mDataStructureTracks.mTracks = &mTracks;

	// Display: input image with predictions of lost tracks and tentative tracks
	DisplayEditor de(&mDisplayOutput);
	if (! de.IsActive()) return;

	cv::Mat image;
	cv::Mat input = mCore->mDataStructureInput.mImage;
	if (input.empty()) return;
	if (input.channels() == 1) cv::cvtColor(input, image, cv::COLOR_GRAY2BGR);
	else image = input.clone();

	const double gate = mTracker.GetParameters().maxDistance;
	for (auto & info : mTracker.GetTracks()) {
		cv::Point center((int)std::lround(info.position.x), (int)std::lround(info.position.y));
		if (info.state == MultiObjectTracker::sTentative) {
			cv::circle(image, center, 4, cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
		} else if (info.state == MultiObjectTracker::sLost) {
			// Predicted position and 2-sigma uncertainty of a lost track
			double sigma = std::sqrt(std::max(info.positionCovariance(0, 0), info.positionCovariance(1, 1)));
			int radius = (int)std::min(gate, 2.0 * sigma);
			cv::circle(image, center, std::max(radius, 3), cv::Scalar(0, 140, 255), 1, cv::LINE_AA);
			cv::putText(image, std::to_string(info.id) + "?", center + cv::Point(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 140, 255), 1, cv::LINE_AA);
		} else {
			cv::Point tip((int)std::lround(info.position.x + 5 * info.velocity.x), (int)std::lround(info.position.y + 5 * info.velocity.y));
			cv::arrowedLine(image, center, tip, cv::Scalar(0, 200, 0), 1, cv::LINE_AA, 0, 0.3);
		}
	}

	de.SetParticles(particles);
	de.SetMainImage(image);
	de.SetTrajectories(true);
}

void THISCLASS::OnStepCleanup() {
}

void THISCLASS::OnStop() {
}
