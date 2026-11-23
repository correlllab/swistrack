#include "MultiObjectTracker.h"
#define THISCLASS MultiObjectTracker

#include <algorithm>
#include <cmath>
#include <limits>
#include "LinearAssignment.h"

static const double sInfinity = std::numeric_limits<double>::infinity();

THISCLASS::Parameters::Parameters():
	processNoise(1.0), measurementNoise(2.0), initialVelocityStd(10.0),
	maxDistance(30.0), useMahalanobis(true), mahalanobisGate(13.8),
	highScoreThreshold(0.5f), lowScoreThreshold(0.1f), newTrackScoreThreshold(0.6f),
	minHits(3), maxAge(30),
	directionWeight(0.2), directionDeltaFrames(3),
	observationCentricRecovery(true), observationCentricReupdate(true),
	useDetectionIDs(true), maxTracks(0), fixedNumberOfObjects(false), reidentificationDistance(0) {
}

THISCLASS::MultiObjectTracker():
	mParameters(), mTracks(), mNextID(0), mFrameCount(0), mLastFrameNumber(0) {
}

void THISCLASS::SetParameters(const Parameters & parameters) {
	mParameters = parameters;
}

void THISCLASS::Reset() {
	mTracks.clear();
	mNextID = 0;
	mFrameCount = 0;
	mLastFrameNumber = 0;
}

// --- Kalman filter (constant velocity, white acceleration noise) ------------

THISCLASS::tCovariance THISCLASS::ProcessNoise(double dt) const {
	const double q = mParameters.processNoise * mParameters.processNoise;
	const double a = dt * dt * dt * dt / 4.0 * q;
	const double b = dt * dt * dt / 2.0 * q;
	const double c = dt * dt * q;
	return tCovariance(
		a, 0, b, 0,
		0, a, 0, b,
		b, 0, c, 0,
		0, b, 0, c);
}

void THISCLASS::Initiate(Track & track, cv::Point2f z, int frameNumber) const {
	const double r2 = mParameters.measurementNoise * mParameters.measurementNoise;
	const double v2 = mParameters.initialVelocityStd * mParameters.initialVelocityStd;
	track.x = tState(z.x, z.y, 0, 0);
	track.P = tCovariance::zeros();
	track.P(0, 0) = r2;
	track.P(1, 1) = r2;
	track.P(2, 2) = v2;
	track.P(3, 3) = v2;
	track.lastFrame = frameNumber;
	track.xAtLastObservation = track.x;
	track.PAtLastObservation = track.P;
}

void THISCLASS::Predict(Track & track, int frameNumber) const {
	const double dt = frameNumber - track.lastFrame;
	if (dt <= 0) return;
	tCovariance F = tCovariance::eye();
	F(0, 2) = dt;
	F(1, 3) = dt;
	track.x = F * track.x;
	track.P = F * track.P * F.t() + ProcessNoise(dt);
	track.lastFrame = frameNumber;
}

void THISCLASS::Correct(Track & track, cv::Point2f z) const {
	const double r2 = mParameters.measurementNoise * mParameters.measurementNoise;
	cv::Matx<double, 2, 4> H(1, 0, 0, 0, 0, 1, 0, 0);
	cv::Matx22d S = H * track.P * H.t() + cv::Matx22d(r2, 0, 0, r2);
	cv::Matx<double, 4, 2> K = track.P * H.t() * S.inv();
	cv::Matx21d y(z.x - track.x(0), z.y - track.x(1));
	track.x = track.x + K * y;
	// Joseph form keeps P symmetric positive definite
	tCovariance IKH = tCovariance::eye() - K * H;
	track.P = IKH * track.P * IKH.t() + K * cv::Matx22d(r2, 0, 0, r2) * K.t();
}

// --- Association -------------------------------------------------------------

double THISCLASS::Cost(const Track & track, const Detection & detection, CostMode mode) const {
	const Parameters & p = mParameters;

	if (mode == sCostReidentification) {
		cv::Point2f last = track.observations.empty() ? cv::Point2f((float)track.x(0), (float)track.x(1)) : track.observations.back().position;
		double d = cv::norm(detection.position - last);
		if (p.reidentificationDistance > 0 && d > p.reidentificationDistance) return sInfinity;
		return d;
	}

	if (mode == sCostLastObservation) {
		if (track.observations.empty()) return sInfinity;
		double d = cv::norm(detection.position - track.observations.back().position);
		if (d > p.maxDistance) return sInfinity;
		return d * d / (p.maxDistance * p.maxDistance);
	}

	// Distance to the prediction
	const double r2 = p.measurementNoise * p.measurementNoise;
	cv::Matx21d y(detection.position.x - track.x(0), detection.position.y - track.x(1));
	const double euclidean2 = y(0) * y(0) + y(1) * y(1);
	if (euclidean2 > p.maxDistance * p.maxDistance) return sInfinity;

	double cost;
	if (p.useMahalanobis) {
		cv::Matx22d S(track.P(0, 0) + r2, track.P(0, 1), track.P(1, 0), track.P(1, 1) + r2);
		double d2 = (y.t() * S.inv() * y)(0, 0);
		if (d2 > p.mahalanobisGate) return sInfinity;
		cost = d2 / p.mahalanobisGate;
	} else {
		cost = euclidean2 / (p.maxDistance * p.maxDistance);
	}

	// Velocity-direction consistency (OC-SORT observation-centric momentum)
	if (p.directionWeight > 0 && track.observations.size() >= 2) {
		const Observation & last = track.observations.back();
		const Observation * previous = &track.observations.front();
		for (auto it = track.observations.rbegin() + 1; it != track.observations.rend(); it++) {
			previous = &*it;
			if (last.frameNumber - it->frameNumber >= p.directionDeltaFrames) break;
		}
		cv::Point2f trackDirection = last.position - previous->position;
		cv::Point2f candidateDirection = detection.position - last.position;
		const double minNorm = 2.0 * p.measurementNoise;	// below this, directions are dominated by noise
		double n1 = cv::norm(trackDirection), n2 = cv::norm(candidateDirection);
		if (n1 > minNorm && n2 > minNorm) {
			double c = (trackDirection.x * candidateDirection.x + trackDirection.y * candidateDirection.y) / (n1 * n2);
			c = std::max(-1.0, std::min(1.0, c));
			cost += p.directionWeight * std::acos(c) / CV_PI;
		}
	}

	return cost;
}

void THISCLASS::Associate(std::vector<int> & trackIDs, std::vector<int> & detectionIndices, const std::vector<Detection> & detections, CostMode mode, int frameNumber) {
	if (trackIDs.empty() || detectionIndices.empty()) return;

	const int rows = (int)trackIDs.size(), cols = (int)detectionIndices.size();
	LinearAssignment::tCostMatrix cost((size_t)rows * cols);
	for (int r = 0; r < rows; r++) {
		const Track & track = mTracks[trackIDs[r]];
		for (int c = 0; c < cols; c++)
			cost[(size_t)r * cols + c] = Cost(track, detections[detectionIndices[c]], mode);
	}

	std::vector<int> unassignedRows, unassignedCols;
	auto pairs = LinearAssignment::Solve(cost, rows, cols, sInfinity, unassignedRows, unassignedCols);
	for (auto & pair : pairs) {
		int detectionIndex = detectionIndices[pair.second];
		Assign(mTracks[trackIDs[pair.first]], detectionIndex, detections[detectionIndex], frameNumber);
	}

	std::vector<int> remainingTracks, remainingDetections;
	for (int r : unassignedRows) remainingTracks.push_back(trackIDs[r]);
	for (int c : unassignedCols) remainingDetections.push_back(detectionIndices[c]);
	trackIDs.swap(remainingTracks);
	detectionIndices.swap(remainingDetections);
}

void THISCLASS::Assign(Track & track, int detectionIndex, const Detection & detection, int frameNumber) {
	// Observation-centric re-update (OC-SORT): after missed frames, the filter coasted on its own prediction and
	// accumulated error. Rewind to the state at the last observation, and re-run the filter along a virtual
	// trajectory that linearly interpolates between the last and the new observation.
	const int missedFrames = track.framesSinceUpdate - 1;
	if (mParameters.observationCentricReupdate && missedFrames > 0 && ! track.observations.empty()) {
		const Observation last = track.observations.back();
		const int span = frameNumber - last.frameNumber;
		if (span > 1) {
			track.x = track.xAtLastObservation;
			track.P = track.PAtLastObservation;
			track.lastFrame = last.frameNumber;
			for (int f = last.frameNumber + 1; f < frameNumber; f++) {
				float t = (float)(f - last.frameNumber) / (float)span;
				Predict(track, f);
				Correct(track, last.position + (detection.position - last.position) * t);
			}
			Predict(track, frameNumber);
		}
	}

	Correct(track, detection.position);

	track.hits++;
	track.framesSinceUpdate = 0;
	track.detectionIndex = detectionIndex;
	track.observations.push_back(Observation{frameNumber, detection.position});
	while ((int)track.observations.size() > std::max(2, mParameters.directionDeltaFrames + 1))
		track.observations.pop_front();
	track.xAtLastObservation = track.x;
	track.PAtLastObservation = track.P;

	if (track.state == sLost) track.state = sTracked;
	else if (track.state == sTentative && (track.hits >= mParameters.minHits || mFrameCount == 1)) track.state = sTracked;
}

int THISCLASS::NumberOfIdentities() const {
	return (int)mTracks.size();
}

int THISCLASS::AllocateID() {
	if (mParameters.fixedNumberOfObjects && mParameters.maxTracks > 0) {
		for (int id = 0; id < mParameters.maxTracks; id++)
			if (mTracks.find(id) == mTracks.end()) return id;
		return -1;
	}
	while (mTracks.find(mNextID) != mTracks.end()) mNextID++;
	return mNextID++;
}

// --- Main update ---------------------------------------------------------------

std::vector<int> THISCLASS::Update(int frameNumber, const std::vector<Detection> & detections) {
	const Parameters & p = mParameters;
	mFrameCount++;
	mLastFrameNumber = frameNumber;

	// Predict all tracks to the current frame
	for (auto & entry : mTracks) {
		Predict(entry.second, frameNumber);
		entry.second.framesSinceUpdate++;
		entry.second.detectionIndex = -1;
	}

	// Detections with a known identity bypass data association
	std::vector<int> high, low;
	for (int i = 0; i < (int)detections.size(); i++) {
		const Detection & detection = detections[i];
		if (p.useDetectionIDs && detection.id >= 0) {
			auto it = mTracks.find(detection.id);
			if (it == mTracks.end()) {
				Track track;
				track.id = detection.id;
				track.state = sTracked;
				track.hits = 0;
				track.framesSinceUpdate = 1;
				track.detectionIndex = -1;
				Initiate(track, detection.position, frameNumber);
				it = mTracks.insert(std::make_pair(detection.id, track)).first;
			}
			if (it->second.detectionIndex < 0) {	// ignore duplicate IDs in the same frame
				Assign(it->second, i, detection, frameNumber);
				it->second.state = sTracked;
			}
		} else if (detection.score >= p.highScoreThreshold) {
			high.push_back(i);
		} else if (detection.score >= p.lowScoreThreshold) {
			low.push_back(i);
		}
	}

	// Candidate tracks: confirmed tracks (tracked or lost) that were not associated by ID
	std::vector<int> confirmed, tentative;
	for (auto & entry : mTracks) {
		if (entry.second.detectionIndex >= 0) continue;
		if (entry.second.state == sTentative) tentative.push_back(entry.first);
		else confirmed.push_back(entry.first);
	}

	// Stage 1 (ByteTrack): confirmed tracks vs. high-confidence detections
	Associate(confirmed, high, detections, sCostPrediction, frameNumber);

	// Stage 2 (ByteTrack): tracks that were tracked in the previous frame vs. low-confidence detections
	std::vector<int> recentlyTracked, others;
	for (int id : confirmed) (mTracks[id].state == sTracked ? recentlyTracked : others).push_back(id);
	Associate(recentlyTracked, low, detections, sCostPrediction, frameNumber);
	confirmed = others;
	confirmed.insert(confirmed.end(), recentlyTracked.begin(), recentlyTracked.end());

	// Stage 3 (OC-SORT recovery): remaining tracks vs. remaining high detections, using the last observation
	if (p.observationCentricRecovery)
		Associate(confirmed, high, detections, sCostLastObservation, frameNumber);

	// Stage 4: tentative tracks vs. remaining high detections
	Associate(tentative, high, detections, sCostPrediction, frameNumber);

	// Known number of objects: re-identify lost tracks with the remaining detections
	if (p.fixedNumberOfObjects)
		Associate(confirmed, high, detections, sCostReidentification, frameNumber);

	// Unmatched tentative tracks are dropped, unmatched confirmed tracks are lost
	for (int id : tentative) mTracks.erase(id);
	for (int id : confirmed) {
		Track & track = mTracks[id];
		track.state = sLost;
		if (! p.fixedNumberOfObjects && track.framesSinceUpdate > p.maxAge) mTracks.erase(id);
	}

	// New tracks from unmatched high-confidence detections (most confident first)
	std::sort(high.begin(), high.end(), [&](int a, int b) { return detections[a].score > detections[b].score; });
	for (int i : high) {
		const Detection & detection = detections[i];
		if (detection.score < p.newTrackScoreThreshold) continue;
		if (p.maxTracks > 0 && NumberOfIdentities() >= p.maxTracks) break;
		int id = AllocateID();
		if (id < 0) break;

		Track track;
		track.id = id;
		track.state = sTentative;
		track.hits = 0;
		track.framesSinceUpdate = 1;
		track.detectionIndex = -1;
		Initiate(track, detection.position, frameNumber);
		Track & inserted = mTracks.insert(std::make_pair(id, track)).first->second;
		inserted.hits = 1;
		inserted.framesSinceUpdate = 0;
		inserted.detectionIndex = i;
		inserted.observations.push_back(Observation{frameNumber, detection.position});
		if (p.minHits <= 1 || mFrameCount == 1) inserted.state = sTracked;
	}

	// Result: detection -> confirmed track ID
	std::vector<int> result(detections.size(), -1);
	for (auto & entry : mTracks) {
		const Track & track = entry.second;
		if (track.detectionIndex >= 0 && track.state != sTentative)
			result[track.detectionIndex] = track.id;
	}
	return result;
}

std::vector<THISCLASS::TrackInfo> THISCLASS::GetTracks() const {
	std::vector<TrackInfo> result;
	result.reserve(mTracks.size());
	for (auto & entry : mTracks) {
		const Track & track = entry.second;
		TrackInfo info;
		info.id = track.id;
		info.state = track.state;
		info.position = cv::Point2f((float)track.x(0), (float)track.x(1));
		info.velocity = cv::Point2f((float)track.x(2), (float)track.x(3));
		info.observation = track.observations.empty() ? info.position : track.observations.back().position;
		info.positionCovariance = cv::Matx22d(track.P(0, 0), track.P(0, 1), track.P(1, 0), track.P(1, 1));
		info.hits = track.hits;
		info.framesSinceUpdate = track.framesSinceUpdate;
		info.detectionIndex = track.detectionIndex;
		result.push_back(info);
	}
	return result;
}
