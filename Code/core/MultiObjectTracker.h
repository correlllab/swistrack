#ifndef HEADER_MultiObjectTracker
#define HEADER_MultiObjectTracker

#include <deque>
#include <map>
#include <set>
#include <vector>
#include <opencv2/core.hpp>

//! Online multi-object tracker for point-like objects (tracking-by-detection).
/*!
	Combines the current best practice of the SORT family of trackers:
	 - a constant-velocity Kalman filter per track (SORT), with frame-gap aware prediction,
	 - optimal (Hungarian) association with Mahalanobis + Euclidean gating,
	 - two-stage association that also uses low-confidence detections to keep tracks alive (ByteTrack),
	 - observation-centric momentum (velocity-direction consistency) in the association cost,
	   observation-centric recovery (second association against the last observation) and
	   observation-centric re-update of the filter after occlusions (OC-SORT),
	 - tentative/confirmed/lost track life cycle,
	 - detections that already carry an identity (e.g. fiducial markers) bypass data association,
	 - an optional "known number of objects" mode that never creates more than N identities and
	   re-identifies lost objects instead (typical for animal experiments with a fixed group size).

	This class has no GUI or SwisTrack core dependencies, so it can be unit-tested and reused.
*/
class MultiObjectTracker {
public:
	//! Configuration.
	struct Parameters {
		double processNoise;				//!< Standard deviation of the (white) acceleration noise [px/frame^2].
		double measurementNoise;			//!< Standard deviation of the position measurement noise [px].
		double initialVelocityStd;			//!< Standard deviation of the velocity of a new track [px/frame].
		double maxDistance;					//!< Euclidean gate: max. distance between prediction and detection [px].
		bool useMahalanobis;				//!< Gate and score with the Mahalanobis distance of the Kalman innovation (otherwise Euclidean only).
		double mahalanobisGate;				//!< Chi-square (2 DOF) gate on the squared Mahalanobis distance (13.8 = 99.9%).
		float highScoreThreshold;			//!< Detections with score >= this are "high" (ByteTrack first stage).
		float lowScoreThreshold;			//!< Detections with score < this are ignored; [low, high) are used in the second stage.
		float newTrackScoreThreshold;		//!< Min. score of an unmatched detection to start a new track.
		int minHits;						//!< Number of associated detections before a track is confirmed.
		int maxAge;							//!< Number of frames a track may stay lost before it is deleted.
		double directionWeight;				//!< Weight of the velocity-direction consistency cost (OC-SORT OCM), 0 to disable.
		int directionDeltaFrames;			//!< Time span used to estimate a track's direction from its observations [frames].
		bool observationCentricRecovery;	//!< Second association of remaining tracks against their last observation (OC-SORT OCR).
		bool observationCentricReupdate;	//!< Re-run the filter along a virtual trajectory after an occlusion (OC-SORT ORU).
		bool useDetectionIDs;				//!< Detections with an ID (>= 0) are assigned to the track with that ID.
		int maxTracks;						//!< Max. number of simultaneous tracks (0 = unlimited).
		bool fixedNumberOfObjects;			//!< Known number of objects: only IDs 0..maxTracks-1 are used; newly confirmed tracks inherit the ID of a lost object.
		double reidentificationDistance;	//!< (fixed number of objects) Max. distance for re-identifying a lost track, 0 = unlimited [px].

		Parameters();
	};

	//! A detection fed into the tracker.
	struct Detection {
		cv::Point2f position;		//!< Position [px].
		float score;				//!< Confidence in [0, 1].
		int id;						//!< Known identity (>= 0), or -1.
		Detection(): position(), score(1.0f), id(-1) {}
		Detection(cv::Point2f position, float score = 1.0f, int id = -1): position(position), score(score), id(id) {}
	};

	//! Track life cycle state.
	enum TrackState {
		sTentative,		//!< New track, not yet confirmed.
		sTracked,		//!< Confirmed and associated in the current frame.
		sLost			//!< Confirmed, but not associated in the current frame.
	};

	//! Public view of a track.
	struct TrackInfo {
		int id;
		TrackState state;
		cv::Point2f position;		//!< Filtered position [px].
		cv::Point2f velocity;		//!< Filtered velocity [px/frame].
		cv::Point2f observation;	//!< Last associated detection [px].
		cv::Matx22d positionCovariance;
		int hits;					//!< Number of associated detections.
		int framesSinceUpdate;		//!< 0 if associated in the current frame.
		int detectionIndex;			//!< Index of the detection associated in the current frame, or -1.
	};

	MultiObjectTracker();

	//! Sets the configuration (can be called at any time).
	void SetParameters(const Parameters & parameters);
	//! Returns the configuration.
	const Parameters & GetParameters() const { return mParameters; }

	//! Removes all tracks and restarts ID numbering.
	void Reset();

	//! Processes the detections of one frame. Frame numbers must increase (gaps are allowed).
	/*!
		Returns, for each detection, the ID of the confirmed track it was associated with, or -1.
	*/
	std::vector<int> Update(int frameNumber, const std::vector<Detection> & detections);

	//! Returns all tracks (tentative, tracked and lost) after the last update.
	std::vector<TrackInfo> GetTracks() const;

	//! Returns the number of frames processed since the last reset.
	int GetFrameCount() const { return mFrameCount; }

private:
	typedef cv::Matx<double, 4, 1> tState;
	typedef cv::Matx<double, 4, 4> tCovariance;

	struct Observation {
		int frameNumber;
		cv::Point2f position;
	};

	struct Track {
		int id;
		TrackState state;
		tState x;							//!< [x, y, vx, vy]
		tCovariance P;
		int lastFrame;						//!< Frame the filter state refers to.
		int hits;
		int framesSinceUpdate;
		int detectionIndex;
		std::deque<Observation> observations;	//!< Recent associated observations (newest last).
		tState xAtLastObservation;			//!< Filter state right after the last update (for re-update).
		tCovariance PAtLastObservation;
	};

	Parameters mParameters;
	std::map<int, Track> mTracks;			//!< Tracks by ID.
	int mNextID;
	int mFrameCount;
	int mLastFrameNumber;

	// Kalman filter
	void Predict(Track & track, int frameNumber) const;
	void Correct(Track & track, cv::Point2f z) const;
	void Initiate(Track & track, cv::Point2f z, int frameNumber) const;
	tCovariance ProcessNoise(double dt) const;

	// Association
	enum CostMode {
		sCostPrediction,		//!< Distance to the predicted position (gated, with direction term).
		sCostLastObservation	//!< Distance to the last observation (OC-SORT recovery).
	};
	double Cost(const Track & track, const Detection & detection, CostMode mode) const;
	void Associate(std::vector<int> & trackIDs, std::vector<int> & detectionIndices, const std::vector<Detection> & detections, CostMode mode, int frameNumber);
	void Assign(Track & track, int detectionIndex, const Detection & detection, int frameNumber);
	int AllocateID();
	bool FixedMode() const;
	void PromoteConfirmedTracks();
	static const int sTemporaryIDBase = 1 << 28;	//!< Temporary IDs of unconfirmed tracks (known number of objects).
	int NumberOfIdentities() const;
};

#endif
