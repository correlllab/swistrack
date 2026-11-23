#ifndef HEADER_ComponentMultiObjectTracking
#define HEADER_ComponentMultiObjectTracking

#include <opencv2/core.hpp>
#include "Component.h"
#include "DataStructureParticles.h"
#include "DataStructureTracks.h"
#include "MultiObjectTracker.h"

//! Multi-object tracking with a Kalman filter, optimal association, ByteTrack two-stage association and OC-SORT occlusion handling.
class ComponentMultiObjectTracking: public Component {

public:
	//! Constructor.
	ComponentMultiObjectTracking(SwisTrackCore * stc);
	//! Destructor.
	~ComponentMultiObjectTracking();

	// Overwritten Component methods
	void OnStart();
	void OnReloadConfiguration();
	void OnStep();
	void OnStepCleanup();
	void OnStop();

	Component * Create() {
		return new ComponentMultiObjectTracking(mCore);
	}

private:
	MultiObjectTracker mTracker;				//!< The tracker.
	unsigned int mMaxNumberOfPoints;			//!< (configuration) The maximum number of points per track to keep in memory.
	bool mOutputFilteredPositions;				//!< (configuration) Whether trajectories contain filtered positions (instead of raw detections).

	DataStructureTracks::tTrackMap mTracks;		//!< Confirmed tracks (published to the core).
	DataStructureTracks::tTrackMap mTentativeTracks;	//!< Trajectories of tentative tracks (published once confirmed).
	Display mDisplayOutput;						//!< The Display showing the tracks.
};

#endif
