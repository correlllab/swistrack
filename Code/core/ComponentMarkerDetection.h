#ifndef HEADER_ComponentMarkerDetection
#define HEADER_ComponentMarkerDetection

#include <vector>
#include <opencv2/core.hpp>
#include "Component.h"
#include "DataStructureParticles.h"
#include "MarkerDetector.h"

//! A component that detects fiducial markers (ArUco, AprilTag) on the grayscale image and stores them as particles.
/*!
	Each detected marker becomes one particle with mID set to the marker ID, so that trackers can use it as track ID.
	The particle orientation is the angle of the marker's x-axis in image coordinates (see MarkerDetector).
*/
class ComponentMarkerDetection: public Component {

public:
	//! Constructor.
	ComponentMarkerDetection(SwisTrackCore * stc);
	//! Destructor.
	~ComponentMarkerDetection();

	// Overwritten Component methods
	void OnStart();
	void OnReloadConfiguration();
	void OnStep();
	void OnStepCleanup();
	void OnStop();

	Component * Create() {
		return new ComponentMarkerDetection(mCore);
	}

private:
	MarkerDetector mDetector;							//!< The marker detector.
	bool mConfigured;									//!< Whether the detector was configured successfully.
	std::vector<MarkerDetector::Marker> mMarkers;		//!< The markers of the current frame.
	DataStructureParticles::tParticleVector mParticles;	//!< The particles of the current frame.
	Display mDisplayOutput;								//!< The Display showing the image and the detected markers.
};

#endif
