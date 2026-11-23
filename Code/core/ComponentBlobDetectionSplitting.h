#ifndef HEADER_ComponentBlobDetectionSplitting
#define HEADER_ComponentBlobDetectionSplitting

#include <opencv2/core.hpp>
#include "Component.h"
#include "DataStructureParticles.h"
#include "BlobSplitter.h"

//! Blob detection that splits blobs of touching objects (e.g. animals in contact) into individual particles.
class ComponentBlobDetectionSplitting: public Component {

public:
	//! Constructor.
	ComponentBlobDetectionSplitting(SwisTrackCore * stc);
	//! Destructor.
	~ComponentBlobDetectionSplitting();

	// Overwritten Component methods
	void OnStart();
	void OnReloadConfiguration();
	void OnStep();
	void OnStepCleanup();
	void OnStop();

	Component * Create() {
		return new ComponentBlobDetectionSplitting(mCore);
	}

private:
	BlobSplitter mSplitter;									//!< The blob detector.
	DataStructureParticles::tParticleVector mParticles;		//!< The detected particles.
	Display mDisplayOutput;									//!< Display.
};

#endif
