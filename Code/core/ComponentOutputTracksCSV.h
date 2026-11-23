#ifndef HEADER_ComponentOutputTracksCSV
#define HEADER_ComponentOutputTracksCSV

#include <fstream>
#include "Component.h"

//! Writes all tracks into a single CSV file (frame,id,x,y,vx,vy), one row per track updated in the current frame.
/*!
	A relative file name is resolved with respect to the run folder (like all other output components),
	i.e. "<config folder>/testing/" when running unattended (SwisTrack-batch) or in test mode, and
	"<config folder>/<timestamp>/" in production mode. Absolute paths are used as they are.
*/
class ComponentOutputTracksCSV: public Component {

public:
	//! Constructor.
	ComponentOutputTracksCSV(SwisTrackCore * stc);
	//! Destructor.
	~ComponentOutputTracksCSV();

	// Overwritten Component methods
	void OnStart();
	void OnReloadConfiguration();
	void OnStep();
	void OnStepCleanup();
	void OnStop();

	Component * Create() {
		return new ComponentOutputTracksCSV(mCore);
	}

private:
	std::ofstream mFile;	//!< The output file.
};

#endif
