#ifndef HEADER_ComponentObjectDetectionDNN
#define HEADER_ComponentObjectDetectionDNN

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "Component.h"
#include "DataStructureParticles.h"
#include "ObjectDetectorDNN.h"

//! A component that detects objects with a deep learning model (ONNX, YOLO family) and stores them as particles with bounding boxes and scores.
class ComponentObjectDetectionDNN: public Component {

public:
	//! Constructor.
	ComponentObjectDetectionDNN(SwisTrackCore * stc);
	//! Destructor.
	~ComponentObjectDetectionDNN();

	// Overwritten Component methods
	void OnStart();
	void OnReloadConfiguration();
	void OnStep();
	void OnStepCleanup();
	void OnStop();

	Component * Create() {
		return new ComponentObjectDetectionDNN(mCore);
	}

private:
	ObjectDetectorDNN mDetector;							//!< The model.
	ObjectDetectorDNN::Thresholds mThresholds;				//!< Score/NMS thresholds and class filter.
	std::vector<ObjectDetectorDNN::Detection> mDetections;	//!< The detections of the last frame.
	DataStructureParticles::tParticleVector mParticles;		//!< The particles of the last frame.
	std::vector<std::string> mClassNames;					//!< Class names (for display).
	float mDisplayScoreThreshold;							//!< Detections with a score at least this high are highlighted.
	wxString mLoadedModelKey;								//!< Describes the configuration of the loaded model, to avoid reloading it unnecessarily.
	std::vector<wxString> mLoadWarnings;					//!< Warnings from loading the model.
	Display mDisplayOutput;									//!< The Display showing the input image with the detections.

	//! Draws the detections onto an image.
	void DrawDetections(cv::Mat & image) const;
};

#endif
