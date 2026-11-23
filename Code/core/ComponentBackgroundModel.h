#ifndef HEADER_ComponentBackgroundModel
#define HEADER_ComponentBackgroundModel

#include <opencv2/core.hpp>
#include <opencv2/video/background_segm.hpp>
#include "Component.h"

//! Adaptive background model (Gaussian mixture MOG2 or k-nearest-neighbors KNN) producing a foreground mask.
class ComponentBackgroundModel: public Component {

public:
	//! Constructor.
	ComponentBackgroundModel(SwisTrackCore * stc);
	//! Destructor.
	~ComponentBackgroundModel();

	// Overwritten Component methods
	void OnStart();
	void OnReloadConfiguration();
	void OnStep();
	void OnStepCleanup();
	void OnStop();

	Component * Create() {
		return new ComponentBackgroundModel(mCore);
	}

private:
	wxString mMethod;						//!< (configuration) "MOG2" or "KNN".
	bool mUseColor;							//!< (configuration) Model the color image (otherwise the grayscale image).
	int mHistory;							//!< (configuration) Number of frames that shape the model.
	double mThreshold;						//!< (configuration) MOG2: squared Mahalanobis distance; KNN: squared distance threshold.
	bool mDetectShadows;					//!< (configuration) Detect and remove shadows.
	double mLearningRate;					//!< (configuration) Learning rate (negative = automatic from history).
	int mFreezeAfterFrames;					//!< (configuration) Stop learning after this many frames (0 = never).
	int mOpeningSize;						//!< (configuration) Kernel size of a morphological opening applied to the mask (0 = none).

	cv::Ptr<cv::BackgroundSubtractor> mSubtractor;	//!< The background model.
	int mFramesLearned;						//!< Number of frames the model has learned.
	Display mDisplayOutput;					//!< Foreground mask.
	Display mDisplayBackground;				//!< Current background image.

	void CreateModel();
};

#endif
