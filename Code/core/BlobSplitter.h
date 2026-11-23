#ifndef HEADER_BlobSplitter
#define HEADER_BlobSplitter

#include <deque>
#include <vector>
#include <opencv2/core.hpp>

//! Detects blobs in a binary image and splits blobs of touching objects into individual detections.
/*!
	Blobs are connected components. The area of a single object is either given or learned as the
	running median of blob areas (most blobs are single objects). A blob whose area is about k times
	the single-object area is split into k detections by k-means clustering of its pixels.
	Split detections get a lower score, so that trackers can treat them with more caution.
	No GUI or SwisTrack core dependencies.
*/
class BlobSplitter {
public:
	struct Parameters {
		int minArea;					//!< Blobs smaller than this are ignored [px^2].
		int maxArea;					//!< Blobs larger than this are ignored, 0 = no limit [px^2].
		double objectArea;				//!< Area of one object, 0 = learn automatically [px^2].
		double splitRatio;				//!< A blob is split if its area is at least this many times the object area.
		int maxObjectsPerBlob;			//!< Max. number of objects a blob is split into.
		int maxNumber;					//!< Max. number of detections (largest first), 0 = unlimited.
		float splitScore;				//!< Score assigned to detections obtained by splitting.
		Parameters();
	};

	struct Blob {
		cv::Point2f center;
		double area;
		float orientation;				//!< Major axis angle [rad].
		cv::Rect2f boundingBox;
		float score;
		int splitCount;					//!< Number of objects the original blob was split into (1 = not split).
		int label;						//!< Connected component label (for display).
	};

	BlobSplitter();
	void SetParameters(const Parameters & parameters) { mParameters = parameters; }
	const Parameters & GetParameters() const { return mParameters; }
	void Reset();

	//! Detects (and splits) blobs in a binary image (non-zero = foreground).
	std::vector<Blob> Detect(const cv::Mat & binary);

	//! Returns the current object area estimate (given or learned).
	double GetObjectArea() const;
	//! Returns the learned pixel variances of a single object along and across its major axis (0 if unknown).
	cv::Vec2d GetObjectShape() const;

	//! Returns the connected component labels of the last call (for display).
	const cv::Mat & GetLabels() const { return mLabels; }

private:
	Parameters mParameters;
	std::deque<double> mAreaHistory;	//!< Recent blob areas used to learn the object area.
	std::deque<double> mMajorHistory;	//!< Recent pixel variances along the major axis of single objects.
	std::deque<double> mMinorHistory;	//!< Recent pixel variances along the minor axis of single objects.
	cv::Mat mLabels;

	static float Orientation(double mu20, double mu02, double mu11);
	//! Assigns each pixel to one of k objects. Returns the assignment.
	std::vector<int> Split(const std::vector<cv::Point2f> & pixels, int k) const;
	//! Fits k fixed-shape Gaussians by EM, starting from the given means/orientations. Returns the log-likelihood.
	static double FitShapeMixture(const std::vector<cv::Point2f> & pixels, std::vector<cv::Point2d> & means, std::vector<double> & angles, cv::Vec2d shape, std::vector<int> & assignment);
};

#endif
