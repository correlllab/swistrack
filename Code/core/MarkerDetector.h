#ifndef HEADER_MarkerDetector
#define HEADER_MarkerDetector

#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

// Which OpenCV ArUco API this build uses:
//  - OpenCV >= 4.7 ships ArUco/AprilTag detection in the objdetect module (cv::aruco::ArucoDetector).
//  - Older OpenCV versions only have it in the opencv_contrib "aruco" module (legacy free functions);
//    CMake defines SWISTRACK_HAVE_ARUCO_CONTRIB when that module was found.
//  - Otherwise, marker detection is not available and MarkerDetector reports an error.
#if defined(SWISTRACK_HAVE_OPENCV_OBJDETECT) && (CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7))
#define SWISTRACK_MARKERS_OBJDETECT 1
#elif defined(SWISTRACK_HAVE_ARUCO_CONTRIB) && (CV_VERSION_MAJOR == 4 || CV_VERSION_MAJOR == 3)
#define SWISTRACK_MARKERS_CONTRIB 1
#endif

//! Detects square fiducial markers (ArUco, AprilTag) in grayscale images.
/*!
	This class only depends on OpenCV (not on wxWidgets or the SwisTrack core), so that it can be unit-tested on its own.
	It is used by ComponentMarkerDetection.

	Corner order: OpenCV returns the 4 corners of each marker clockwise (on screen), starting with the marker's top-left
	corner (as the marker was generated). The marker's x-axis points from its top-left to its top-right corner.

	Orientation convention: the orientation is the angle of the marker's x-axis in image coordinates (x to the right, y
	downwards), i.e. atan2(dy, dx), in radians in (-pi, pi]. A marker seen upright has orientation 0, and the angle grows
	when the marker is rotated clockwise on screen. This is the same convention as the other SwisTrack detectors and the
	particle display (cos(orientation), sin(orientation)) in image pixels.

	Note that OpenCV's AprilTag dictionaries are rotated by 180 degrees with respect to the official AprilTag images
	(https://github.com/AprilRobotics/apriltag-imgs): a tag printed from those images upright has orientation +-pi.
*/
class MarkerDetector {

public:
	//! A detected marker.
	struct Marker {
		int id;								//!< Marker ID within the dictionary.
		std::vector<cv::Point2f> corners;	//!< The 4 corners [pixels], clockwise starting at the marker's top-left corner.
		cv::Point2f center;					//!< Mean of the 4 corners [pixels].
		float orientation;					//!< Angle of the marker's x-axis [rad], see class description.
		double area;						//!< Area of the marker polygon [pixels^2].
		cv::Rect2f boundingBox;				//!< Bounding box of the corners [pixels].

		Marker(): id(-1), corners(), center(), orientation(0), area(0), boundingBox() {}
	};

	//! Corner refinement methods.
	enum CornerRefinement {
		CornerRefineNone,
		CornerRefineSubpix,
		CornerRefineContour,
		CornerRefineAprilTag
	};

	//! Detection settings.
	struct Settings {
		std::string dictionary;				//!< Dictionary name, see DictionaryNames().
		double minMarkerPerimeterRate;		//!< Minimum marker perimeter, relative to the largest image dimension.
		CornerRefinement cornerRefinement;	//!< Corner refinement method.
		bool filterByID;					//!< Whether to discard markers outside [minID, maxID].
		int minID;							//!< Smallest accepted ID (if filterByID).
		int maxID;							//!< Largest accepted ID (if filterByID).
		bool keepRejected;					//!< Whether to keep the rejected candidates (for display).

		Settings(): dictionary("APRILTAG_36h11"), minMarkerPerimeterRate(0.03), cornerRefinement(CornerRefineSubpix),
			filterByID(false), minID(0), maxID(1000), keepRejected(false) {}
	};

	//! Constructor.
	MarkerDetector();
	//! Destructor.
	~MarkerDetector();

	//! Returns true if this build has an ArUco implementation.
	static bool IsAvailable();
	//! Returns a short description of the ArUco implementation used by this build.
	static const char * Backend();
	//! Returns the supported dictionary names.
	static std::vector<std::string> DictionaryNames();
	//! Parses a corner refinement name ("None", "Subpix", "Contour", "AprilTag"). Returns false if unknown.
	static bool ParseCornerRefinement(const std::string & name, CornerRefinement & result);

	//! Applies new settings. Returns false and sets error if the settings are invalid or ArUco is unavailable.
	bool Configure(const Settings & settings, std::string & error);
	//! Detects markers in an 8-bit grayscale image. Returns false and sets error on failure.
	bool Detect(const cv::Mat & gray, std::vector<Marker> & markers, std::string & error);
	//! The rejected candidates of the last Detect() call (only filled if Settings::keepRejected).
	const std::vector<std::vector<cv::Point2f> > & Rejected() const {
		return mRejected;
	}

	//! Computes center, orientation, area and bounding box from the 4 corners (clockwise, starting top-left).
	static Marker MakeMarker(int id, const std::vector<cv::Point2f> & corners);
	//! Draws markers (outline, ID, orientation arrow) and optionally rejected candidates on a BGR image.
	static void Draw(cv::Mat & image, const std::vector<Marker> & markers, const std::vector<std::vector<cv::Point2f> > & rejected);
	//! Renders the image of a marker (black/white, sidePixels x sidePixels). Returns false if unavailable.
	static bool GenerateMarkerImage(const std::string & dictionary, int id, int sidePixels, cv::Mat & image, std::string & error);

private:
	struct Impl;
	std::unique_ptr<Impl> mImpl;		//!< ArUco detector state (depends on the OpenCV version).
	Settings mSettings;					//!< The current settings.
	std::vector<std::vector<cv::Point2f> > mRejected;	//!< Rejected candidates of the last detection.
};

#endif
