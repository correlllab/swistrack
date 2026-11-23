#include "MarkerDetector.h"
#define THISCLASS MarkerDetector

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <opencv2/imgproc.hpp>

#if defined(SWISTRACK_MARKERS_OBJDETECT)
#include <opencv2/objdetect/aruco_detector.hpp>
#elif defined(SWISTRACK_MARKERS_CONTRIB)
#include <opencv2/aruco.hpp>
#endif

namespace {

struct DictionaryEntry {
	const char * name;
	int id;		// cv::aruco::PredefinedDictionaryType / PREDEFINED_DICTIONARY_NAME
};

#if defined(SWISTRACK_MARKERS_OBJDETECT) || defined(SWISTRACK_MARKERS_CONTRIB)
const DictionaryEntry kDictionaries[] = {
	{"APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11},
	{"APRILTAG_25h9", cv::aruco::DICT_APRILTAG_25h9},
	{"APRILTAG_16h5", cv::aruco::DICT_APRILTAG_16h5},
	{"APRILTAG_36h10", cv::aruco::DICT_APRILTAG_36h10},
	{"4X4_50", cv::aruco::DICT_4X4_50},
	{"4X4_250", cv::aruco::DICT_4X4_250},
	{"5X5_100", cv::aruco::DICT_5X5_100},
	{"6X6_250", cv::aruco::DICT_6X6_250},
	{"ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
};
#else
const DictionaryEntry kDictionaries[] = {
	{"APRILTAG_36h11", 0}, {"APRILTAG_25h9", 0}, {"APRILTAG_16h5", 0}, {"APRILTAG_36h10", 0},
	{"4X4_50", 0}, {"4X4_250", 0}, {"5X5_100", 0}, {"6X6_250", 0}, {"ARUCO_ORIGINAL", 0},
};
#endif

bool FindDictionary(const std::string & name, int & id) {
	for (const DictionaryEntry & entry : kDictionaries) {
		if (name == entry.name) {
			id = entry.id;
			return true;
		}
	}
	return false;
}

const char * kUnavailable = "SwisTrack was built without ArUco support (requires OpenCV >= 4.7 with objdetect, or the opencv_contrib aruco module).";

#if defined(SWISTRACK_MARKERS_OBJDETECT) || defined(SWISTRACK_MARKERS_CONTRIB)
int CornerRefineMethod(MarkerDetector::CornerRefinement refinement) {
	switch (refinement) {
	case MarkerDetector::CornerRefineSubpix:
		return cv::aruco::CORNER_REFINE_SUBPIX;
	case MarkerDetector::CornerRefineContour:
		return cv::aruco::CORNER_REFINE_CONTOUR;
	case MarkerDetector::CornerRefineAprilTag:
		return cv::aruco::CORNER_REFINE_APRILTAG;
	default:
		return cv::aruco::CORNER_REFINE_NONE;
	}
}

template <typename Parameters>
void ApplySettings(Parameters & parameters, const MarkerDetector::Settings & settings) {
	parameters.minMarkerPerimeterRate = settings.minMarkerPerimeterRate;
	parameters.cornerRefinementMethod = CornerRefineMethod(settings.cornerRefinement);
}
#endif

}

struct THISCLASS::Impl {
#if defined(SWISTRACK_MARKERS_OBJDETECT)
	cv::aruco::ArucoDetector detector;
#elif defined(SWISTRACK_MARKERS_CONTRIB)
	cv::Ptr<cv::aruco::Dictionary> dictionary;
	cv::Ptr<cv::aruco::DetectorParameters> parameters;
#endif
	bool configured = false;
};

THISCLASS::MarkerDetector(): mImpl(new Impl()), mSettings(), mRejected() {
}

THISCLASS::~MarkerDetector() {
}

bool THISCLASS::IsAvailable() {
#if defined(SWISTRACK_MARKERS_OBJDETECT) || defined(SWISTRACK_MARKERS_CONTRIB)
	return true;
#else
	return false;
#endif
}

const char * THISCLASS::Backend() {
#if defined(SWISTRACK_MARKERS_OBJDETECT)
	return "OpenCV " CV_VERSION " objdetect (cv::aruco::ArucoDetector)";
#elif defined(SWISTRACK_MARKERS_CONTRIB)
	return "OpenCV " CV_VERSION " contrib aruco (legacy API)";
#else
	return "none (built without ArUco support)";
#endif
}

std::vector<std::string> THISCLASS::DictionaryNames() {
	std::vector<std::string> names;
	for (const DictionaryEntry & entry : kDictionaries) {
		names.push_back(entry.name);
	}
	return names;
}

bool THISCLASS::ParseCornerRefinement(const std::string & name, CornerRefinement & result) {
	if (name == "None") result = CornerRefineNone;
	else if (name == "Subpix") result = CornerRefineSubpix;
	else if (name == "Contour") result = CornerRefineContour;
	else if (name == "AprilTag") result = CornerRefineAprilTag;
	else return false;
	return true;
}

bool THISCLASS::Configure(const Settings & settings, std::string & error) {
	mImpl->configured = false;
	if (! IsAvailable()) {
		error = kUnavailable;
		return false;
	}

	int dictionaryId = 0;
	if (! FindDictionary(settings.dictionary, dictionaryId)) {
		error = "Unknown marker dictionary '" + settings.dictionary + "'.";
		return false;
	}
	if (settings.minMarkerPerimeterRate <= 0 || settings.minMarkerPerimeterRate > 4) {
		error = "The minimum marker perimeter rate must be in (0, 4].";
		return false;
	}
	if (settings.filterByID && settings.minID > settings.maxID) {
		error = "The minimum marker ID must not be larger than the maximum marker ID.";
		return false;
	}

#if defined(SWISTRACK_MARKERS_OBJDETECT)
	cv::aruco::DetectorParameters parameters;
	ApplySettings(parameters, settings);
	mImpl->detector = cv::aruco::ArucoDetector(cv::aruco::getPredefinedDictionary(dictionaryId), parameters);
#elif defined(SWISTRACK_MARKERS_CONTRIB)
	mImpl->dictionary = cv::aruco::getPredefinedDictionary(dictionaryId);
	mImpl->parameters = cv::aruco::DetectorParameters::create();
	ApplySettings(*mImpl->parameters, settings);
#endif

	mSettings = settings;
	mImpl->configured = true;
	return true;
}

bool THISCLASS::Detect(const cv::Mat & gray, std::vector<Marker> & markers, std::string & error) {
	markers.clear();
	mRejected.clear();
	if (! IsAvailable()) {
		error = kUnavailable;
		return false;
	}
	if (! mImpl->configured) {
		error = "The marker detector is not configured.";
		return false;
	}
	if (gray.empty() || gray.type() != CV_8UC1) {
		error = "Marker detection requires an 8-bit grayscale image.";
		return false;
	}

	std::vector<std::vector<cv::Point2f> > corners;
	std::vector<int> ids;
#if defined(SWISTRACK_MARKERS_OBJDETECT)
	if (mSettings.keepRejected) {
		mImpl->detector.detectMarkers(gray, corners, ids, mRejected);
	} else {
		mImpl->detector.detectMarkers(gray, corners, ids);
	}
#elif defined(SWISTRACK_MARKERS_CONTRIB)
	if (mSettings.keepRejected) {
		cv::aruco::detectMarkers(gray, mImpl->dictionary, corners, ids, mImpl->parameters, mRejected);
	} else {
		cv::aruco::detectMarkers(gray, mImpl->dictionary, corners, ids, mImpl->parameters);
	}
#endif

	for (size_t i = 0; i < ids.size() && i < corners.size(); i++) {
		if (corners[i].size() != 4) continue;
		if (mSettings.filterByID && (ids[i] < mSettings.minID || ids[i] > mSettings.maxID)) continue;
		markers.push_back(MakeMarker(ids[i], corners[i]));
	}
	return true;
}

THISCLASS::Marker THISCLASS::MakeMarker(int id, const std::vector<cv::Point2f> & corners) {
	Marker marker;
	marker.id = id;
	marker.corners = corners;
	if (corners.size() != 4) return marker;

	// Center: mean of the corners
	marker.center = (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25f;

	// Orientation: the marker's x-axis, averaged over the top (0 -> 1) and bottom (3 -> 2) edges
	cv::Point2f xAxis = (corners[1] - corners[0]) + (corners[2] - corners[3]);
	marker.orientation = std::atan2(xAxis.y, xAxis.x);

	// Area (shoelace formula) and bounding box
	double twiceArea = 0;
	for (size_t i = 0; i < 4; i++) {
		const cv::Point2f & a = corners[i];
		const cv::Point2f & b = corners[(i + 1) % 4];
		twiceArea += (double)a.x * b.y - (double)b.x * a.y;
	}
	marker.area = std::fabs(twiceArea) * 0.5;
	float x1 = corners[0].x, x2 = corners[0].x, y1 = corners[0].y, y2 = corners[0].y;
	for (const cv::Point2f & p : corners) {
		x1 = std::min(x1, p.x);
		x2 = std::max(x2, p.x);
		y1 = std::min(y1, p.y);
		y2 = std::max(y2, p.y);
	}
	marker.boundingBox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
	return marker;
}

void THISCLASS::Draw(cv::Mat & image, const std::vector<Marker> & markers, const std::vector<std::vector<cv::Point2f> > & rejected) {
	const int shift = 4;	// fixed-point bits for sub-pixel drawing
	const float scale = (float)(1 << shift);
	auto fixedPoint = [scale](const cv::Point2f & p) {
		return cv::Point(cvRound(p.x * scale), cvRound(p.y * scale));
	};

	// Rejected candidates (thin red outlines)
	for (const std::vector<cv::Point2f> & candidate : rejected) {
		std::vector<cv::Point> polygon;
		for (const cv::Point2f & p : candidate) polygon.push_back(fixedPoint(p));
		const cv::Point * pts = polygon.data();
		int npts = (int)polygon.size();
		cv::polylines(image, &pts, &npts, 1, true, cv::Scalar(0, 0, 255), 1, cv::LINE_AA, shift);
	}

	for (const Marker & marker : markers) {
		if (marker.corners.size() != 4) continue;

		// Outline (green), first corner (blue square)
		std::vector<cv::Point> polygon;
		for (const cv::Point2f & p : marker.corners) polygon.push_back(fixedPoint(p));
		const cv::Point * pts = polygon.data();
		int npts = 4;
		cv::polylines(image, &pts, &npts, 1, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA, shift);
		cv::Point first(cvRound(marker.corners[0].x), cvRound(marker.corners[0].y));
		cv::rectangle(image, first - cv::Point(3, 3), first + cv::Point(3, 3), cv::Scalar(255, 0, 0), 2);

		// Orientation arrow (yellow), from the center to the middle of the right edge
		float halfSide = (float)std::sqrt(std::max(marker.area, 1.0)) * 0.5f;
		cv::Point2f tip = marker.center + cv::Point2f(std::cos(marker.orientation), std::sin(marker.orientation)) * halfSide;
		cv::arrowedLine(image, fixedPoint(marker.center), fixedPoint(tip), cv::Scalar(0, 255, 255), 2, cv::LINE_AA, shift, 0.25);

		// ID (magenta), above the bounding box
		char text[16];
		std::snprintf(text, sizeof(text), "%d", marker.id);
		cv::putText(image, text, cv::Point(cvRound(marker.boundingBox.x), cvRound(marker.boundingBox.y) - 4),
			cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
	}
}

bool THISCLASS::GenerateMarkerImage(const std::string & dictionary, int id, int sidePixels, cv::Mat & image, std::string & error) {
	if (! IsAvailable()) {
		error = kUnavailable;
		return false;
	}
	int dictionaryId = 0;
	if (! FindDictionary(dictionary, dictionaryId)) {
		error = "Unknown marker dictionary '" + dictionary + "'.";
		return false;
	}
#if defined(SWISTRACK_MARKERS_OBJDETECT)
	cv::aruco::generateImageMarker(cv::aruco::getPredefinedDictionary(dictionaryId), id, sidePixels, image, 1);
#elif defined(SWISTRACK_MARKERS_CONTRIB)
	cv::aruco::drawMarker(cv::aruco::getPredefinedDictionary(dictionaryId), id, sidePixels, image, 1);
#else
	(void) id;
	(void) sidePixels;
	(void) image;
#endif
	return true;
}
