// Renders synthetic images with markers at known positions and rotations, and checks MarkerDetector's output.
#include "MarkerDetector.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>

static int failures = 0;

static void Check(bool condition, const char * what, int id) {
	if (! condition) {
		std::printf("FAIL: %s (marker %d)\n", what, id);
		failures++;
	}
}

static double AngleDifference(double a, double b) {
	double d = std::fmod(a - b, 2 * CV_PI);
	if (d > CV_PI) d -= 2 * CV_PI;
	if (d < -CV_PI) d += 2 * CV_PI;
	return std::fabs(d);
}

// Geometry of MakeMarker, independent of ArUco: a square of side 20 centered at (50, 40), rotated by 90 degrees clockwise.
static void TestGeometry() {
	std::vector<cv::Point2f> corners = {{60, 30}, {60, 50}, {40, 50}, {40, 30}};
	MarkerDetector::Marker m = MarkerDetector::MakeMarker(7, corners);
	Check(m.id == 7, "geometry: id", 7);
	Check(std::fabs(m.center.x - 50) < 1e-4 && std::fabs(m.center.y - 40) < 1e-4, "geometry: center", 7);
	Check(AngleDifference(m.orientation, CV_PI / 2) < 1e-4, "geometry: orientation", 7);
	Check(std::fabs(m.area - 400) < 1e-3, "geometry: area", 7);
	Check(std::fabs(m.boundingBox.x - 40) < 1e-4 && std::fabs(m.boundingBox.y - 30) < 1e-4 &&
		std::fabs(m.boundingBox.width - 20) < 1e-4 && std::fabs(m.boundingBox.height - 20) < 1e-4, "geometry: bounding box", 7);
}

struct Placement {
	int id;
	float cx, cy;		// center [pixels]
	float side;			// side length [pixels]
	float angleDeg;		// rotation (clockwise on screen) [degrees]
};

static void TestDictionary(const std::string & dictionary, const std::vector<Placement> & placements, MarkerDetector::CornerRefinement refinement) {
	const int markerPixels = 160;
	cv::Mat canvas(720, 960, CV_8UC1, cv::Scalar(255));

	for (const Placement & p : placements) {
		cv::Mat markerImage;
		std::string error;
		if (! MarkerDetector::GenerateMarkerImage(dictionary, p.id, markerPixels, markerImage, error)) {
			std::printf("FAIL: %s\n", error.c_str());
			failures++;
			return;
		}

		// Similarity transform: pixel (i, j) covers [i - 0.5, i + 0.5], so the marker's outer edges are at -0.5 and
		// markerPixels - 0.5, and its center at (markerPixels - 1) / 2. Map that center to (cx, cy), scale to the requested
		// side length, and rotate clockwise (on screen) by angleDeg.
		double a = p.angleDeg * CV_PI / 180.0;
		double s = p.side / markerPixels;
		double c = std::cos(a) * s, d = std::sin(a) * s;
		double m0 = (markerPixels - 1) / 2.0;
		double values[6] = {c, -d, p.cx - (c * m0 - d * m0), d, c, p.cy - (d * m0 + c * m0)};
		cv::Mat transform(2, 3, CV_64F, values);
		cv::warpAffine(markerImage, canvas, transform, canvas.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
	}
	cv::GaussianBlur(canvas, canvas, cv::Size(3, 3), 0.7);

	MarkerDetector detector;
	MarkerDetector::Settings settings;
	settings.dictionary = dictionary;
	settings.cornerRefinement = refinement;
	std::string error;
	std::vector<MarkerDetector::Marker> markers;
	if (! detector.Configure(settings, error) || ! detector.Detect(canvas, markers, error)) {
		std::printf("FAIL: %s\n", error.c_str());
		failures++;
		return;
	}

	Check(markers.size() == placements.size(), "number of detected markers", -1);
	for (const Placement & p : placements) {
		const MarkerDetector::Marker * found = 0;
		for (const MarkerDetector::Marker & m : markers) {
			if (m.id == p.id) found = &m;
		}
		if (! found) {
			std::printf("FAIL: %s marker %d not detected\n", dictionary.c_str(), p.id);
			failures++;
			continue;
		}
		double dx = found->center.x - p.cx, dy = found->center.y - p.cy;
		double dAngle = AngleDifference(found->orientation, p.angleDeg * CV_PI / 180.0) * 180.0 / CV_PI;
		double expectedArea = p.side * p.side;
		std::printf("%-15s id %3d: center error %.3f px, orientation error %.3f deg, area %.0f (expected %.0f)\n",
			dictionary.c_str(), p.id, std::sqrt(dx * dx + dy * dy), dAngle, found->area, expectedArea);
		Check(std::sqrt(dx * dx + dy * dy) < 1.0, "center within 1 px", p.id);
		Check(dAngle < 3.0, "orientation within 3 degrees", p.id);
		Check(std::fabs(found->area - expectedArea) < 0.05 * expectedArea, "area within 5%", p.id);
		Check(found->boundingBox.contains(found->center), "bounding box contains center", p.id);
	}

	// ID filter
	settings.filterByID = true;
	settings.minID = placements[0].id;
	settings.maxID = placements[0].id;
	if (! detector.Configure(settings, error) || ! detector.Detect(canvas, markers, error)) {
		std::printf("FAIL: %s\n", error.c_str());
		failures++;
		return;
	}
	Check(markers.size() == 1 && markers[0].id == placements[0].id, "ID filter", placements[0].id);
}

int main() {
	TestGeometry();

	if (! MarkerDetector::IsAvailable()) {
		std::printf("Skipped: %s\n", MarkerDetector::Backend());
		return failures ? 1 : 0;
	}
	std::printf("Backend: %s\n", MarkerDetector::Backend());

	std::vector<Placement> placements = {
		{0, 150.0f, 150.0f, 120.0f, 0.0f},
		{17, 420.3f, 160.7f, 140.0f, 30.0f},
		{42, 740.0f, 170.0f, 130.0f, -45.0f},
		{123, 170.5f, 480.25f, 150.0f, 90.0f},
		{200, 470.0f, 500.0f, 110.0f, 135.0f},
		{301, 760.8f, 520.4f, 160.0f, -170.0f},
	};
	TestDictionary("APRILTAG_36h11", placements, MarkerDetector::CornerRefineSubpix);

	// ArUco 4x4_50 only has IDs 0..49
	std::vector<Placement> arucoPlacements = placements;
	for (size_t i = 0; i < arucoPlacements.size(); i++) arucoPlacements[i].id = (int)(i * 7);
	TestDictionary("4X4_50", arucoPlacements, MarkerDetector::CornerRefineContour);

	if (failures) {
		std::printf("%d check(s) failed\n", failures);
		return 1;
	}
	std::printf("All marker detection checks passed\n");
	return 0;
}
