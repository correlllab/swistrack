// Tests BlobSplitter: touching ellipses must be split into the right number of detections at the right places.
#include "BlobSplitter.h"
#include <cstdio>
#include <opencv2/imgproc.hpp>

static int sFailures = 0;
#define CHECK(condition, ...) do { if (! (condition)) { sFailures++; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static void DrawFish(cv::Mat & image, cv::Point2f center, double angleDeg) {
	cv::ellipse(image, cv::RotatedRect(center, cv::Size2f(30, 10), (float)angleDeg), cv::Scalar(255), cv::FILLED);
}

static bool HasDetectionNear(const std::vector<BlobSplitter::Blob> & blobs, cv::Point2f p, double tolerance) {
	for (auto & b : blobs) if (cv::norm(b.center - p) < tolerance) return true;
	return false;
}

int main() {
	BlobSplitter splitter;

	// Learn the object area from frames with isolated objects
	for (int f = 0; f < 10; f++) {
		cv::Mat image = cv::Mat::zeros(240, 320, CV_8UC1);
		for (int k = 0; k < 4; k++) DrawFish(image, cv::Point2f(40.0f + 70 * k, 60.0f + f), 20.0 * k);
		auto blobs = splitter.Detect(image);
		CHECK(blobs.size() == 4, "isolated: %zu blobs", blobs.size());
		for (auto & b : blobs) CHECK(b.splitCount == 1 && b.score == 1.0f, "isolated blob was split");
	}
	double area = splitter.GetObjectArea();
	CHECK(area > 230 && area < 300, "learned object area %.1f (expected ~270 incl. boundary pixels)", area);

	// Two fish side by side, touching; three fish in a row; plus one isolated fish
	cv::Mat image = cv::Mat::zeros(240, 320, CV_8UC1);
	DrawFish(image, cv::Point2f(60, 60), 0);
	DrawFish(image, cv::Point2f(60, 68), 0);
	DrawFish(image, cv::Point2f(150, 150), 0);
	DrawFish(image, cv::Point2f(178, 150), 0);
	DrawFish(image, cv::Point2f(206, 150), 0);
	DrawFish(image, cv::Point2f(270, 40), 45);
	auto blobs = splitter.Detect(image);
	CHECK(blobs.size() == 6, "touching: %zu detections (expected 6)", blobs.size());
	CHECK(HasDetectionNear(blobs, cv::Point2f(60, 60), 3) && HasDetectionNear(blobs, cv::Point2f(60, 68), 3), "pair not split at the right places");
	CHECK(HasDetectionNear(blobs, cv::Point2f(150, 150), 5) && HasDetectionNear(blobs, cv::Point2f(178, 150), 5) && HasDetectionNear(blobs, cv::Point2f(206, 150), 5), "triple not split at the right places");
	int split = 0;
	for (auto & b : blobs) if (b.splitCount > 1) { split++; CHECK(b.score < 1.0f, "split detection has full score"); }
	CHECK(split == 5, "%d split detections (expected 5)", split);

	// Fixed object area and the max. objects per blob limit
	BlobSplitter::Parameters p;
	p.objectArea = 235;
	p.maxObjectsPerBlob = 2;
	splitter.SetParameters(p);
	blobs = splitter.Detect(image);
	CHECK(blobs.size() == 5, "limit: %zu detections (expected 5)", blobs.size());

	std::printf("BlobSplitter: %d failures\n", sFailures);
	return sFailures == 0 ? 0 : 1;
}
