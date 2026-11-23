// Tests the ObjectDetectorDNN output decoders, letterbox mapping and NMS on synthetic network outputs
// (no model file needed). Each format is checked by placing known boxes (in original image coordinates)
// into a fake output tensor, and verifying that they are recovered after decoding, NMS and mapping back.
#include "ObjectDetectorDNN.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static int gFailures = 0;
static int gChecks = 0;

#define CHECK(cond) do { gChecks++; if (! (cond)) { gFailures++; std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)

static bool Near(float a, float b, float tol = 0.05f) {
	return std::fabs(a - b) <= tol;
}

static bool NearBox(const cv::Rect2f & a, const cv::Rect2f & b, float tol = 0.05f) {
	return Near(a.x, b.x, tol) && Near(a.y, b.y, tol) && Near(a.width, b.width, tol) && Near(a.height, b.height, tol);
}

typedef ObjectDetectorDNN DD;

// A box in original image coordinates, with the score and class the fake network should output.
struct Truth {
	cv::Rect2f box;
	float score;
	int classId;
	int anchor;		// Where to put it in the output tensor
};

static const cv::Size scOriginal(1280, 720);
static const cv::Size scInput(640, 640);
static const int scClasses = 3;

// A: strong box; B: near-duplicate of A (same class, suppressed by NMS); C: weak box (kept with min score 0.1);
// D: same place as A but another class (kept by class-aware NMS only); E: below min score.
static std::vector<Truth> MakeTruth(int numAnchors) {
	return {
		{cv::Rect2f(100, 200, 300, 150), 0.90f, 2, 10},
		{cv::Rect2f(104, 204, 300, 150), 0.60f, 2, numAnchors / 3},
		{cv::Rect2f(700, 300, 200, 200), 0.15f, 0, numAnchors / 2},
		{cv::Rect2f(100, 200, 300, 150), 0.80f, 1, numAnchors - 100},
		{cv::Rect2f(1000, 500, 100, 100), 0.05f, 0, numAnchors - 1},
	};
}

static cv::Rect2f ToInput(const cv::Rect2f & r, const DD::Letterbox & lb) {
	cv::Point2f tl = lb.ToInput(r.tl());
	cv::Point2f br = lb.ToInput(r.br());
	return cv::Rect2f(tl, br);
}

// Sets attribute a of anchor i, for an anchors-first (N x K) or attributes-first (K x N) matrix stored in out.
static void Set(cv::Mat & out, bool attributesFirst, int n, int k, int i, int a, float value) {
	float * data = out.ptr<float>();
	if (attributesFirst) data[(size_t)a * n + i] = value;
	else data[(size_t)i * k + a] = value;
}

static cv::Mat MakeTensor(std::vector<int> shape) {
	cv::Mat m((int)shape.size(), shape.data(), CV_32F, cv::Scalar(0));
	return m;
}

// YOLOv8: (4+C) x N, or N x (4+C) if transposed
static cv::Mat MakeYOLOv8(const std::vector<Truth> & truth, int n, bool transposed, bool squeeze) {
	int k = 4 + scClasses;
	cv::Mat out = transposed ? (squeeze ? MakeTensor({n, k}) : MakeTensor({1, n, k})) : (squeeze ? MakeTensor({k, n}) : MakeTensor({1, k, n}));
	DD::Letterbox lb = DD::Letterbox::Compute(scOriginal, scInput, true);
	for (const Truth & t : truth) {
		cv::Rect2f r = ToInput(t.box, lb);
		Set(out, ! transposed, n, k, t.anchor, 0, r.x + r.width / 2);
		Set(out, ! transposed, n, k, t.anchor, 1, r.y + r.height / 2);
		Set(out, ! transposed, n, k, t.anchor, 2, r.width);
		Set(out, ! transposed, n, k, t.anchor, 3, r.height);
		Set(out, ! transposed, n, k, t.anchor, 4 + t.classId, t.score);
		// A low score for another class, which must neither win nor pass the min. score of 0.1
		Set(out, ! transposed, n, k, t.anchor, 4 + (t.classId + 1) % scClasses, std::min(t.score / 2, 0.05f));
	}
	return out;
}

// YOLOv5: N x (5+C), score = objectness * class probability
static cv::Mat MakeYOLOv5(const std::vector<Truth> & truth, int n) {
	int k = 5 + scClasses;
	cv::Mat out = MakeTensor({1, n, k});
	DD::Letterbox lb = DD::Letterbox::Compute(scOriginal, scInput, true);
	for (const Truth & t : truth) {
		cv::Rect2f r = ToInput(t.box, lb);
		Set(out, false, n, k, t.anchor, 0, r.x + r.width / 2);
		Set(out, false, n, k, t.anchor, 1, r.y + r.height / 2);
		Set(out, false, n, k, t.anchor, 2, r.width);
		Set(out, false, n, k, t.anchor, 3, r.height);
		Set(out, false, n, k, t.anchor, 4, std::sqrt(t.score));
		Set(out, false, n, k, t.anchor, 5 + t.classId, std::sqrt(t.score));
	}
	return out;
}

// YOLOX: N x (5+C) raw grid outputs (offsets relative to the cell, log sizes), top-left letterbox
static cv::Mat MakeYOLOX(const std::vector<Truth> & truth, int n, std::vector<int> shape) {
	int k = 5 + scClasses;
	cv::Mat out = MakeTensor(shape);
	DD::Letterbox lb = DD::Letterbox::Compute(scOriginal, scInput, false);
	const int strides[] = {8, 16, 32};
	for (const Truth & t : truth) {
		// Find the cell of this anchor
		int start = 0, stride = 0, cell = 0, width = 0;
		for (int s : strides) {
			int w = scInput.width / s, h = scInput.height / s;
			if (t.anchor >= start && t.anchor < start + w * h) {
				stride = s;
				cell = t.anchor - start;
				width = w;
			}
			start += w * h;
		}
		int gx = cell % width, gy = cell / width;
		cv::Rect2f r = ToInput(t.box, lb);
		Set(out, false, n, k, t.anchor, 0, (r.x + r.width / 2) / stride - gx);
		Set(out, false, n, k, t.anchor, 1, (r.y + r.height / 2) / stride - gy);
		Set(out, false, n, k, t.anchor, 2, std::log(r.width / stride));
		Set(out, false, n, k, t.anchor, 3, std::log(r.height / stride));
		Set(out, false, n, k, t.anchor, 4, 1.0f);
		Set(out, false, n, k, t.anchor, 5 + t.classId, t.score);
	}
	return out;
}

static std::vector<DD::Detection> Run(const cv::Mat & out, DD::Format format, const DD::Thresholds & th, bool centered) {
	std::vector<DD::Detection> dets;
	std::string error;
	bool ok = DD::Decode(out, format, scInput, th, dets, error);
	if (! ok) std::printf("  decode error: %s\n", error.c_str());
	CHECK(ok);
	DD::NonMaximumSuppression(dets, th);
	DD::MapToOriginal(dets, DD::Letterbox::Compute(scOriginal, scInput, centered));
	return dets;
}

// Checks the expected results for the truth set from MakeTruth under various thresholds.
static void CheckFormat(const char * name, const cv::Mat & out, DD::Format format, int n, bool centered) {
	std::printf("%s (%s)\n", name, DD::FormatName(format));
	std::vector<Truth> truth = MakeTruth(n);
	DD::Thresholds th;
	th.minScore = 0.1f;
	th.nmsIoU = 0.45f;

	// Class-aware NMS: A, D, C
	th.classAware = true;
	std::vector<DD::Detection> dets = Run(out, format, th, centered);
	CHECK(dets.size() == 3);
	if (dets.size() == 3) {
		CHECK(NearBox(dets[0].box, truth[0].box) && Near(dets[0].score, 0.90f, 1e-4f) && dets[0].classId == 2);
		CHECK(NearBox(dets[1].box, truth[3].box) && Near(dets[1].score, 0.80f, 1e-4f) && dets[1].classId == 1);
		CHECK(NearBox(dets[2].box, truth[2].box) && Near(dets[2].score, 0.15f, 1e-4f) && dets[2].classId == 0);
	}

	// Class-agnostic NMS: A, C
	th.classAware = false;
	dets = Run(out, format, th, centered);
	CHECK(dets.size() == 2);
	if (dets.size() == 2) {
		CHECK(NearBox(dets[0].box, truth[0].box) && dets[0].classId == 2);
		CHECK(NearBox(dets[1].box, truth[2].box) && dets[1].classId == 0);
	}

	// Higher min. score: A, D
	th.classAware = true;
	th.minScore = 0.2f;
	dets = Run(out, format, th, centered);
	CHECK(dets.size() == 2);
	if (dets.size() == 2) CHECK(dets[0].classId == 2 && dets[1].classId == 1);

	// Min. score low enough for E as well
	th.minScore = 0.01f;
	dets = Run(out, format, th, centered);
	CHECK(dets.size() == 4);
	if (dets.size() == 4) CHECK(NearBox(dets[3].box, truth[4].box) && Near(dets[3].score, 0.05f, 1e-4f));

	// Class filter: only class 0 (C), and only class 1 (D, which is no longer suppressed by A)
	th.minScore = 0.1f;
	th.classes = {0};
	dets = Run(out, format, th, centered);
	CHECK(dets.size() == 1 && dets[0].classId == 0 && NearBox(dets[0].box, truth[2].box));
	th.classes = {1};
	th.classAware = false;
	dets = Run(out, format, th, centered);
	CHECK(dets.size() == 1 && dets[0].classId == 1 && NearBox(dets[0].box, truth[3].box));
	th.classes.clear();

	// Max. detections
	th.classAware = true;
	th.maxDetections = 1;
	dets = Run(out, format, th, centered);
	CHECK(dets.size() == 1 && dets[0].classId == 2);
}

static void TestLetterbox() {
	std::printf("Letterbox\n");
	DD::Letterbox lb = DD::Letterbox::Compute(cv::Size(1280, 720), cv::Size(640, 640), true);
	CHECK(Near(lb.scale, 0.5f, 1e-6f) && lb.padX == 0 && lb.padY == 140 && lb.scaled == cv::Size(640, 360));
	lb = DD::Letterbox::Compute(cv::Size(480, 960), cv::Size(640, 640), true);
	CHECK(Near(lb.scale, 2.f / 3.f, 1e-6f) && lb.padX == 160 && lb.padY == 0 && lb.scaled == cv::Size(320, 640));
	lb = DD::Letterbox::Compute(cv::Size(1280, 720), cv::Size(640, 640), false);
	CHECK(lb.padX == 0 && lb.padY == 0);
	lb = DD::Letterbox::Compute(cv::Size(641, 479), cv::Size(320, 256), true);
	for (float x : {0.f, 13.5f, 640.f})
		for (float y : {0.f, 77.25f, 478.f}) {
			cv::Point2f p = lb.ToOriginal(lb.ToInput(cv::Point2f(x, y)));
			CHECK(Near(p.x, x, 1e-3f) && Near(p.y, y, 1e-3f));
		}

	// Boxes are clipped to the original image
	lb = DD::Letterbox::Compute(cv::Size(1280, 720), cv::Size(640, 640), true);
	cv::Rect2f clipped = lb.ToOriginal(cv::Rect2f(-10, 130, 60, 60));
	CHECK(NearBox(clipped, cv::Rect2f(0, 0, 100, 100)));
	std::vector<DD::Detection> dets = {{cv::Rect2f(10, 10, 20, 20), 0.9f, 0}, {cv::Rect2f(100, 300, 50, 50), 0.8f, 0}};
	DD::MapToOriginal(dets, lb);	// The first one lies entirely in the padding
	CHECK(dets.size() == 1 && NearBox(dets[0].box, cv::Rect2f(200, 320, 100, 100)));

	// Letterboxed image
	cv::Mat image(720, 1280, CV_8UC3, cv::Scalar(200, 100, 50));
	cv::Mat dst;
	DD::LetterboxImage(image, dst, DD::Letterbox::Compute(image.size(), cv::Size(640, 640), true));
	CHECK(dst.size() == cv::Size(640, 640) && dst.type() == CV_8UC3);
	CHECK(dst.at<cv::Vec3b>(0, 0) == cv::Vec3b(114, 114, 114) && dst.at<cv::Vec3b>(639, 639) == cv::Vec3b(114, 114, 114));
	CHECK(dst.at<cv::Vec3b>(140, 0) == cv::Vec3b(200, 100, 50) && dst.at<cv::Vec3b>(499, 639) == cv::Vec3b(200, 100, 50));
	CHECK(dst.at<cv::Vec3b>(139, 320) == cv::Vec3b(114, 114, 114) && dst.at<cv::Vec3b>(500, 320) == cv::Vec3b(114, 114, 114));
	uchar * buffer = dst.data;
	DD::LetterboxImage(image, dst, DD::Letterbox::Compute(image.size(), cv::Size(640, 640), false));
	CHECK(dst.data == buffer);	// Buffer reused
	CHECK(dst.at<cv::Vec3b>(0, 0) == cv::Vec3b(200, 100, 50) && dst.at<cv::Vec3b>(359, 639) == cv::Vec3b(200, 100, 50));
	CHECK(dst.at<cv::Vec3b>(360, 0) == cv::Vec3b(114, 114, 114));
}

static void TestInferFormat() {
	std::printf("Format inference\n");
	std::string message;
	bool assumed = true;
	CHECK(DD::InferFormat(MakeTensor({1, 84, 8400}), scInput, 0, message, &assumed) == DD::FormatYOLOv8 && ! assumed);
	CHECK(DD::InferFormat(MakeTensor({84, 8400}), scInput, 0, message) == DD::FormatYOLOv8);
	CHECK(DD::InferFormat(MakeTensor({1, 5, 8400}), scInput, 0, message) == DD::FormatYOLOv8);	// Single class
	CHECK(DD::InferFormat(MakeTensor({1, 25200, 85}), scInput, 0, message) == DD::FormatYOLOv5);
	CHECK(DD::InferFormat(MakeTensor({1, 1, 25200, 85}), scInput, 0, message) == DD::FormatYOLOv5);
	CHECK(DD::InferFormat(MakeTensor({1, 3 * (6400 + 1600 + 400 + 100), 6}), scInput, 0, message) == DD::FormatYOLOv5);	// P6 model
	CHECK(DD::InferFormat(MakeTensor({1, 8400, 85}), scInput, 0, message, &assumed) == DD::FormatYOLOX && assumed);
	CHECK(DD::InferFormat(MakeTensor({1, 8400, 85}), scInput, 80, message, &assumed) == DD::FormatYOLOX && ! assumed);
	CHECK(DD::InferFormat(MakeTensor({1, 8400, 84}), scInput, 80, message, &assumed) == DD::FormatYOLOv8 && ! assumed);
	CHECK(DD::InferFormat(MakeTensor({1, 2100, 85}), cv::Size(320, 320), 0, message) == DD::FormatYOLOX);
	CHECK(DD::InferFormat(MakeTensor({1, 1000, 85}), scInput, 0, message) == DD::FormatUnknown && ! message.empty());
	CHECK(DD::InferFormat(MakeTensor({2, 8400, 85}), scInput, 0, message) == DD::FormatUnknown);	// Batch of 2
	CHECK(DD::InferFormat(cv::Mat(), scInput, 0, message) == DD::FormatUnknown);

	CHECK(DD::NumClasses(MakeTensor({1, 84, 8400}), DD::FormatYOLOv8) == 80);
	CHECK(DD::NumClasses(MakeTensor({1, 25200, 85}), DD::FormatYOLOv5) == 80);
	CHECK(DD::NumClasses(MakeTensor({1, 4, 8400}), DD::FormatYOLOv8) == 0);

	// Mismatching shapes are rejected by the decoder
	std::vector<DD::Detection> dets;
	std::string error;
	CHECK(! DD::Decode(MakeTensor({1, 8000, 85}), DD::FormatYOLOX, scInput, DD::Thresholds(), dets, error) && ! error.empty());
	CHECK(! DD::Decode(MakeTensor({1, 8400, 5}), DD::FormatYOLOv5, scInput, DD::Thresholds(), dets, error));
	CHECK(! DD::Decode(MakeTensor({1, 8400, 85}), DD::FormatAuto, scInput, DD::Thresholds(), dets, error));
}

static void TestConfigHelpers() {
	std::printf("Configuration helpers\n");
	std::vector<int> classes;
	CHECK(DD::ParseClassList("0, 2 5;17", classes) && classes == std::vector<int>({0, 2, 5, 17}));
	CHECK(DD::ParseClassList("  ", classes) && classes.empty());
	CHECK(! DD::ParseClassList("0, person", classes));
	CHECK(! DD::ParseClassList("-1", classes));

	const char * path = "TestObjectDetectorDNN_classes.txt";
	{
		std::ofstream f(path, std::ios::binary);
		f << "person\r\n  car \r\n\r\nbicycle\n\n\n";
	}
	std::vector<std::string> names;
	CHECK(DD::ReadClassNames(path, names));
	CHECK(names == std::vector<std::string>({"person", "car", "", "bicycle"}));
	std::remove(path);
	CHECK(! DD::ReadClassNames("does/not/exist.txt", names));
}

static void TestLoadErrors() {
	std::printf("Load errors\n");
	DD detector;
	DD::Config config;
	std::string error;
	std::vector<std::string> warnings;
	CHECK(! detector.Load(config, error, warnings) && ! error.empty() && ! detector.IsLoaded());
	config.modelPath = "does/not/exist.onnx";
	error.clear();
	CHECK(! detector.Load(config, error, warnings) && ! error.empty() && ! detector.IsLoaded());
	std::vector<DD::Detection> dets;
	CHECK(! detector.Detect(cv::Mat(10, 10, CV_8UC3), DD::Thresholds(), dets, error));
}

int main() {
	if (! DD::IsAvailable()) {
		std::printf("OpenCV dnn not available\n");
		return 0;
	}

	TestLetterbox();
	TestInferFormat();
	TestConfigHelpers();
	TestLoadErrors();

	const int n = 8400;			// Anchor-free models, 640x640: 80*80 + 40*40 + 20*20
	const int n5 = 25200;		// YOLOv5: 3 anchors per cell
	CheckFormat("YOLOv8 1x(4+C)xN", MakeYOLOv8(MakeTruth(n), n, false, false), DD::FormatYOLOv8, n, true);
	CheckFormat("YOLOv8 (4+C)xN (2D)", MakeYOLOv8(MakeTruth(n), n, false, true), DD::FormatYOLOv8, n, true);
	CheckFormat("YOLOv8 1xNx(4+C) (transposed)", MakeYOLOv8(MakeTruth(n), n, true, false), DD::FormatYOLOv8, n, true);
	CheckFormat("YOLOv5 1xNx(5+C)", MakeYOLOv5(MakeTruth(n5), n5), DD::FormatYOLOv5, n5, true);
	CheckFormat("YOLOX 1xNx(5+C)", MakeYOLOX(MakeTruth(n), n, {1, n, 5 + scClasses}), DD::FormatYOLOX, n, false);
	CheckFormat("YOLOX NxK (2D)", MakeYOLOX(MakeTruth(n), n, {n, 5 + scClasses}), DD::FormatYOLOX, n, false);

	// The formats of the synthetic outputs are inferred correctly
	std::string message;
	CHECK(DD::InferFormat(MakeYOLOv8(MakeTruth(n), n, false, false), scInput, 0, message) == DD::FormatYOLOv8);
	CHECK(DD::InferFormat(MakeYOLOv8(MakeTruth(n), n, true, false), scInput, scClasses, message) == DD::FormatYOLOv8);
	CHECK(DD::InferFormat(MakeYOLOv5(MakeTruth(n5), n5), scInput, 0, message) == DD::FormatYOLOv5);
	CHECK(DD::InferFormat(MakeYOLOX(MakeTruth(n), n, {1, n, 5 + scClasses}), scInput, 0, message) == DD::FormatYOLOX);

	std::printf("%d checks, %d failures\n", gChecks, gFailures);
	return gFailures == 0 ? 0 : 1;
}
