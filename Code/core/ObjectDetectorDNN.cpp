#include "ObjectDetectorDNN.h"
#define THISCLASS ObjectDetectorDNN

#include <opencv2/imgproc.hpp>
#if SWISTRACK_HAVE_DNN
#include <opencv2/dnn.hpp>
#endif
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace {

// A 2D network output seen as N anchors x K attributes, whatever its memory layout.
struct OutputView {
	const float * data;
	int n;					// Number of anchors (candidate boxes)
	int k;					// Number of attributes per anchor
	size_t strideAnchor;
	size_t strideAttribute;

	float At(int anchor, int attribute) const {
		return data[anchor * strideAnchor + attribute * strideAttribute];
	}
};

// The attribute dimension is the smaller one: YOLOv8 outputs (4+C) x N, YOLOv5/YOLOX output N x (5+C).
OutputView MakeView(const cv::Mat & matrix) {
	OutputView v;
	v.data = matrix.ptr<float>();
	if (matrix.rows < matrix.cols) {
		v.k = matrix.rows;
		v.n = matrix.cols;
		v.strideAnchor = 1;
		v.strideAttribute = matrix.cols;
	} else {
		v.k = matrix.cols;
		v.n = matrix.rows;
		v.strideAnchor = matrix.cols;
		v.strideAttribute = 1;
	}
	return v;
}

// A level of the YOLOX feature pyramid.
struct GridLevel {
	int start;
	int width;
	int height;
	int stride;
};

int MakeGrid(cv::Size inputSize, const int * strides, int count, std::vector<GridLevel> * levels) {
	int total = 0;
	for (int i = 0; i < count; i++) {
		GridLevel level = {total, inputSize.width / strides[i], inputSize.height / strides[i], strides[i]};
		if (levels) levels->push_back(level);
		total += level.width * level.height;
	}
	return total;
}

const int scStrides[] = {8, 16, 32, 64};

std::string ShapeString(const cv::Mat & m) {
	std::ostringstream ss;
	for (int i = 0; i < m.dims; i++) ss << (i ? "x" : "") << m.size[i];
	return ss.str();
}

}

// --- Letterbox ---------------------------------------------------------------

THISCLASS::Letterbox THISCLASS::Letterbox::Compute(cv::Size original, cv::Size input, bool centered) {
	Letterbox lb;
	lb.original = original;
	lb.input = input;
	lb.scale = std::min((float)input.width / std::max(original.width, 1), (float)input.height / std::max(original.height, 1));
	lb.scaled.width = std::min(input.width, std::max(1, (int)std::lround(original.width * lb.scale)));
	lb.scaled.height = std::min(input.height, std::max(1, (int)std::lround(original.height * lb.scale)));
	lb.padX = centered ? (input.width - lb.scaled.width) / 2 : 0;
	lb.padY = centered ? (input.height - lb.scaled.height) / 2 : 0;
	return lb;
}

cv::Point2f THISCLASS::Letterbox::ToOriginal(cv::Point2f p) const {
	return cv::Point2f((p.x - padX) / scale, (p.y - padY) / scale);
}

cv::Point2f THISCLASS::Letterbox::ToInput(cv::Point2f p) const {
	return cv::Point2f(p.x * scale + padX, p.y * scale + padY);
}

cv::Rect2f THISCLASS::Letterbox::ToOriginal(const cv::Rect2f & r) const {
	cv::Point2f tl = ToOriginal(r.tl());
	cv::Point2f br = ToOriginal(r.br());
	float x1 = std::min(std::max(tl.x, 0.f), (float)original.width);
	float y1 = std::min(std::max(tl.y, 0.f), (float)original.height);
	float x2 = std::min(std::max(br.x, 0.f), (float)original.width);
	float y2 = std::min(std::max(br.y, 0.f), (float)original.height);
	return cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
}

void THISCLASS::LetterboxImage(const cv::Mat & image, cv::Mat & dst, const Letterbox & letterbox) {
	dst.create(letterbox.input, image.type());
	if (letterbox.scaled == letterbox.input) {
		cv::resize(image, dst, letterbox.input, 0, 0, cv::INTER_LINEAR);
		return;
	}
	dst.setTo(cv::Scalar::all(114));
	cv::Mat roi = dst(cv::Rect(cv::Point(letterbox.padX, letterbox.padY), letterbox.scaled));
	cv::resize(image, roi, letterbox.scaled, 0, 0, cv::INTER_LINEAR);
}

// --- Decoding ----------------------------------------------------------------

const char * THISCLASS::FormatName(Format format) {
	switch (format) {
	case FormatAuto: return "Auto";
	case FormatYOLOv8: return "YOLOv8/YOLO11";
	case FormatYOLOv5: return "YOLOv5/YOLOv7";
	case FormatYOLOX: return "YOLOX";
	default: return "unknown";
	}
}

bool THISCLASS::IsAvailable() {
	return SWISTRACK_HAVE_DNN != 0;
}

bool THISCLASS::GetMatrix2D(const cv::Mat & output, cv::Mat & matrix) {
	if (output.empty() || output.channels() != 1) return false;
	cv::Mat f = output;
	if (f.depth() != CV_32F) output.convertTo(f, CV_32F);
	if (! f.isContinuous()) f = f.clone();

	// Drop leading singleton dimensions: 1x84x8400 (OpenCV 4.x/5.x), 1x1xNxK, or already 2D
	std::vector<int> dims;
	for (int i = 0; i < f.dims; i++) dims.push_back(f.size[i]);
	while (dims.size() > 2 && dims[0] == 1) dims.erase(dims.begin());
	if (dims.size() != 2 || dims[0] < 1 || dims[1] < 1) return false;
	matrix = f.reshape(1, 2, dims.data());
	return true;
}

int THISCLASS::NumClasses(const cv::Mat & output, Format format) {
	if (format != FormatYOLOv8 && format != FormatYOLOv5 && format != FormatYOLOX) return 0;
	cv::Mat matrix;
	if (! GetMatrix2D(output, matrix)) return 0;
	int classes = MakeView(matrix).k - (format == FormatYOLOv8 ? 4 : 5);
	return classes > 0 ? classes : 0;
}

THISCLASS::Format THISCLASS::InferFormat(const cv::Mat & output, cv::Size inputSize, int numClassesHint, std::string & message, bool * assumed) {
	if (assumed) *assumed = false;
	cv::Mat matrix;
	if (! GetMatrix2D(output, matrix)) {
		message = "Output of shape " + ShapeString(output) + " is not a detection matrix.";
		return FormatUnknown;
	}
	OutputView v = MakeView(matrix);
	if (v.k < 5) {
		message = "Output of shape " + ShapeString(output) + " has too few attributes per box.";
		return FormatUnknown;
	}

	// (4+C) x N with N > 4+C: only Ultralytics YOLOv8/YOLO11 put the attributes first
	if (matrix.rows < matrix.cols) {
		message = "Output " + ShapeString(output) + ": attributes first, YOLOv8/YOLO11 with " + std::to_string(v.k - 4) + " classes.";
		return FormatYOLOv8;
	}

	// N x K: anchor-based (YOLOv5/v7, 3 anchors per cell) or anchor-free (YOLOX, 1 per cell)
	int grid = MakeGrid(inputSize, scStrides, 3, nullptr);
	int grid6 = MakeGrid(inputSize, scStrides, 4, nullptr);
	if (v.n == 3 * grid || v.n == 3 * grid6) {
		message = "Output " + ShapeString(output) + ": 3 anchors per grid cell, YOLOv5/YOLOv7 with " + std::to_string(v.k - 5) + " classes.";
		return FormatYOLOv5;
	}
	if (v.n == grid) {
		if (numClassesHint > 0 && v.k == 4 + numClassesHint) {
			message = "Output " + ShapeString(output) + ": transposed YOLOv8/YOLO11 with " + std::to_string(numClassesHint) + " classes (matches the class names).";
			return FormatYOLOv8;
		}
		if (numClassesHint > 0 && v.k == 5 + numClassesHint) {
			message = "Output " + ShapeString(output) + ": 1 box per grid cell with objectness, YOLOX with " + std::to_string(numClassesHint) + " classes (matches the class names).";
			return FormatYOLOX;
		}
		if (assumed) *assumed = true;
		message = "Output " + ShapeString(output) + ": 1 box per grid cell, assuming YOLOX with " + std::to_string(v.k - 5) + " classes. Select the model family explicitly if this is wrong.";
		return FormatYOLOX;
	}

	message = "Cannot infer the model family from the output shape " + ShapeString(output) + " for an input of " + std::to_string(inputSize.width) + "x" + std::to_string(inputSize.height) + ". Check the input size or select the model family explicitly.";
	return FormatUnknown;
}

bool THISCLASS::Decode(const cv::Mat & output, Format format, cv::Size inputSize, const Thresholds & thresholds, std::vector<Detection> & detections, std::string & error) {
	detections.clear();
	if (format != FormatYOLOv8 && format != FormatYOLOv5 && format != FormatYOLOX) {
		error = "Unknown model output format.";
		return false;
	}

	cv::Mat matrix;
	if (! GetMatrix2D(output, matrix)) {
		error = "Model output of shape " + ShapeString(output) + " is not a detection matrix.";
		return false;
	}
	const OutputView v = MakeView(matrix);
	const bool hasObjectness = (format != FormatYOLOv8);
	const int classOffset = hasObjectness ? 5 : 4;
	const int numClasses = v.k - classOffset;
	if (numClasses < 1) {
		error = "Model output of shape " + ShapeString(output) + " does not match the " + FormatName(format) + " format.";
		return false;
	}

	std::vector<GridLevel> levels;
	if (format == FormatYOLOX) {
		int total = MakeGrid(inputSize, scStrides, 3, &levels);
		if (total != v.n) {
			error = "Model output of shape " + ShapeString(output) + " does not match the YOLOX grid (" + std::to_string(total) + " cells) for an input of " + std::to_string(inputSize.width) + "x" + std::to_string(inputSize.height) + ". Check the input size.";
			return false;
		}
	}

	std::vector<char> allowed;
	if (! thresholds.classes.empty()) {
		allowed.assign(numClasses, 0);
		for (int c : thresholds.classes)
			if (c >= 0 && c < numClasses) allowed[c] = 1;
	}

	const float minScore = thresholds.minScore;
	auto emit = [&](int i, float score, int classId) {
		float cx = v.At(i, 0), cy = v.At(i, 1), w = v.At(i, 2), h = v.At(i, 3);
		if (format == FormatYOLOX) {
			const GridLevel * level = &levels[0];
			for (const GridLevel & l : levels)
				if (i >= l.start) level = &l;
			int cell = i - level->start;
			cx = (cx + cell % level->width) * level->stride;
			cy = (cy + cell / level->width) * level->stride;
			w = std::exp(w) * level->stride;
			h = std::exp(h) * level->stride;
		}
		if (! (w > 0 && h > 0) || ! std::isfinite(cx) || ! std::isfinite(cy) || ! std::isfinite(w) || ! std::isfinite(h)) return;
		Detection d;
		d.box = cv::Rect2f(cx - w / 2, cy - h / 2, w, h);
		d.score = score;
		d.classId = classId;
		detections.push_back(d);
	};

	if (v.strideAnchor == 1) {
		// Attributes first (YOLOv8): go through the class rows (contiguous in memory) to find the best class of each anchor
		std::vector<float> best(v.n, -1.f);
		std::vector<int> bestClass(v.n, -1);
		for (int c = 0; c < numClasses; c++) {
			if (! allowed.empty() && ! allowed[c]) continue;
			const float * row = v.data + (classOffset + c) * v.strideAttribute;
			for (int i = 0; i < v.n; i++) {
				if (row[i] > best[i]) {
					best[i] = row[i];
					bestClass[i] = c;
				}
			}
		}
		for (int i = 0; i < v.n; i++) {
			if (bestClass[i] < 0) continue;
			float score = hasObjectness ? best[i] * v.At(i, 4) : best[i];
			if (score >= minScore) emit(i, score, bestClass[i]);
		}
	} else {
		// Anchors first (YOLOv5, YOLOX, transposed YOLOv8): one contiguous row per anchor
		for (int i = 0; i < v.n; i++) {
			const float * row = v.data + i * v.strideAnchor;
			float objectness = hasObjectness ? row[4] : 1.f;
			if (objectness < minScore) continue;
			float best = -1.f;
			int bestClass = -1;
			for (int c = 0; c < numClasses; c++) {
				if (row[classOffset + c] > best && (allowed.empty() || allowed[c])) {
					best = row[classOffset + c];
					bestClass = c;
				}
			}
			if (bestClass < 0) continue;
			float score = best * objectness;
			if (score >= minScore) emit(i, score, bestClass);
		}
	}
	return true;
}

void THISCLASS::NonMaximumSuppression(std::vector<Detection> & detections, const Thresholds & thresholds) {
	if (detections.empty()) return;

#if SWISTRACK_HAVE_DNN
	// Class-aware NMS: shift the boxes of each class to a separate region so that boxes of different classes never overlap
	// (cv::dnn::NMSBoxesBatched does the same, but only exists since OpenCV 4.7).
	double offset = 0;
	if (thresholds.classAware) {
		for (const Detection & d : detections)
			offset = std::max(offset, (double)std::max(std::abs(d.box.x) + d.box.width, std::abs(d.box.y) + d.box.height));
		offset += 1;
	}

	std::vector<cv::Rect2d> boxes;
	std::vector<float> scores;
	boxes.reserve(detections.size());
	scores.reserve(detections.size());
	for (const Detection & d : detections) {
		double shift = thresholds.classAware ? d.classId * offset : 0;
		boxes.push_back(cv::Rect2d(d.box.x + shift, d.box.y + shift, d.box.width, d.box.height));
		scores.push_back(d.score);
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, scores, 0.f, thresholds.nmsIoU, indices);

	std::vector<Detection> kept;
	kept.reserve(indices.size());
	for (int i : indices) kept.push_back(detections[i]);
	detections.swap(kept);
#endif

	std::stable_sort(detections.begin(), detections.end(), [](const Detection & a, const Detection & b) {
		return a.score > b.score;
	});
	if (thresholds.maxDetections > 0 && (int)detections.size() > thresholds.maxDetections)
		detections.resize(thresholds.maxDetections);
}

void THISCLASS::MapToOriginal(std::vector<Detection> & detections, const Letterbox & letterbox) {
	for (Detection & d : detections)
		d.box = letterbox.ToOriginal(d.box);
	detections.erase(std::remove_if(detections.begin(), detections.end(), [](const Detection & d) {
		return d.box.width <= 0 || d.box.height <= 0;
	}), detections.end());
}

// --- Configuration helpers ---------------------------------------------------

bool THISCLASS::ParseClassList(const std::string & text, std::vector<int> & classes) {
	classes.clear();
	std::string s = text;
	for (char & c : s)
		if (c == ',' || c == ';') c = ' ';
	std::istringstream ss(s);
	std::string token;
	while (ss >> token) {
		if (token.find_first_not_of("0123456789") != std::string::npos || token.size() > 9) return false;
		classes.push_back(std::stoi(token));
	}
	return true;
}

bool THISCLASS::ReadClassNames(const std::string & path, std::vector<std::string> & names) {
	names.clear();
	std::ifstream file(path);
	if (! file) return false;
	std::string line;
	while (std::getline(file, line)) {
		size_t first = line.find_first_not_of(" \t\r");
		size_t last = line.find_last_not_of(" \t\r");
		names.push_back(first == std::string::npos ? std::string() : line.substr(first, last - first + 1));
	}
	while (! names.empty() && names.back().empty()) names.pop_back();
	return true;
}

// --- Detector ----------------------------------------------------------------

struct THISCLASS::Impl {
#if SWISTRACK_HAVE_DNN
	cv::dnn::Net net;
	std::vector<std::string> outputNames;
#endif
	std::vector<cv::Mat> outputs;
	cv::Mat bgr;
	cv::Mat letterboxed;
	cv::Mat blob;
	int outputIndex = 0;
	bool loaded = false;
};

THISCLASS::ObjectDetectorDNN():
	mImpl(new Impl()), mConfig(), mFormat(FormatUnknown), mNumClasses(0), mSwapRB(true), mPixelScale(1.0 / 255.0),
	mWarmupMs(0), mLastInferenceMs(0), mLastTotalMs(0), mBackendDescription() {
}

THISCLASS::~ObjectDetectorDNN() {
}

bool THISCLASS::IsLoaded() const {
	return mImpl->loaded;
}

void THISCLASS::Unload() {
	mImpl.reset(new Impl());
	mFormat = FormatUnknown;
	mNumClasses = 0;
	mBackendDescription.clear();
}

#if SWISTRACK_HAVE_DNN
namespace {

struct BackendInfo {
	int backend;
	int target;
	const char * name;
};

BackendInfo GetBackendInfo(ObjectDetectorDNN::Backend backend) {
	switch (backend) {
	case ObjectDetectorDNN::BackendOpenCL: return {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL, "OpenCV (OpenCL)"};
	case ObjectDetectorDNN::BackendOpenCLFP16: return {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_OPENCL_FP16, "OpenCV (OpenCL FP16)"};
	case ObjectDetectorDNN::BackendCUDA: return {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA, "CUDA"};
	case ObjectDetectorDNN::BackendCUDAFP16: return {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16, "CUDA (FP16)"};
	default: return {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU, "OpenCV (CPU)"};
	}
}

bool IsTargetAvailable(const BackendInfo & info) {
	std::vector<cv::dnn::Target> targets = cv::dnn::getAvailableTargets((cv::dnn::Backend)info.backend);
	return std::find(targets.begin(), targets.end(), (cv::dnn::Target)info.target) != targets.end();
}

}
#endif

bool THISCLASS::Load(const Config & config, std::string & error, std::vector<std::string> & warnings) {
	Unload();

#if ! SWISTRACK_HAVE_DNN
	(void)config;
	(void)warnings;
	error = "This version of SwisTrack was built with an OpenCV library without the dnn module. Rebuild with OpenCV dnn to use deep learning detectors.";
	return false;
#else
	if (config.modelPath.empty()) {
		error = "No model file selected.";
		return false;
	}
	if (config.inputSize.width < 32 || config.inputSize.height < 32) {
		error = "Invalid input size.";
		return false;
	}
	if (config.inputSize.width % 32 != 0 || config.inputSize.height % 32 != 0)
		warnings.push_back("The input width and height should be multiples of 32.");

	Impl & impl = *mImpl;
	try {
		impl.net = cv::dnn::readNetFromONNX(config.modelPath);
	} catch (const cv::Exception & e) {
		error = "Cannot load the ONNX model '" + config.modelPath + "': " + e.what();
		return false;
	}
	if (impl.net.empty()) {
		error = "Cannot load the ONNX model '" + config.modelPath + "'.";
		return false;
	}

	// Backend and target
	BackendInfo backend = GetBackendInfo(config.backend);
	if (config.backend != BackendCPU && ! IsTargetAvailable(backend)) {
		warnings.push_back(std::string(backend.name) + " is not available in this OpenCV build; using the CPU.");
		backend = GetBackendInfo(BackendCPU);
	}

	// Warm-up inference (also determines the output shape)
	cv::Mat dummy(config.inputSize, CV_8UC3, cv::Scalar::all(114));
	cv::dnn::blobFromImage(dummy, impl.blob, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
	cv::TickMeter tm;
	for (int attempt = 0; ; attempt++) {
		try {
			// CPU is the default (and OpenCV 5's new engine warns about any explicit target), so only set it to undo a failed attempt
			if (config.backend != BackendCPU || attempt > 0) {
				impl.net.setPreferableBackend(backend.backend);
				impl.net.setPreferableTarget(backend.target);
			}
			impl.outputNames = impl.net.getUnconnectedOutLayersNames();
			tm.reset();
			tm.start();
			impl.net.setInput(impl.blob);
			impl.net.forward(impl.outputs, impl.outputNames);
			tm.stop();
			break;
		} catch (const cv::Exception & e) {
			if (backend.target != cv::dnn::DNN_TARGET_CPU) {
				warnings.push_back(std::string(backend.name) + " failed (" + e.what() + "); using the CPU.");
				backend = GetBackendInfo(BackendCPU);
				continue;
			}
			error = std::string("Inference failed (check the input width and height, and export the model with opset 12 if your OpenCV is old): ") + e.what();
			return false;
		}
	}
	mWarmupMs = tm.getTimeMilli();
	mBackendDescription = backend.name;

	// Pick the detection output (the largest 2D one; some YOLOv5 exports also output the raw head tensors)
	impl.outputIndex = -1;
	size_t largest = 0;
	std::string shapes;
	for (size_t i = 0; i < impl.outputs.size(); i++) {
		shapes += (i ? ", " : "") + ShapeString(impl.outputs[i]);
		cv::Mat matrix;
		if (GetMatrix2D(impl.outputs[i], matrix) && matrix.total() > largest) {
			largest = matrix.total();
			impl.outputIndex = (int)i;
		}
	}
	if (impl.outputIndex < 0) {
		error = "The model has no detection output (output shapes: " + shapes + ").";
		return false;
	}
	const cv::Mat & output = impl.outputs[impl.outputIndex];

	// Output format
	if (config.format == FormatAuto) {
		std::string message;
		bool assumed = false;
		mFormat = InferFormat(output, config.inputSize, config.numClassesHint, message, &assumed);
		if (mFormat == FormatUnknown) {
			error = message;
			return false;
		}
		if (assumed) warnings.push_back(message);
	} else {
		mFormat = config.format;
	}
	Thresholds validation;
	validation.minScore = 2.f;		// Nothing passes; this only checks that the output matches the format
	std::vector<Detection> none;
	if (! Decode(output, mFormat, config.inputSize, validation, none, error)) return false;
	mNumClasses = NumClasses(output, mFormat);
	if (config.numClassesHint > 0 && config.numClassesHint != mNumClasses)
		warnings.push_back("The class names file has " + std::to_string(config.numClassesHint) + " names, but the model outputs " + std::to_string(mNumClasses) + " classes.");

	// Preprocessing
	mSwapRB = (config.colorOrder == ColorRGB) || (config.colorOrder == ColorAuto && mFormat != FormatYOLOX);
	bool unit = (config.pixelRange == RangeUnit) || (config.pixelRange == RangeAuto && mFormat != FormatYOLOX);
	mPixelScale = unit ? 1.0 / 255.0 : 1.0;

	mConfig = config;
	impl.loaded = true;
	return true;
#endif
}

#if SWISTRACK_HAVE_DNN
bool THISCLASS::DetectSingle(const cv::Mat & bgr, const Thresholds & thresholds, std::vector<Detection> & detections, double & inferenceMs, std::string & error) {
	Impl & impl = *mImpl;
	detections.clear();
	Letterbox letterbox = Letterbox::Compute(bgr.size(), mConfig.inputSize, mFormat != FormatYOLOX);
	LetterboxImage(bgr, impl.letterboxed, letterbox);
	cv::dnn::blobFromImage(impl.letterboxed, impl.blob, mPixelScale, cv::Size(), cv::Scalar(), mSwapRB, false, CV_32F);

	// Inference
	cv::TickMeter inference;
	try {
		inference.start();
		impl.net.setInput(impl.blob);
		impl.net.forward(impl.outputs, impl.outputNames);
		inference.stop();
	} catch (const cv::Exception & e) {
		error = std::string("Inference failed: ") + e.what();
		return false;
	}
	inferenceMs += inference.getTimeMilli();
	if (impl.outputIndex >= (int)impl.outputs.size()) {
		error = "The model did not produce the expected output.";
		return false;
	}

	// Postprocessing
	if (! Decode(impl.outputs[impl.outputIndex], mFormat, mConfig.inputSize, thresholds, detections, error)) return false;
	NonMaximumSuppression(detections, thresholds);
	MapToOriginal(detections, letterbox);
	return true;
}
#endif

std::vector<cv::Rect> THISCLASS::TileRects(cv::Size imageSize, int tilesX, int tilesY, float overlap) {
	std::vector<cv::Rect> tiles;
	tilesX = std::max(1, tilesX);
	tilesY = std::max(1, tilesY);
	overlap = std::min(std::max(overlap, 0.0f), 0.9f);
	// n tiles of width w with overlap o*w cover W = w * (n - (n - 1) * o)
	const int w = (int)std::ceil(imageSize.width / (tilesX - (tilesX - 1) * overlap));
	const int h = (int)std::ceil(imageSize.height / (tilesY - (tilesY - 1) * overlap));
	for (int ty = 0; ty < tilesY; ty++) {
		for (int tx = 0; tx < tilesX; tx++) {
			int x = tilesX == 1 ? 0 : (int)std::lround((double)(imageSize.width - w) * tx / (tilesX - 1));
			int y = tilesY == 1 ? 0 : (int)std::lround((double)(imageSize.height - h) * ty / (tilesY - 1));
			tiles.push_back(cv::Rect(x, y, std::min(w, imageSize.width - x), std::min(h, imageSize.height - y)));
		}
	}
	return tiles;
}

bool THISCLASS::Detect(const cv::Mat & image, const Thresholds & thresholds, std::vector<Detection> & detections, std::string & error) {
	detections.clear();
	if (! mImpl->loaded) {
		error = "No model loaded.";
		return false;
	}
	if (image.empty() || image.depth() != CV_8U) {
		error = "The input image must be an 8-bit image.";
		return false;
	}

#if ! SWISTRACK_HAVE_DNN
	(void)thresholds;
	(void)mSwapRB;
	(void)mPixelScale;
	return false;
#else
	Impl & impl = *mImpl;
	cv::TickMeter total;
	total.start();

	// Preprocessing: BGR, letterbox, blob
	const cv::Mat * bgr = &image;
	if (image.channels() == 1) {
		cv::cvtColor(image, impl.bgr, cv::COLOR_GRAY2BGR);
		bgr = &impl.bgr;
	} else if (image.channels() == 4) {
		cv::cvtColor(image, impl.bgr, cv::COLOR_BGRA2BGR);
		bgr = &impl.bgr;
	} else if (image.channels() != 3) {
		error = "Unsupported number of image channels.";
		return false;
	}
	const std::vector<cv::Rect> tiles = TileRects(bgr->size(), thresholds.tilesX, thresholds.tilesY, thresholds.tileOverlap);
	double inferenceMs = 0;
	if (tiles.size() <= 1) {
		if (! DetectSingle(*bgr, thresholds, detections, inferenceMs, error)) return false;
	} else {
		const cv::Rect bounds(0, 0, bgr->cols, bgr->rows);
		std::vector<Detection> tileDetections;
		for (const cv::Rect & tile : tiles) {
			if (! DetectSingle((*bgr)(tile), thresholds, tileDetections, inferenceMs, error)) return false;
			for (Detection d : tileDetections) {
				d.box.x += tile.x;
				d.box.y += tile.y;
				// Drop boxes cut off at an inner tile border: the object is (more) fully visible in a neighboring tile
				const float margin = 2.0f;
				bool cutLeft = tile.x > bounds.x && d.box.x <= tile.x + margin;
				bool cutTop = tile.y > bounds.y && d.box.y <= tile.y + margin;
				bool cutRight = tile.br().x < bounds.br().x && d.box.br().x >= tile.br().x - margin;
				bool cutBottom = tile.br().y < bounds.br().y && d.box.br().y >= tile.br().y - margin;
				if (cutLeft || cutTop || cutRight || cutBottom) continue;
				detections.push_back(d);
			}
		}
		if (thresholds.tileFullImage) {
			if (! DetectSingle(*bgr, thresholds, tileDetections, inferenceMs, error)) return false;
			detections.insert(detections.end(), tileDetections.begin(), tileDetections.end());
		}
		NonMaximumSuppression(detections, thresholds);
	}
	mLastInferenceMs = inferenceMs;

	total.stop();
	mLastTotalMs = total.getTimeMilli();
	return true;
#endif
}
