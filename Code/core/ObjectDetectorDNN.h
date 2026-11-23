#ifndef HEADER_ObjectDetectorDNN
#define HEADER_ObjectDetectorDNN

// GUI-independent ONNX object detector (YOLOv8/YOLO11, YOLOv5/YOLOv7, YOLOX) on top of OpenCV's dnn module.
// This file must not depend on wxWidgets: it is unit-tested on its own (Code/tests/TestObjectDetectorDNN.cpp).

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

// SWISTRACK_HAVE_DNN is 1 if OpenCV's dnn module is available. CMake defines SWISTRACK_OPENCV_DNN to 0 or 1
// (depending on whether the dnn component was found); other build systems fall back to looking for the header.
#if defined(SWISTRACK_OPENCV_DNN)
#define SWISTRACK_HAVE_DNN SWISTRACK_OPENCV_DNN
#elif defined(__has_include)
#if __has_include(<opencv2/dnn.hpp>)
#define SWISTRACK_HAVE_DNN 1
#else
#define SWISTRACK_HAVE_DNN 0
#endif
#else
#define SWISTRACK_HAVE_DNN 0
#endif

//! Runs an ONNX object detection model and decodes its output into boxes in original image coordinates.
class ObjectDetectorDNN {

public:
	//! Layout of the model output.
	enum Format {
		FormatAuto,		//!< Infer from the output shape (only for loading).
		FormatYOLOv8,	//!< Ultralytics YOLOv8/YOLO11: 1 x (4+C) x N, boxes cx,cy,w,h in input pixels, no objectness.
		FormatYOLOv5,	//!< YOLOv5/YOLOv7: 1 x N x (5+C), boxes cx,cy,w,h in input pixels, objectness at index 4.
		FormatYOLOX,	//!< YOLOX (exported without decoding): 1 x N x (5+C), raw grid outputs for strides 8/16/32.
		FormatUnknown	//!< Could not be inferred.
	};

	//! Channel order of the network input.
	enum ColorOrder {
		ColorAuto,		//!< RGB, except BGR for YOLOX.
		ColorRGB,
		ColorBGR
	};

	//! Pixel value range of the network input.
	enum PixelRange {
		RangeAuto,		//!< 0..1, except 0..255 for YOLOX.
		RangeUnit,		//!< 0..1 (pixel / 255).
		RangeByte		//!< 0..255 (unnormalized).
	};

	//! Inference backend and target.
	enum Backend {
		BackendCPU,
		BackendOpenCL,
		BackendOpenCLFP16,
		BackendCUDA,
		BackendCUDAFP16
	};

	//! One detection.
	struct Detection {
		cv::Rect2f box;		//!< Bounding box (in network input coordinates after decoding, in original image coordinates after mapping).
		float score;		//!< Confidence in [0, 1].
		int classId;		//!< Class index.
	};

	//! Letterbox transformation: the original image is scaled by "scale" (keeping its aspect ratio) and placed at (padX, padY) on a canvas of size "input".
	struct Letterbox {
		cv::Size original;
		cv::Size input;
		float scale;
		int padX;
		int padY;
		cv::Size scaled;	//!< Size of the scaled image on the canvas.

		//! Computes the letterbox for an image. If centered is false, the image is placed in the top-left corner (YOLOX).
		static Letterbox Compute(cv::Size original, cv::Size input, bool centered);
		//! Maps a point from network input to original image coordinates.
		cv::Point2f ToOriginal(cv::Point2f p) const;
		//! Maps a point from original image to network input coordinates.
		cv::Point2f ToInput(cv::Point2f p) const;
		//! Maps a box from network input to original image coordinates, clipped to the image.
		cv::Rect2f ToOriginal(const cv::Rect2f & r) const;
	};

	//! Score and NMS thresholds.
	struct Thresholds {
		float minScore = 0.1f;			//!< Detections below this score are dropped.
		float nmsIoU = 0.45f;			//!< IoU threshold for non-maximum suppression.
		bool classAware = true;			//!< Suppress only boxes of the same class.
		int maxDetections = 300;		//!< Maximum number of detections kept after NMS (0 = unlimited).
		std::vector<int> classes;		//!< Classes to keep (empty = all).
	};

	//! Model configuration.
	struct Config {
		std::string modelPath;
		cv::Size inputSize = cv::Size(640, 640);
		Format format = FormatAuto;
		ColorOrder colorOrder = ColorAuto;
		PixelRange pixelRange = RangeAuto;
		Backend backend = BackendCPU;
		int numClassesHint = 0;			//!< Number of classes if known (e.g. from a class names file), 0 otherwise. Helps the format inference.
	};

	//! Constructor.
	ObjectDetectorDNN();
	//! Destructor.
	~ObjectDetectorDNN();

	//! Loads a model, runs one warm-up inference and determines the output format. Returns false and sets error on failure. Non-fatal problems are added to warnings.
	bool Load(const Config & config, std::string & error, std::vector<std::string> & warnings);
	//! Unloads the model.
	void Unload();
	//! Returns true if a model is loaded.
	bool IsLoaded() const;

	//! Detects objects on an 8-bit image (1, 3 or 4 channels, BGR). Detections are in original image coordinates, sorted by decreasing score.
	bool Detect(const cv::Mat & image, const Thresholds & thresholds, std::vector<Detection> & detections, std::string & error);

	//! Output format (resolved, never FormatAuto once loaded).
	Format GetFormat() const { return mFormat; }
	//! Number of classes the model outputs.
	int GetNumClasses() const { return mNumClasses; }
	//! Duration of the warm-up inference [ms].
	double GetWarmupMs() const { return mWarmupMs; }
	//! Duration of the last forward pass [ms].
	double GetLastInferenceMs() const { return mLastInferenceMs; }
	//! Duration of the last call to Detect (preprocessing, inference, postprocessing) [ms].
	double GetLastTotalMs() const { return mLastTotalMs; }
	//! Description of the backend in use.
	const std::string & GetBackendDescription() const { return mBackendDescription; }
	//! Configuration in use.
	const Config & GetConfig() const { return mConfig; }

	// --- Building blocks (static, also used by the unit tests) ---

	//! Returns a human-readable name of a format.
	static const char * FormatName(Format format);
	//! Returns whether OpenCV was built with the dnn module.
	static bool IsAvailable();

	//! Views a network output as a 2D CV_32F matrix by dropping leading singleton dimensions (handles 2D/3D/4D outputs of different OpenCV versions). Returns false if that is not possible.
	static bool GetMatrix2D(const cv::Mat & output, cv::Mat & matrix);
	//! Infers the output format from the shape of a network output. Returns FormatUnknown (and an explanation in message) if that is not possible. Sets *assumed (if given) to true if the result is a plausible guess rather than certain.
	static Format InferFormat(const cv::Mat & output, cv::Size inputSize, int numClassesHint, std::string & message, bool * assumed = nullptr);
	//! Number of classes for a given output and format, or 0 if the output does not match the format.
	static int NumClasses(const cv::Mat & output, Format format);
	//! Decodes a network output into detections (in network input coordinates). Detections below thresholds.minScore or not in thresholds.classes are dropped. No NMS.
	static bool Decode(const cv::Mat & output, Format format, cv::Size inputSize, const Thresholds & thresholds, std::vector<Detection> & detections, std::string & error);
	//! Non-maximum suppression (in place). Keeps at most thresholds.maxDetections, sorted by decreasing score.
	static void NonMaximumSuppression(std::vector<Detection> & detections, const Thresholds & thresholds);
	//! Maps detections from network input to original image coordinates and drops boxes that end up empty.
	static void MapToOriginal(std::vector<Detection> & detections, const Letterbox & letterbox);
	//! Scales image into dst (reusing its buffer) according to the letterbox, padding with gray (114).
	static void LetterboxImage(const cv::Mat & image, cv::Mat & dst, const Letterbox & letterbox);

	//! Parses a comma/space-separated list of class ids. Returns false on invalid input.
	static bool ParseClassList(const std::string & text, std::vector<int> & classes);
	//! Reads class names (one per line). Returns false if the file cannot be read.
	static bool ReadClassNames(const std::string & path, std::vector<std::string> & names);

private:
	struct Impl;
	std::unique_ptr<Impl> mImpl;
	Config mConfig;
	Format mFormat;
	int mNumClasses;
	bool mSwapRB;
	double mPixelScale;
	double mWarmupMs;
	double mLastInferenceMs;
	double mLastTotalMs;
	std::string mBackendDescription;
};

#endif
