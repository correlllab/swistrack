// swistrack-synth: deterministic synthetic tracking scenarios with exact ground truth.
//
// Writes into an output folder:
//   video.avi          the rendered video (MJPG by default, or lossless FFV1)
//   background.png     the static background (grayscale, without noise)
//   groundtruth.csv    frame,id,x,y for every frame in which an object is visible
//   nn.swistrack       blob pipeline + NearestNeighborTracking + OutputTracksCSV(nn.csv)
//   mot.swistrack      blob pipeline + MultiObjectTracking + OutputTracksCSV(mot.csv)
//   split.swistrack    BlobDetectionSplitting + MultiObjectTracking + OutputTracksCSV(split.csv)
//   known.swistrack    BlobDetectionSplitting + MultiObjectTracking (known number of objects) + OutputTracksCSV(known.csv)
//   scenario.txt       the parameters used
//
// Coordinates follow the convention of SwisTrack's blob detector: pixel (i, j) covers [i, i+1) x [j, j+1),
// i.e. the center of the top-left pixel is (0.5, 0.5). Frames are numbered from 0, like InputFileAVI.

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

const double PI = 3.14159265358979323846;

// Portable random numbers: std::mt19937 is fully specified by the standard, but the std::*_distribution classes
// are not (libstdc++, libc++ and MSVC produce different sequences). Derive everything from raw mt19937 output
// so that a given seed produces the same scenario on every platform.
class Random {
public:
	explicit Random(unsigned int seed): mEngine(seed), mHasSpare(false), mSpare(0) { }

	//! Uniform in [0, 1).
	double Uniform() {
		uint64_t a = mEngine() >> 5, b = mEngine() >> 6;
		return (a * 67108864.0 + b) / 9007199254740992.0;
	}

	//! Standard normal (Marsaglia polar method).
	double Gauss() {
		if (mHasSpare) {
			mHasSpare = false;
			return mSpare;
		}
		double u, v, q;
		do {
			u = 2 * Uniform() - 1;
			v = 2 * Uniform() - 1;
			q = u * u + v * v;
		} while (q >= 1 || q == 0);
		double f = std::sqrt(-2 * std::log(q) / q);
		mSpare = v * f;
		mHasSpare = true;
		return u * f;
	}

	//! Poisson (Knuth; fine for small means).
	int Poisson(double mean) {
		double limit = std::exp(-mean), p = 1;
		int k = -1;
		do {
			k++;
			p *= Uniform();
		} while (p > limit);
		return k;
	}

private:
	std::mt19937 mEngine;
	bool mHasSpare;
	double mSpare;
};

struct Options {
	std::string scenario = "fish";
	std::string outputFolder;
	int numObjects = -1;
	int frames = 600;
	unsigned int seed = 1;
	int width = 640;
	int height = 480;
	double fps = 25;
	std::string codec = "mjpg";
};

// Everything that defines a scenario.
struct Scenario {
	// Objects
	int numObjects = 10;
	double halfLength = 9;			// ellipse semi-major axis [px]
	double halfWidth = 3.5;			// ellipse semi-minor axis [px]
	double objectGray = 60;			// object intensity

	// Arena
	bool circularArena = true;
	double arenaScale = 1;			// arena size relative to the image
	double backgroundGray = 205;	// arena intensity
	double outsideGray = 150;		// intensity outside the circular arena

	// Motion (correlated random walk)
	double meanSpeed = 2.5;			// [px/frame]
	double speedNoise = 0.15;		// speed Ornstein-Uhlenbeck noise [px/frame]
	double speedRelaxation = 0.05;	// speed relaxation towards the mean (per frame)
	double minSpeedFactor = 0.2;
	double maxSpeedFactor = 2.0;
	double turnPersistence = 0.9;	// angular velocity autocorrelation (per frame)
	double turnNoise = 0.03;		// angular velocity noise [rad/frame]
	double sharpTurnProbability = 0;	// probability of a sudden turn (per object and frame)
	double sharpTurnMin = 0;		// [rad]
	double sharpTurnMax = 0;		// [rad]

	// Image noise and artifacts
	double noiseSigma = 2;			// Gaussian sensor noise (gray levels)
	int occluders = 0;				// static occluding rectangles (drawn over the objects)
	double specksPerFrame = 0;		// mean number of 1-frame false-positive specks
	double dropoutProbability = 0;	// probability that an object starts flickering (per object and frame)
	int dropoutMaxFrames = 3;		// max. duration of a flicker
	double dropoutContrast = 0.12;	// object contrast while flickering (relative)
};

struct Object {
	double x, y;		// center [px]
	double heading;		// [rad]
	double omega;		// angular velocity [rad/frame]
	double speed;		// [px/frame]
	int dropout;		// remaining frames of a flicker
};

struct Rect {
	double x0, y0, x1, y1;
	bool Contains(double x, double y) const {
		return x >= x0 && x < x1 && y >= y0 && y < y1;
	}
};

void Usage() {
	std::cerr <<
		"Usage: swistrack-synth --out FOLDER [options]\n"
		"  --scenario NAME   fish | crowded | occlusion | fast | noise   (default: fish)\n"
		"  --n N             number of objects (default: scenario dependent)\n"
		"  --frames F        number of frames (default: 600)\n"
		"  --seed S          random seed (default: 1)\n"
		"  --size WxH        image size (default: 640x480)\n"
		"  --fps FPS         video frame rate (default: 25)\n"
		"  --codec C         mjpg (default, readable everywhere) | ffv1 (lossless, needs FFmpeg)\n";
}

bool ParseArguments(int argc, char ** argv, Options & options) {
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		auto next = [&]() -> const char * {
			if (i + 1 >= argc) {
				std::cerr << "Missing value for " << arg << "\n";
				std::exit(2);
			}
			return argv[++i];
		};
		if (arg == "--scenario") options.scenario = next();
		else if (arg == "--out" || arg == "-o") options.outputFolder = next();
		else if (arg == "--n") options.numObjects = std::atoi(next());
		else if (arg == "--frames") options.frames = std::atoi(next());
		else if (arg == "--seed") options.seed = (unsigned int) std::strtoul(next(), nullptr, 10);
		else if (arg == "--fps") options.fps = std::atof(next());
		else if (arg == "--codec") options.codec = next();
		else if (arg == "--size") {
			if (std::sscanf(next(), "%dx%d", &options.width, &options.height) != 2) {
				std::cerr << "Invalid size (expected WxH)\n";
				return false;
			}
		} else if (arg == "--help" || arg == "-h") {
			Usage();
			std::exit(0);
		} else {
			std::cerr << "Unknown argument: " << arg << "\n";
			return false;
		}
	}
	if (options.outputFolder.empty()) return false;
	if (options.frames < 1 || options.width < 64 || options.height < 64) {
		std::cerr << "Invalid frames or size\n";
		return false;
	}
	if (options.codec != "mjpg" && options.codec != "ffv1") {
		std::cerr << "Unknown codec: " << options.codec << "\n";
		return false;
	}
	return true;
}

bool MakeScenario(const Options & options, Scenario & s) {
	if (options.scenario == "fish") {
		s.numObjects = 10;
	} else if (options.scenario == "crowded") {
		s.numObjects = 30;
		s.arenaScale = 0.65;
		s.halfLength = 5.5;
		s.halfWidth = 2.5;
		s.circularArena = false;
		s.meanSpeed = 1.6;
		s.turnNoise = 0.04;
	} else if (options.scenario == "occlusion") {
		s.numObjects = 8;
		s.occluders = 6;
	} else if (options.scenario == "fast") {
		s.numObjects = 8;
		s.halfLength = 8;
		s.halfWidth = 3.5;
		s.circularArena = false;
		s.meanSpeed = 10;
		s.speedNoise = 0.6;
		s.turnNoise = 0.05;
		s.sharpTurnProbability = 0.04;
		s.sharpTurnMin = PI / 3;
		s.sharpTurnMax = PI * 0.85;
	} else if (options.scenario == "noise") {
		s.numObjects = 10;
		s.noiseSigma = 7;
		s.specksPerFrame = 2;
		s.dropoutProbability = 0.02;
	} else {
		std::cerr << "Unknown scenario: " << options.scenario << "\n";
		return false;
	}
	if (options.numObjects > 0) s.numObjects = options.numObjects;
	return true;
}

// Arena geometry (circle or rectangle), with reflecting walls for the object centers.
struct Arena {
	bool circular;
	double cx, cy, radius;	// circle
	Rect rect;				// rectangle
	double margin;			// distance kept between object centers and walls

	bool Inside(double x, double y) const {
		if (circular) return std::hypot(x - cx, y - cy) <= radius;
		return x >= rect.x0 && x <= rect.x1 && y >= rect.y0 && y <= rect.y1;
	}

	// Reflects position and heading at the (shrunk) walls.
	void Reflect(Object & o) const {
		if (circular) {
			double r = radius - margin;
			double dx = o.x - cx, dy = o.y - cy;
			double d = std::hypot(dx, dy);
			if (d <= r) return;
			double nx = dx / d, ny = dy / d;
			double vx = std::cos(o.heading), vy = std::sin(o.heading);
			double dot = vx * nx + vy * ny;
			if (dot > 0) {
				vx -= 2 * dot * nx;
				vy -= 2 * dot * ny;
			}
			o.heading = std::atan2(vy, vx);
			double nd = std::max(0.0, 2 * r - d);
			o.x = cx + nx * nd;
			o.y = cy + ny * nd;
			o.omega = 0;
		} else {
			double x0 = rect.x0 + margin, x1 = rect.x1 - margin;
			double y0 = rect.y0 + margin, y1 = rect.y1 - margin;
			if (o.x >= x0 && o.x <= x1 && o.y >= y0 && o.y <= y1) return;
			double vx = std::cos(o.heading), vy = std::sin(o.heading);
			if (o.x < x0) { o.x = 2 * x0 - o.x; vx = std::fabs(vx); }
			if (o.x > x1) { o.x = 2 * x1 - o.x; vx = -std::fabs(vx); }
			if (o.y < y0) { o.y = 2 * y0 - o.y; vy = std::fabs(vy); }
			if (o.y > y1) { o.y = 2 * y1 - o.y; vy = -std::fabs(vy); }
			o.x = std::clamp(o.x, x0, x1);
			o.y = std::clamp(o.y, y0, y1);
			o.heading = std::atan2(vy, vx);
			o.omega = 0;
		}
	}

	void RandomPoint(Random & rng, double & x, double & y) const {
		if (circular) {
			double r = (radius - margin) * std::sqrt(rng.Uniform());
			double a = 2 * PI * rng.Uniform();
			x = cx + r * std::cos(a);
			y = cy + r * std::sin(a);
		} else {
			x = rect.x0 + margin + (rect.x1 - rect.x0 - 2 * margin) * rng.Uniform();
			y = rect.y0 + margin + (rect.y1 - rect.y0 - 2 * margin) * rng.Uniform();
		}
	}
};

// Supersampled coverage of an oriented ellipse, accumulated (max) into a float coverage image.
// If occluders are given, also returns the visible and total coverage.
const int SUPERSAMPLING = 4;

void RenderEllipse(cv::Mat & coverage, double x, double y, double a, double b, double angle, double strength,
                   const std::vector<Rect> & occluders, double & visible, double & total) {
	double c = std::cos(angle), s = std::sin(angle);
	double extent = a + 1;
	int i0 = std::max(0, (int) std::floor(x - extent)), i1 = std::min(coverage.cols - 1, (int) std::floor(x + extent));
	int j0 = std::max(0, (int) std::floor(y - extent)), j1 = std::min(coverage.rows - 1, (int) std::floor(y + extent));
	visible = 0;
	total = 0;
	const double step = 1.0 / SUPERSAMPLING;
	for (int j = j0; j <= j1; j++) {
		float * row = coverage.ptr<float>(j);
		for (int i = i0; i <= i1; i++) {
			int inside = 0;
			for (int sj = 0; sj < SUPERSAMPLING; sj++) {
				double py = j + (sj + 0.5) * step - y;
				for (int si = 0; si < SUPERSAMPLING; si++) {
					double px = i + (si + 0.5) * step - x;
					double u = (px * c + py * s) / a;
					double v = (-px * s + py * c) / b;
					if (u * u + v * v <= 1) inside++;
				}
			}
			if (! inside) continue;
			double cov = (double) inside / (SUPERSAMPLING * SUPERSAMPLING);
			total += cov;
			bool occluded = false;
			for (auto & r : occluders)
				if (r.Contains(i + 0.5, j + 0.5)) occluded = true;
			if (! occluded) visible += cov;
			row[i] = (float) std::max((double) row[i], cov * strength);
		}
	}
}

std::string FormatDouble(double v) {
	std::ostringstream ss;
	ss.precision(6);
	ss << std::fixed << v;
	std::string str = ss.str();
	while (! str.empty() && str.back() == '0') str.pop_back();
	if (! str.empty() && str.back() == '.') str.pop_back();
	return str;
}

void WriteConfiguration(const std::string & path, const std::string & trackerXML, const std::string & csvFile,
                        int threshold, int minArea, int maxArea, int maxBlobs, bool splitBlobs = false) {
	std::ofstream f(path);
	f << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
	  << "<swistrack>\n"
	  << "  <components>\n"
	  << "    <component type=\"InputFileAVI\">\n"
	  << "      <parameter name=\"File\" value=\"video.avi\"/>\n"
	  << "    </component>\n"
	  << "    <component type=\"ConvertInputToGray\">\n"
	  << "      <parameter name=\"Channel\" value=\"-1\"/>\n"
	  << "    </component>\n"
	  << "    <component type=\"BackgroundSubtractionGray\">\n"
	  << "      <parameter name=\"BackgroundImage\" value=\"background.png\"/>\n"
	  << "      <parameter name=\"Mode\" value=\"SubBackgroundImage\"/>\n"
	  << "      <parameter name=\"CorrectMean\" value=\"false\"/>\n"
	  << "    </component>\n"
	  << "    <component type=\"ThresholdGray\">\n"
	  << "      <parameter name=\"Threshold\" value=\"" << threshold << "\"/>\n"
	  << "      <parameter name=\"InvertThreshold\" value=\"false\"/>\n"
	  << "    </component>\n"
	  << (splitBlobs ?
	      "    <component type=\"BlobDetectionSplitting\">\n"
	      "      <parameter name=\"MaxNumber\" value=\"" + std::to_string(maxBlobs) + "\"/>\n"
	      "      <parameter name=\"MinArea\" value=\"" + std::to_string(minArea) + "\"/>\n"
	      "      <parameter name=\"MaxArea\" value=\"0\"/>\n"
	      "    </component>\n"
	    :
	      "    <component type=\"BlobDetectionMinMax\">\n"
	      "      <parameter name=\"MaxNumber\" value=\"" + std::to_string(maxBlobs) + "\"/>\n"
	      "      <parameter name=\"SelectionByArea\" value=\"true\"/>\n"
	      "      <parameter name=\"MinArea\" value=\"" + std::to_string(minArea) + "\"/>\n"
	      "      <parameter name=\"MaxArea\" value=\"" + std::to_string(maxArea) + "\"/>\n"
	      "      <parameter name=\"SelectionByCompactness\" value=\"false\"/>\n"
	      "      <parameter name=\"SelectionByOrientation\" value=\"false\"/>\n"
	      "    </component>\n")
	  << trackerXML
	  << "    <component type=\"OutputTracksCSV\">\n"
	  << "      <parameter name=\"File\" value=\"" << csvFile << "\"/>\n"
	  << "    </component>\n"
	  << "  </components>\n"
	  << "</swistrack>\n";
}

} // namespace

int main(int argc, char ** argv) {
	Options options;
	if (! ParseArguments(argc, argv, options)) {
		Usage();
		return 2;
	}

	Scenario s;
	if (! MakeScenario(options, s)) return 2;

	namespace fs = std::filesystem;
	fs::path folder(options.outputFolder);
	std::error_code ec;
	fs::create_directories(folder, ec);
	if (ec) {
		std::cerr << "Cannot create " << folder << ": " << ec.message() << "\n";
		return 1;
	}

	const int W = options.width, H = options.height;
	Random rng(options.seed);

	// Arena
	Arena arena;
	arena.circular = s.circularArena;
	arena.cx = W / 2.0;
	arena.cy = H / 2.0;
	arena.radius = s.arenaScale * std::min(W, H) / 2.0 - 4;
	double arenaHalfWidth = s.arenaScale * W / 2.0 - 4, arenaHalfHeight = s.arenaScale * H / 2.0 - 4;
	arena.rect = Rect{W / 2.0 - arenaHalfWidth, H / 2.0 - arenaHalfHeight, W / 2.0 + arenaHalfWidth, H / 2.0 + arenaHalfHeight};
	arena.margin = s.halfLength + 2;

	// Static background: light arena with a gentle gradient and low-frequency texture (darker outside the arena)
	cv::Mat background(H, W, CV_32F);
	{
		double p1 = 2 * PI * rng.Uniform(), p2 = 2 * PI * rng.Uniform();
		for (int j = 0; j < H; j++) {
			float * row = background.ptr<float>(j);
			for (int i = 0; i < W; i++) {
				double x = i + 0.5, y = j + 0.5;
				double v = s.backgroundGray - 10.0 * (x / W - 0.5) + 6.0 * (y / H - 0.5)
					+ 3.0 * std::sin(x * 0.031 + p1) * std::cos(y * 0.027 + p2);
				if (! arena.Inside(x, y)) v = s.outsideGray;
				row[i] = (float) v;
			}
		}
	}

	// Occluders: static bars inside the arena, slightly darker than the arena floor, drawn over the objects.
	// Their width (12-28 px) hides an object crossing them for a few frames.
	std::vector<Rect> occluders;
	for (int k = 0; k < s.occluders; k++) {
		for (int attempt = 0; attempt < 100; attempt++) {
			double w = 12 + 16 * rng.Uniform(), h = 60 + 80 * rng.Uniform();
			if (rng.Uniform() < 0.5) std::swap(w, h);
			double x, y;
			arena.RandomPoint(rng, x, y);
			Rect r{std::floor(x - w / 2), std::floor(y - h / 2), std::floor(x + w / 2), std::floor(y + h / 2)};
			if (! arena.Inside(r.x0, r.y0) || ! arena.Inside(r.x1, r.y1) || ! arena.Inside(r.x0, r.y1) || ! arena.Inside(r.x1, r.y0)) continue;
			bool overlap = false;
			for (auto & o : occluders)
				if (r.x0 < o.x1 + 10 && o.x0 < r.x1 + 10 && r.y0 < o.y1 + 10 && o.y0 < r.y1 + 10) overlap = true;
			if (overlap) continue;
			occluders.push_back(r);
			break;
		}
	}
	for (auto & r : occluders)
		for (int j = (int) r.y0; j < (int) r.y1; j++)
			for (int i = (int) r.x0; i < (int) r.x1; i++)
				background.at<float>(j, i) -= 12;

	cv::Mat background8;
	background.convertTo(background8, CV_8U);
	if (! cv::imwrite((folder / "background.png").string(), background8)) {
		std::cerr << "Cannot write background.png\n";
		return 1;
	}

	// Initial object states (non-overlapping)
	std::vector<Object> objects;
	for (int k = 0; k < s.numObjects; k++) {
		Object o{};
		for (int attempt = 0; attempt < 1000; attempt++) {
			arena.RandomPoint(rng, o.x, o.y);
			bool ok = true;
			for (auto & other : objects)
				if (std::hypot(o.x - other.x, o.y - other.y) < 2.5 * s.halfLength) ok = false;
			if (ok) break;
		}
		o.heading = 2 * PI * rng.Uniform() - PI;
		o.omega = 0;
		o.speed = s.meanSpeed * (0.7 + 0.6 * rng.Uniform());
		o.dropout = 0;
		objects.push_back(o);
	}

	// Video writer
	std::string videoPath = (folder / "video.avi").string();
	cv::VideoWriter writer;
	int fourcc = options.codec == "ffv1" ? cv::VideoWriter::fourcc('F', 'F', 'V', '1') : cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
	writer.open(videoPath, fourcc, options.fps, cv::Size(W, H), true);
	if (! writer.isOpened()) {
		std::cerr << "Cannot open the video writer for " << videoPath << " (codec " << options.codec << ")\n";
		return 1;
	}
	if (options.codec == "mjpg") writer.set(cv::VIDEOWRITER_PROP_QUALITY, 100);

	std::ofstream gt((folder / "groundtruth.csv").string());
	gt << "frame,id,x,y\n";

	const double contrast = s.backgroundGray - s.objectGray;
	long closeEncounters = 0, gtRows = 0, hiddenRows = 0, flickerRows = 0;

	cv::Mat frame(H, W, CV_32F), coverage(H, W, CV_32F), frame8, frameColor;
	for (int t = 0; t < options.frames; t++) {
		// Render the objects at their current positions
		coverage.setTo(0);
		for (int k = 0; k < (int) objects.size(); k++) {
			auto & o = objects[k];
			if (o.dropout == 0 && s.dropoutProbability > 0 && rng.Uniform() < s.dropoutProbability)
				o.dropout = 1 + (int) (rng.Uniform() * s.dropoutMaxFrames);
			double strength = o.dropout > 0 ? s.dropoutContrast : 1.0;
			if (o.dropout > 0) flickerRows++;

			double visible, total;
			RenderEllipse(coverage, o.x, o.y, s.halfLength, s.halfWidth, o.heading, strength, occluders, visible, total);

			// Ground truth: objects that are at least half visible (flickering objects are present, hence listed)
			if (total > 0 && visible >= 0.5 * total) {
				gt << t << ',' << k << ',' << FormatDouble(o.x) << ',' << FormatDouble(o.y) << '\n';
				gtRows++;
			} else {
				hiddenRows++;
			}
			for (int m = 0; m < k; m++)
				if (std::hypot(o.x - objects[m].x, o.y - objects[m].y) < 2 * s.halfLength) closeEncounters++;
		}

		// False-positive specks (1 frame)
		if (s.specksPerFrame > 0) {
			int count = rng.Poisson(s.specksPerFrame);
			std::vector<Rect> none;
			for (int k = 0; k < count; k++) {
				double x, y, visible, total;
				arena.RandomPoint(rng, x, y);
				double r = 1.5 + 3.5 * rng.Uniform();
				RenderEllipse(coverage, x, y, r, r * (0.6 + 0.4 * rng.Uniform()), PI * rng.Uniform(), 1.0, none, visible, total);
			}
		}

		// Compose: background, objects, occluders on top, noise
		for (int j = 0; j < H; j++) {
			const float * bg = background.ptr<float>(j);
			const float * cov = coverage.ptr<float>(j);
			float * out = frame.ptr<float>(j);
			for (int i = 0; i < W; i++)
				out[i] = (float) (bg[i] - contrast * cov[i] + s.noiseSigma * rng.Gauss());
		}
		for (auto & r : occluders)
			for (int j = (int) r.y0; j < (int) r.y1; j++)
				for (int i = (int) r.x0; i < (int) r.x1; i++)
					frame.at<float>(j, i) = (float) (background.at<float>(j, i) + s.noiseSigma * rng.Gauss());

		frame.convertTo(frame8, CV_8U);	// rounds and saturates
		cv::cvtColor(frame8, frameColor, cv::COLOR_GRAY2BGR);
		writer.write(frameColor);

		// Move the objects (correlated random walk with reflecting walls)
		for (auto & o : objects) {
			if (o.dropout > 0) o.dropout--;
			o.omega = s.turnPersistence * o.omega + s.turnNoise * rng.Gauss();
			o.heading += o.omega;
			if (s.sharpTurnProbability > 0 && rng.Uniform() < s.sharpTurnProbability) {
				double turn = s.sharpTurnMin + (s.sharpTurnMax - s.sharpTurnMin) * rng.Uniform();
				o.heading += rng.Uniform() < 0.5 ? turn : -turn;
			}
			o.speed += s.speedRelaxation * (s.meanSpeed - o.speed) + s.speedNoise * rng.Gauss();
			o.speed = std::clamp(o.speed, s.minSpeedFactor * s.meanSpeed, s.maxSpeedFactor * s.meanSpeed);
			o.x += o.speed * std::cos(o.heading);
			o.y += o.speed * std::sin(o.heading);
			arena.Reflect(o);
			o.heading = std::remainder(o.heading, 2 * PI);
		}
	}
	writer.release();
	gt.close();

	// Detection parameters adapted to the rendered objects
	double area = PI * s.halfLength * s.halfWidth;
	int threshold = (int) std::lround(contrast / 2);
	int minArea = std::max(4, (int) std::lround(0.35 * area));
	int maxArea = (int) std::lround(4 * area);
	int maxBlobs = 2 * s.numObjects + 10;
	double maxSpeed = s.maxSpeedFactor * s.meanSpeed;

	std::ostringstream nn;
	nn << "    <component type=\"NearestNeighborTracking\">\n"
	   << "      <parameter name=\"MaxNumberOfTracks\" value=\"" << 2 * s.numObjects << "\"/>\n"
	   << "      <parameter name=\"MaxNumberOfPoints\" value=\"50\"/>\n"
	   << "      <parameter name=\"MaxDistance\" value=\"" << FormatDouble(std::max(25.0, std::ceil(1.5 * maxSpeed))) << "\"/>\n"
	   << "      <parameter name=\"MaxLostTrackFrames\" value=\"10\"/>\n"
	   << "    </component>\n";
	WriteConfiguration((folder / "nn.swistrack").string(), nn.str(), "nn.csv", threshold, minArea, maxArea, maxBlobs);
	WriteConfiguration((folder / "mot.swistrack").string(), "    <component type=\"MultiObjectTracking\"/>\n", "mot.csv", threshold, minArea, maxArea, maxBlobs);
	WriteConfiguration((folder / "split.swistrack").string(), "    <component type=\"MultiObjectTracking\"/>\n", "split.csv", threshold, minArea, maxArea, maxBlobs, true);
	std::ostringstream known;
	known << "    <component type=\"MultiObjectTracking\">\n"
	      << "      <parameter name=\"MaxNumberOfTracks\" value=\"" << s.numObjects << "\"/>\n"
	      << "      <parameter name=\"FixedNumberOfObjects\" value=\"true\"/>\n"
	      << "    </component>\n";
	WriteConfiguration((folder / "known.swistrack").string(), known.str(), "known.csv", threshold, minArea, maxArea, maxBlobs, true);

	std::ofstream info((folder / "scenario.txt").string());
	info << "scenario " << options.scenario << "\n"
	     << "objects " << s.numObjects << "\n"
	     << "frames " << options.frames << "\n"
	     << "seed " << options.seed << "\n"
	     << "size " << W << "x" << H << "\n"
	     << "codec " << options.codec << "\n"
	     << "object_half_axes " << FormatDouble(s.halfLength) << " " << FormatDouble(s.halfWidth) << "\n"
	     << "mean_speed " << FormatDouble(s.meanSpeed) << "\n"
	     << "arena " << (s.circularArena ? "circle" : "rectangle") << "\n"
	     << "occluders " << occluders.size() << "\n"
	     << "noise_sigma " << FormatDouble(s.noiseSigma) << "\n"
	     << "specks_per_frame " << FormatDouble(s.specksPerFrame) << "\n"
	     << "dropout_probability " << FormatDouble(s.dropoutProbability) << "\n"
	     << "groundtruth_rows " << gtRows << "\n"
	     << "hidden_object_frames " << hiddenRows << "\n"
	     << "flicker_object_frames " << flickerRows << "\n"
	     << "close_encounters " << closeEncounters << "\n";

	std::cout << "swistrack-synth: " << options.scenario << ", " << s.numObjects << " objects, " << options.frames << " frames, seed " << options.seed
	          << " -> " << folder.string() << " (" << gtRows << " ground truth rows, " << hiddenRows << " occluded, "
	          << closeEncounters << " close encounters)\n";
	return 0;
}
