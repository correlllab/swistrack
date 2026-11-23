#include "BlobSplitter.h"
#define THISCLASS BlobSplitter

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>

THISCLASS::Parameters::Parameters():
	minArea(10), maxArea(0), objectArea(0), splitRatio(1.6), maxObjectsPerBlob(4), maxNumber(0), splitScore(0.7f) {
}

THISCLASS::BlobSplitter(): mParameters(), mAreaHistory(), mLabels() {
}

void THISCLASS::Reset() {
	mAreaHistory.clear();
	mMajorHistory.clear();
	mMinorHistory.clear();
}

static double Median(const std::deque<double> & values) {
	std::vector<double> v(values.begin(), values.end());
	std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
	return v[v.size() / 2];
}

cv::Vec2d THISCLASS::GetObjectShape() const {
	if (mMajorHistory.size() < 5) return cv::Vec2d(0, 0);
	return cv::Vec2d(Median(mMajorHistory), Median(mMinorHistory));
}

double THISCLASS::GetObjectArea() const {
	if (mParameters.objectArea > 0) return mParameters.objectArea;
	if (mAreaHistory.size() < 5) return 0;
	return Median(mAreaHistory);
}

float THISCLASS::Orientation(double mu20, double mu02, double mu11) {
	return (float)(0.5 * std::atan2(2.0 * mu11, mu20 - mu02));
}

std::vector<THISCLASS::Blob> THISCLASS::Detect(const cv::Mat & binary) {
	const Parameters & p = mParameters;
	std::vector<Blob> blobs;
	if (binary.empty()) return blobs;

	cv::Mat mask;
	if (binary.type() != CV_8UC1) binary.convertTo(mask, CV_8UC1);
	else mask = binary;

	cv::Mat stats, centroids;
	int count = cv::connectedComponentsWithStats(mask, mLabels, stats, centroids, 8, CV_32S);

	// Object area used for this frame (learned from previous frames)
	const double objectArea = GetObjectArea();

	for (int label = 1; label < count; label++) {
		const int area = stats.at<int>(label, cv::CC_STAT_AREA);
		if (area < p.minArea) continue;
		if (p.maxArea > 0 && area > p.maxArea) continue;
		cv::Rect box(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP), stats.at<int>(label, cv::CC_STAT_WIDTH), stats.at<int>(label, cv::CC_STAT_HEIGHT));

		// Learn the object area
		if (p.objectArea <= 0) {
			mAreaHistory.push_back(area);
			while (mAreaHistory.size() > 500) mAreaHistory.pop_front();
		}

		// Number of objects in this blob
		int k = 1;
		if (objectArea > 0 && area >= p.splitRatio * objectArea)
			k = std::max(2, std::min(std::min(p.maxObjectsPerBlob, 16), (int)std::lround(area / objectArea)));

		// Collect the pixels
		cv::Mat roi = mLabels(box);
		std::vector<cv::Point2f> pixels;
		pixels.reserve(area);
		for (int y = 0; y < roi.rows; y++) {
			const int * row = roi.ptr<int>(y);
			for (int x = 0; x < roi.cols; x++)
				if (row[x] == label) pixels.push_back(cv::Point2f((float)(box.x + x), (float)(box.y + y)));
		}

		std::vector<int> assignment(pixels.size(), 0);
		if (k > 1 && (int)pixels.size() >= 2 * k) assignment = Split(pixels, k);
		else k = 1;

		// One detection per cluster
		for (int c = 0; c < k; c++) {
			double n = 0, sx = 0, sy = 0;
			float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;
			for (size_t i = 0; i < pixels.size(); i++) {
				if (assignment[i] != c) continue;
				n++;
				sx += pixels[i].x;
				sy += pixels[i].y;
				minX = std::min(minX, pixels[i].x); maxX = std::max(maxX, pixels[i].x);
				minY = std::min(minY, pixels[i].y); maxY = std::max(maxY, pixels[i].y);
			}
			if (n == 0) continue;
			double cx = sx / n, cy = sy / n, mu20 = 0, mu02 = 0, mu11 = 0;
			for (size_t i = 0; i < pixels.size(); i++) {
				if (assignment[i] != c) continue;
				double dx = pixels[i].x - cx, dy = pixels[i].y - cy;
				mu20 += dx * dx;
				mu02 += dy * dy;
				mu11 += dx * dy;
			}

			// Learn the shape of single objects
			if (k == 1 && (objectArea <= 0 || area < 1.3 * objectArea)) {
				double a = mu20 / n, b = mu02 / n, c = mu11 / n;
				double root = std::sqrt((a - b) * (a - b) / 4 + c * c);
				mMajorHistory.push_back((a + b) / 2 + root);
				mMinorHistory.push_back(std::max(0.25, (a + b) / 2 - root));
				while (mMajorHistory.size() > 500) mMajorHistory.pop_front();
				while (mMinorHistory.size() > 500) mMinorHistory.pop_front();
			}

			Blob blob;
			blob.center = cv::Point2f((float)cx, (float)cy);
			blob.area = n;
			blob.orientation = Orientation(mu20, mu02, mu11);
			blob.boundingBox = cv::Rect2f(minX, minY, maxX - minX + 1, maxY - minY + 1);
			blob.score = k > 1 ? p.splitScore : 1.0f;
			blob.splitCount = k;
			blob.label = label;
			blobs.push_back(blob);
		}
	}

	// Largest first, limited number
	std::sort(blobs.begin(), blobs.end(), [](const Blob & a, const Blob & b) { return a.area > b.area; });
	if (p.maxNumber > 0 && (int)blobs.size() > p.maxNumber) blobs.resize(p.maxNumber);
	return blobs;
}

std::vector<int> THISCLASS::Split(const std::vector<cv::Point2f> & pixels, int k) const {
	// k-means (always computed: fallback, and one of the initializations)
	std::vector<int> kmeansAssignment(pixels.size(), 0);
	std::vector<cv::Point2d> kmeansCenters(k);
	{
		cv::Mat samples((int)pixels.size(), 2, CV_32F, (void *)pixels.data());
		cv::Mat labels, centers;
		cv::RNG & rng = cv::theRNG();
		uint64 savedState = rng.state;
		rng.state = 0x5157524b;	// deterministic results
		cv::kmeans(samples, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1), 3, cv::KMEANS_PP_CENTERS, centers);
		rng.state = savedState;
		for (int i = 0; i < labels.rows; i++) kmeansAssignment[i] = labels.at<int>(i);
		for (int c = 0; c < k; c++) kmeansCenters[c] = cv::Point2d(centers.at<float>(c, 0), centers.at<float>(c, 1));
	}

	const cv::Vec2d shape = GetObjectShape();
	if (shape[0] <= 0) return kmeansAssignment;

	// Blob axes
	double n = (double)pixels.size(), mx = 0, my = 0;
	for (auto & q : pixels) { mx += q.x; my += q.y; }
	mx /= n; my /= n;
	double sxx = 0, syy = 0, sxy = 0;
	for (auto & q : pixels) { sxx += (q.x - mx) * (q.x - mx); syy += (q.y - my) * (q.y - my); sxy += (q.x - mx) * (q.y - my); }
	const double blobAngle = 0.5 * std::atan2(2 * sxy, sxx - syy);
	const cv::Point2d major(std::cos(blobAngle), std::sin(blobAngle)), minor(-std::sin(blobAngle), std::cos(blobAngle));
	auto spread = [&](cv::Point2d axis) {
		double v = 0;
		for (auto & q : pixels) { double d = (q.x - mx) * axis.x + (q.y - my) * axis.y; v += d * d; }
		return std::sqrt(12.0 * v / n);	// length of a uniform distribution with this variance
	};

	// Initializations: k-means centers, objects in a row along the major axis, objects side by side along the minor axis
	struct Init { std::vector<cv::Point2d> means; std::vector<double> angles; };
	std::vector<Init> inits;
	inits.push_back(Init{kmeansCenters, std::vector<double>(k, blobAngle)});
	for (int variant = 0; variant < 2; variant++) {
		cv::Point2d axis = variant == 0 ? major : minor;
		double length = spread(axis);
		Init init;
		for (int c = 0; c < k; c++) {
			double offset = ((c + 0.5) / k - 0.5) * length;
			init.means.push_back(cv::Point2d(mx, my) + axis * offset);
			init.angles.push_back(blobAngle);
		}
		inits.push_back(init);
	}

	double bestLikelihood = -std::numeric_limits<double>::infinity();
	std::vector<int> best = kmeansAssignment;
	for (auto & init : inits) {
		std::vector<int> assignment;
		double likelihood = FitShapeMixture(pixels, init.means, init.angles, shape, assignment);
		if (likelihood > bestLikelihood) {
			bestLikelihood = likelihood;
			best = assignment;
		}
	}
	return best;
}

double THISCLASS::FitShapeMixture(const std::vector<cv::Point2f> & pixels, std::vector<cv::Point2d> & means, std::vector<double> & angles, cv::Vec2d shape, std::vector<int> & assignment) {
	const int k = (int)means.size();
	const size_t n = pixels.size();
	const double l1 = shape[0], l2 = shape[1];
	const double logNorm = -std::log(2 * CV_PI * std::sqrt(l1 * l2));
	std::vector<double> responsibility(n * k);
	double logLikelihood = 0;

	for (int iteration = 0; iteration < 25; iteration++) {
		// E-step (equal mixing weights: objects have the same size)
		logLikelihood = 0;
		for (size_t i = 0; i < n; i++) {
			double logp[16];
			double maxLog = -std::numeric_limits<double>::infinity();
			for (int c = 0; c < k; c++) {
				double dx = pixels[i].x - means[c].x, dy = pixels[i].y - means[c].y;
				double ca = std::cos(angles[c]), sa = std::sin(angles[c]);
				double u = dx * ca + dy * sa, v = -dx * sa + dy * ca;
				logp[c] = logNorm - 0.5 * (u * u / l1 + v * v / l2);
				maxLog = std::max(maxLog, logp[c]);
			}
			double sum = 0;
			for (int c = 0; c < k; c++) sum += std::exp(logp[c] - maxLog);
			logLikelihood += maxLog + std::log(sum / k);
			for (int c = 0; c < k; c++) responsibility[i * k + c] = std::exp(logp[c] - maxLog) / sum;
		}

		// M-step: means and orientations (shape fixed)
		double change = 0;
		for (int c = 0; c < k; c++) {
			double w = 0, sx = 0, sy = 0;
			for (size_t i = 0; i < n; i++) { double r = responsibility[i * k + c]; w += r; sx += r * pixels[i].x; sy += r * pixels[i].y; }
			if (w < 1e-6) continue;
			cv::Point2d mean(sx / w, sy / w);
			double a = 0, b = 0, cc = 0;
			for (size_t i = 0; i < n; i++) {
				double r = responsibility[i * k + c], dx = pixels[i].x - mean.x, dy = pixels[i].y - mean.y;
				a += r * dx * dx; b += r * dy * dy; cc += r * dx * dy;
			}
			change += cv::norm(mean - means[c]);
			means[c] = mean;
			angles[c] = 0.5 * std::atan2(2 * cc, a - b);
		}
		if (change < 0.01) break;
	}

	assignment.assign(n, 0);
	for (size_t i = 0; i < n; i++)
		for (int c = 1; c < k; c++)
			if (responsibility[i * k + c] > responsibility[i * k + assignment[i]]) assignment[i] = c;
	return logLikelihood;
}
