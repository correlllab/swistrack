// Scenario tests for MultiObjectTracker (Kalman + Hungarian + ByteTrack + OC-SORT).
#include "MultiObjectTracker.h"
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <set>

typedef MultiObjectTracker::Detection Detection;

static int sFailures = 0;
#define CHECK(condition, ...) do { if (! (condition)) { sFailures++; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static int IdNear(const std::vector<Detection> & detections, const std::vector<int> & ids, cv::Point2f p) {
	for (size_t i = 0; i < detections.size(); i++)
		if (cv::norm(detections[i].position - p) < 1e-3) return ids[i];
	return -2;
}

// Two objects cross in an X; while they overlap, the detector reports a single merged blob.
// Identities must follow the motion (not swap).
static void TestCrossing() {
	MultiObjectTracker tracker;
	int idA = -1, idB = -1;
	for (int f = 0; f <= 100; f++) {
		cv::Point2f a(10.0f + 2 * f, 10.0f + 2 * f), b(10.0f + 2 * f, 210.0f - 2 * f);
		std::vector<Detection> detections;
		if (cv::norm(a - b) < 12) detections.push_back(Detection((a + b) * 0.5f));
		else detections = {Detection(a), Detection(b)};
		auto ids = tracker.Update(f, detections);
		if (f == 10) {
			idA = IdNear(detections, ids, a);
			idB = IdNear(detections, ids, b);
		}
		if (f == 100) {
			CHECK(IdNear(detections, ids, a) == idA && idA >= 0, "crossing: object A changed ID (%d -> %d)", idA, IdNear(detections, ids, a));
			CHECK(IdNear(detections, ids, b) == idB && idB >= 0, "crossing: object B changed ID (%d -> %d)", idB, IdNear(detections, ids, b));
		}
	}
}

// An object disappears behind an occluder for 15 frames and reappears; it must keep its ID.
static void TestOcclusion() {
	MultiObjectTracker tracker;
	int id = -1;
	for (int f = 0; f <= 80; f++) {
		cv::Point2f p(20.0f + 3.0f * f, 100.0f + 0.5f * f);
		std::vector<Detection> detections;
		if (f < 30 || f >= 45) detections.push_back(Detection(p));
		auto ids = tracker.Update(f, detections);
		if (f == 20) id = ids[0];
		if (f == 46) CHECK(ids.size() == 1 && ids[0] == id, "occlusion: ID %d after reappearing, expected %d", ids.empty() ? -9 : ids[0], id);
	}
}

// Low-confidence detections (partial occlusion) keep a track alive (ByteTrack second stage).
static void TestLowScoreDetections() {
	MultiObjectTracker tracker;
	MultiObjectTracker::Parameters p;
	p.maxAge = 3;
	tracker.SetParameters(p);
	int id = -1;
	for (int f = 0; f <= 40; f++) {
		cv::Point2f pos(50.0f + 2.0f * f, 50.0f);
		float score = (f >= 10 && f < 30) ? 0.3f : 0.9f;
		auto ids = tracker.Update(f, {Detection(pos, score)});
		if (f == 5) id = ids[0];
		if (f == 20) CHECK(ids[0] == id, "low score: ID %d during low-confidence phase, expected %d", ids[0], id);
		if (f == 40) CHECK(ids[0] == id, "low score: ID %d afterwards, expected %d", ids[0], id);
	}
	// A low-confidence detection alone never starts a track
	MultiObjectTracker t2;
	auto ids = t2.Update(0, {Detection(cv::Point2f(10, 10), 0.3f)});
	CHECK(ids[0] == -1 && t2.GetTracks().empty(), "low score: a low-confidence detection started a track");
}

// Known number of objects: after a long disappearance, objects are re-identified with the same set of IDs.
static void TestFixedNumberOfObjects() {
	MultiObjectTracker tracker;
	MultiObjectTracker::Parameters p;
	p.maxTracks = 3;
	p.fixedNumberOfObjects = true;
	p.maxAge = 5;
	tracker.SetParameters(p);
	std::set<int> seen;
	for (int f = 0; f <= 200; f++) {
		std::vector<Detection> detections;
		if (f < 50 || f >= 150) {
			float shift = f >= 150 ? 300.0f : 0.0f;
			for (int k = 0; k < 3; k++) detections.push_back(Detection(cv::Point2f(100.0f + 100 * k + shift, 100.0f + f)));
		}
		auto ids = tracker.Update(f, detections);
		for (int id : ids) if (id >= 0) seen.insert(id);
	}
	CHECK(seen == std::set<int>({0, 1, 2}), "fixed number: got %zu distinct IDs", seen.size());
	CHECK(tracker.GetTracks().size() == 3, "fixed number: %zu tracks alive", tracker.GetTracks().size());
}

// Detections carrying an ID (markers) are bound to that track immediately.
static void TestDetectionIDs() {
	MultiObjectTracker tracker;
	tracker.Update(0, {Detection(cv::Point2f(0, 0))});
	auto ids = tracker.Update(1, {Detection(cv::Point2f(500, 500), 1.0f, 42), Detection(cv::Point2f(1, 0))});
	CHECK(ids[0] == 42, "IDs: marker detection got ID %d", ids[0]);
	ids = tracker.Update(2, {Detection(cv::Point2f(400, 400), 1.0f, 42)});
	CHECK(ids[0] == 42, "IDs: marker detection got ID %d after a jump", ids[0]);
}

// Frame gaps (e.g. processing every 2nd frame) are handled by the motion model.
static void TestFrameGaps() {
	MultiObjectTracker tracker;
	int id = -1;
	for (int f = 0; f <= 100; f += 4) {
		auto ids = tracker.Update(f, {Detection(cv::Point2f(10.0f + 5.0f * f, 20.0f))});
		if (f == 8) id = ids[0];
		if (f > 8) CHECK(ids[0] == id, "gaps: ID %d at frame %d, expected %d", ids[0], f, id);
	}
}

// Many objects with random-walk motion and detection noise: count identity switches.
static void TestRandomWalk() {
	std::mt19937 rng(7);
	std::normal_distribution<float> noise(0.0f, 1.0f), turn(0.0f, 0.15f);
	const int n = 20;
	std::vector<cv::Point2f> position(n);
	std::vector<float> heading(n), speed(n);
	for (int k = 0; k < n; k++) {
		position[k] = cv::Point2f(60.0f + 120.0f * (k % 5), 60.0f + 120.0f * (k / 5));
		heading[k] = (float)(rng() % 628) / 100.0f;
		speed[k] = 2.0f + (float)(rng() % 100) / 50.0f;
	}
	MultiObjectTracker tracker;
	std::map<int, int> assigned;	// object -> track ID
	int switches = 0, missed = 0;
	for (int f = 0; f < 500; f++) {
		std::vector<Detection> detections;
		for (int k = 0; k < n; k++) {
			heading[k] += turn(rng);
			position[k] += speed[k] * cv::Point2f(std::cos(heading[k]), std::sin(heading[k]));
			if (position[k].x < 20 || position[k].x > 600) heading[k] = (float)CV_PI - heading[k];
			if (position[k].y < 20 || position[k].y > 480) heading[k] = -heading[k];
			detections.push_back(Detection(position[k] + cv::Point2f(noise(rng), noise(rng))));
		}
		auto ids = tracker.Update(f, detections);
		for (int k = 0; k < n; k++) {
			if (ids[k] < 0) { if (f > 5) missed++; continue; }
			auto it = assigned.find(k);
			if (it != assigned.end() && it->second != ids[k]) switches++;
			assigned[k] = ids[k];
		}
	}
	std::printf("random walk: %d ID switches, %d unassigned detections\n", switches, missed);
	CHECK(switches <= 4, "random walk: %d ID switches", switches);
	CHECK(missed == 0, "random walk: %d unassigned detections", missed);
}

int main() {
	TestCrossing();
	TestOcclusion();
	TestLowScoreDetections();
	TestFixedNumberOfObjects();
	TestDetectionIDs();
	TestFrameGaps();
	TestRandomWalk();
	std::printf("MultiObjectTracker: %d failures\n", sFailures);
	return sFailures == 0 ? 0 : 1;
}
