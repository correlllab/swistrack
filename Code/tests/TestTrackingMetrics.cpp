// Hand-constructed cases for the tracking metrics; expected values computed by hand (see comments).
#include "TrackingMetrics.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using TrackingMetrics::Detection;
using TrackingMetrics::Evaluate;
using TrackingMetrics::Result;

static int failures = 0;

static void Check(const char * test, const char * what, double value, double expected, double tolerance = 1e-9) {
	bool ok = std::isnan(expected) ? std::isnan(value) : std::fabs(value - expected) <= tolerance;
	if (! ok) {
		std::printf("FAIL %s: %s = %.9f, expected %.9f\n", test, what, value, expected);
		failures++;
	}
}

// Object at a fixed position per ID (x = 100 * id), so that different objects are far apart.
static Detection At(int frame, int id, double x, double y = 0) {
	return Detection{frame, id, x, y};
}

int main() {
	const double T = 10;

	// Perfect tracking: 2 objects, 5 frames.
	{
		std::vector<Detection> gt, hyp;
		for (int t = 0; t < 5; t++) {
			gt.push_back(At(t, 1, 0 + t));
			gt.push_back(At(t, 2, 100 + t));
			hyp.push_back(At(t, 7, 0 + t));
			hyp.push_back(At(t, 8, 100 + t));
		}
		Result r = Evaluate(gt, hyp, T);
		const char * n = "perfect";
		Check(n, "MOTA", r.mota, 1);
		Check(n, "MOTP", r.motp, 0);
		Check(n, "FP", r.fp, 0);
		Check(n, "FN", r.fn, 0);
		Check(n, "IDSW", r.idsw, 0);
		Check(n, "Frag", r.fragmentations, 0);
		Check(n, "MT", r.mostlyTracked, 2);
		Check(n, "ML", r.mostlyLost, 0);
		Check(n, "IDF1", r.idf1, 1);
		Check(n, "HOTA", r.hota, 1);
		Check(n, "DetA", r.detA, 1);
		Check(n, "AssA", r.assA, 1);
		Check(n, "LocA", r.locA, 1);
	}

	// One false positive: 1 object over 4 frames tracked perfectly, plus a far-away hypothesis in frame 2.
	// MOTA = 1 - 1/4. IDF1 = 2*4 / (4 + 5) = 8/9.
	// HOTA: TP = 4, FP = 1 -> DetA = 4/5 for every alpha; AssA = 1 -> HOTA = sqrt(0.8).
	{
		std::vector<Detection> gt, hyp;
		for (int t = 0; t < 4; t++) {
			gt.push_back(At(t, 1, 0));
			hyp.push_back(At(t, 1, 0));
		}
		hyp.push_back(At(2, 2, 300));
		Result r = Evaluate(gt, hyp, T);
		const char * n = "one FP";
		Check(n, "MOTA", r.mota, 0.75);
		Check(n, "FP", r.fp, 1);
		Check(n, "FN", r.fn, 0);
		Check(n, "IDSW", r.idsw, 0);
		Check(n, "IDF1", r.idf1, 8.0 / 9.0);
		Check(n, "IDP", r.idp, 4.0 / 5.0);
		Check(n, "IDR", r.idr, 1);
		Check(n, "DetA", r.detA, 0.8);
		Check(n, "AssA", r.assA, 1);
		Check(n, "HOTA", r.hota, std::sqrt(0.8));
	}

	// One false negative: 1 object over 5 frames, missed in frame 2 by the (single) hypothesis.
	// MOTA = 1 - 1/5, Frag = 1, IDF1 = 2*4 / (5 + 4) = 8/9.
	// HOTA: TP = 4, FN = 1 -> DetA = 0.8; AssA = 4 / (5 + 4 - 4) = 0.8 -> HOTA = 0.8.
	{
		std::vector<Detection> gt, hyp;
		for (int t = 0; t < 5; t++) {
			gt.push_back(At(t, 1, 0));
			if (t != 2) hyp.push_back(At(t, 1, 0));
		}
		Result r = Evaluate(gt, hyp, T);
		const char * n = "one FN";
		Check(n, "MOTA", r.mota, 0.8);
		Check(n, "FP", r.fp, 0);
		Check(n, "FN", r.fn, 1);
		Check(n, "IDSW", r.idsw, 0);
		Check(n, "Frag", r.fragmentations, 1);
		Check(n, "MT", r.mostlyTracked, 1);
		Check(n, "IDF1", r.idf1, 8.0 / 9.0);
		Check(n, "IDR", r.idr, 0.8);
		Check(n, "IDP", r.idp, 1);
		Check(n, "DetA", r.detA, 0.8);
		Check(n, "AssA", r.assA, 0.8);
		Check(n, "HOTA", r.hota, 0.8);
	}

	// One ID switch: 1 object over 6 frames, hypothesis ID 1 in frames 0-2, ID 2 in frames 3-5.
	// MOTA = 1 - 1/6. IDF1 = 2*3 / (6 + 6) = 0.5.
	// HOTA: DetA = 1; each (gt, hyp) pair has 3 matches, A = 3 / (6 + 3 - 3) = 0.5 -> AssA = 0.5, HOTA = sqrt(0.5).
	{
		std::vector<Detection> gt, hyp;
		for (int t = 0; t < 6; t++) {
			gt.push_back(At(t, 1, 0));
			hyp.push_back(At(t, t < 3 ? 1 : 2, 0));
		}
		Result r = Evaluate(gt, hyp, T);
		const char * n = "one IDSW";
		Check(n, "MOTA", r.mota, 1.0 - 1.0 / 6.0);
		Check(n, "IDSW", r.idsw, 1);
		Check(n, "Frag", r.fragmentations, 0);
		Check(n, "FP", r.fp, 0);
		Check(n, "FN", r.fn, 0);
		Check(n, "IDF1", r.idf1, 0.5);
		Check(n, "DetA", r.detA, 1);
		Check(n, "AssA", r.assA, 0.5);
		Check(n, "HOTA", r.hota, std::sqrt(0.5));
	}

	// Identities swapped halfway: objects a (x=0) and b (x=100), 4 frames; hypothesis 1 follows a in frames 0-1
	// and b in frames 2-3, hypothesis 2 the opposite.
	// CLEAR: 2 ID switches in frame 2 -> MOTA = 1 - 2/8. IDF1 = 2*4 / (8 + 8) = 0.5.
	// HOTA: DetA = 1; each of the 4 pairs has 2 matches, A = 2 / (4 + 4 - 2) = 1/3 -> AssA = 1/3.
	{
		std::vector<Detection> gt, hyp;
		for (int t = 0; t < 4; t++) {
			gt.push_back(At(t, 1, 0));
			gt.push_back(At(t, 2, 100));
			hyp.push_back(At(t, 1, t < 2 ? 0 : 100));
			hyp.push_back(At(t, 2, t < 2 ? 100 : 0));
		}
		Result r = Evaluate(gt, hyp, T);
		const char * n = "swap";
		Check(n, "MOTA", r.mota, 0.75);
		Check(n, "IDSW", r.idsw, 2);
		Check(n, "IDF1", r.idf1, 0.5);
		Check(n, "IDP", r.idp, 0.5);
		Check(n, "IDR", r.idr, 0.5);
		Check(n, "DetA", r.detA, 1);
		Check(n, "AssA", r.assA, 1.0 / 3.0);
		Check(n, "HOTA", r.hota, std::sqrt(1.0 / 3.0));
	}

	// Previous correspondences are kept while valid: hypothesis 1 matches the object in frame 0; in frame 1,
	// hypothesis 2 appears closer (d=1) but hypothesis 1 is still within the threshold (d=4) -> no ID switch,
	// hypothesis 2 is a FP. MOTA = 1 - 1/2, MOTP = (0 + 4) / 2.
	{
		std::vector<Detection> gt = {At(0, 1, 0), At(1, 1, 0)};
		std::vector<Detection> hyp = {At(0, 1, 0), At(1, 1, 4), At(1, 2, 1)};
		Result r = Evaluate(gt, hyp, T);
		const char * n = "keep correspondence";
		Check(n, "IDSW", r.idsw, 0);
		Check(n, "FP", r.fp, 1);
		Check(n, "MOTA", r.mota, 0.5);
		Check(n, "MOTP", r.motp, 2);
	}

	// Correspondence no longer valid: as above, but hypothesis 1 moves out of the threshold (d=15) -> the object is
	// matched to hypothesis 2: 1 ID switch, 1 FP. MOTA = 1 - 2/2 = 0.
	{
		std::vector<Detection> gt = {At(0, 1, 0), At(1, 1, 0)};
		std::vector<Detection> hyp = {At(0, 1, 0), At(1, 1, 15), At(1, 2, 1)};
		Result r = Evaluate(gt, hyp, T);
		const char * n = "invalid correspondence";
		Check(n, "IDSW", r.idsw, 1);
		Check(n, "FP", r.fp, 1);
		Check(n, "MOTA", r.mota, 0);
		Check(n, "MOTP", r.motp, 0.5);
	}

	// Optimal (not greedy) assignment: objects at x=0 and x=8, hypotheses at x=5 and x=13 (T=10).
	// Greedy would pair (8, 5) first (d=3) and leave x=0 / x=13 unmatched; the optimum matches both pairs.
	{
		std::vector<Detection> gt = {At(0, 1, 0), At(0, 2, 8)};
		std::vector<Detection> hyp = {At(0, 1, 5), At(0, 2, 13)};
		Result r = Evaluate(gt, hyp, T);
		const char * n = "optimal assignment";
		Check(n, "matches", r.matches, 2);
		Check(n, "MOTA", r.mota, 1);
		Check(n, "MOTP", r.motp, 5);
	}

	// Localization: one object, one hypothesis at d = 3.7 (T = 10): similarity 0.63.
	// The match counts for alpha = 0.05..0.60 (12 of 19 alphas): HOTA = DetA = 12/19.
	// LocA = (12 * 0.63 + 7 * 1) / 19 (TrackEval defines LocA = 1 for alphas without matches).
	{
		std::vector<Detection> gt = {At(0, 1, 0)};
		std::vector<Detection> hyp = {At(0, 5, 0, 3.7)};
		Result r = Evaluate(gt, hyp, T);
		const char * n = "localization";
		Check(n, "MOTA", r.mota, 1);
		Check(n, "MOTP", r.motp, 3.7);
		Check(n, "HOTA", r.hota, 12.0 / 19.0);
		Check(n, "DetA", r.detA, 12.0 / 19.0);
		Check(n, "AssA", r.assA, 12.0 / 19.0);
		Check(n, "LocA", r.locA, (12 * 0.63 + 7.0) / 19.0);
	}

	// Mostly tracked / lost and fragmentations: object 1 present in 10 frames, tracked in frames 0-1, 4-5, 8
	// (5/10 -> partially tracked, 2 fragmentations); object 2 present in 10 frames, tracked in frame 0 only
	// (1/10 -> mostly lost, 0 fragmentations); object 3 tracked in 9 of 10 frames (missed in the last: no
	// fragmentation, mostly tracked).
	{
		std::vector<Detection> gt, hyp;
		for (int t = 0; t < 10; t++) {
			gt.push_back(At(t, 1, 0));
			gt.push_back(At(t, 2, 100));
			gt.push_back(At(t, 3, 200));
			if (t == 0 || t == 1 || t == 4 || t == 5 || t == 8) hyp.push_back(At(t, 1, 0));
			if (t == 0) hyp.push_back(At(t, 2, 100));
			if (t < 9) hyp.push_back(At(t, 3, 200));
		}
		Result r = Evaluate(gt, hyp, T);
		const char * n = "MT/ML/Frag";
		Check(n, "MT", r.mostlyTracked, 1);
		Check(n, "PT", r.partiallyTracked, 1);
		Check(n, "ML", r.mostlyLost, 1);
		Check(n, "Frag", r.fragmentations, 2);
		Check(n, "FN", r.fn, 5 + 9 + 1);
	}

	// Merged detections: two objects 4 px apart, a single hypothesis between them (as a merged blob).
	// One match, one FN per frame.
	{
		std::vector<Detection> gt, hyp;
		for (int t = 0; t < 3; t++) {
			gt.push_back(At(t, 1, 0));
			gt.push_back(At(t, 2, 4));
			hyp.push_back(At(t, 1, 2));
		}
		Result r = Evaluate(gt, hyp, T);
		const char * n = "merged";
		Check(n, "matches", r.matches, 3);
		Check(n, "FN", r.fn, 3);
		Check(n, "IDSW", r.idsw, 0);
		Check(n, "MOTA", r.mota, 0.5);
		Check(n, "IDF1", r.idf1, 2.0 * 3 / 9);
	}

	// Empty hypothesis
	{
		std::vector<Detection> gt = {At(0, 1, 0), At(1, 1, 0)};
		std::vector<Detection> hyp;
		Result r = Evaluate(gt, hyp, T);
		const char * n = "empty hypothesis";
		Check(n, "MOTA", r.mota, 0);
		Check(n, "FN", r.fn, 2);
		Check(n, "IDF1", r.idf1, 0);
		Check(n, "HOTA", r.hota, 0);
	}

	std::printf("TrackingMetrics: %d failures\n", failures);
	return failures == 0 ? 0 : 1;
}
