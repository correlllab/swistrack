// swistrack-eval: scores a tracking result (hypothesis) against ground truth with CLEAR MOT, IDF1 and HOTA.
//
//   swistrack-eval groundtruth.csv hypothesis.csv [--threshold PIXELS] [--json]
//
// Both files are CSV with a header line containing the columns frame, id, x, y (other columns are ignored).
// See TrackingMetrics.h for the metric definitions.

#include "TrackingMetrics.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Usage() {
	std::cerr <<
		"Usage: swistrack-eval GROUNDTRUTH.csv HYPOTHESIS.csv [--threshold PIXELS] [--json]\n"
		"  --threshold PIXELS  match distance (default: 10). HOTA similarity = max(0, 1 - d / PIXELS).\n"
		"  --json              print a single JSON line instead of the table\n";
}

std::string Number(double v, int precision = 4) {
	if (std::isnan(v)) return "nan";
	char buffer[64];
	std::snprintf(buffer, sizeof(buffer), "%.*f", precision, v);
	return buffer;
}

std::string JSONNumber(double v) {
	if (! std::isfinite(v)) return "null";
	char buffer[64];
	std::snprintf(buffer, sizeof(buffer), "%.6f", v);
	return buffer;
}

}

int main(int argc, char ** argv) {
	std::string gtFile, hypFile;
	double threshold = 10;
	bool json = false;
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "--threshold" && i + 1 < argc) {
			threshold = std::atof(argv[++i]);
		} else if (arg == "--json") {
			json = true;
		} else if (arg == "--help" || arg == "-h") {
			Usage();
			return 0;
		} else if (! arg.empty() && arg[0] == '-') {
			std::cerr << "Unknown option: " << arg << "\n";
			Usage();
			return 2;
		} else if (gtFile.empty()) {
			gtFile = arg;
		} else if (hypFile.empty()) {
			hypFile = arg;
		} else {
			Usage();
			return 2;
		}
	}
	if (gtFile.empty() || hypFile.empty() || ! (threshold > 0)) {
		Usage();
		return 2;
	}

	std::vector<TrackingMetrics::Detection> gt, hyp;
	std::string error;
	if (! TrackingMetrics::ReadCSV(gtFile, gt, error) || ! TrackingMetrics::ReadCSV(hypFile, hyp, error)) {
		std::cerr << "swistrack-eval: " << error << "\n";
		return 1;
	}
	error = TrackingMetrics::CheckDuplicates(gt);
	if (! error.empty()) {
		std::cerr << "swistrack-eval: " << gtFile << ": " << error << "\n";
		return 1;
	}
	error = TrackingMetrics::CheckDuplicates(hyp);
	if (! error.empty()) {
		std::cerr << "swistrack-eval: " << hypFile << ": " << error << "\n";
		return 1;
	}

	auto r = TrackingMetrics::Evaluate(gt, hyp, threshold);

	if (json) {
		std::cout << "{"
		          << "\"threshold\":" << JSONNumber(threshold)
		          << ",\"frames\":" << r.frames
		          << ",\"gt_ids\":" << r.gtIDs << ",\"hyp_ids\":" << r.hypIDs
		          << ",\"gt_detections\":" << r.gtDetections << ",\"hyp_detections\":" << r.hypDetections
		          << ",\"MOTA\":" << JSONNumber(r.mota) << ",\"MOTP\":" << JSONNumber(r.motp)
		          << ",\"matches\":" << r.matches << ",\"FP\":" << r.fp << ",\"FN\":" << r.fn
		          << ",\"IDSW\":" << r.idsw << ",\"Frag\":" << r.fragmentations
		          << ",\"MT\":" << r.mostlyTracked << ",\"PT\":" << r.partiallyTracked << ",\"ML\":" << r.mostlyLost
		          << ",\"IDF1\":" << JSONNumber(r.idf1) << ",\"IDP\":" << JSONNumber(r.idp) << ",\"IDR\":" << JSONNumber(r.idr)
		          << ",\"IDTP\":" << r.idtp << ",\"IDFP\":" << r.idfp << ",\"IDFN\":" << r.idfn
		          << ",\"HOTA\":" << JSONNumber(r.hota) << ",\"DetA\":" << JSONNumber(r.detA) << ",\"AssA\":" << JSONNumber(r.assA)
		          << ",\"LocA\":" << JSONNumber(r.locA)
		          << ",\"DetRe\":" << JSONNumber(r.detRe) << ",\"DetPr\":" << JSONNumber(r.detPr)
		          << ",\"AssRe\":" << JSONNumber(r.assRe) << ",\"AssPr\":" << JSONNumber(r.assPr)
		          << "}\n";
		return 0;
	}

	std::cout
		<< "Ground truth: " << gtFile << " (" << r.gtIDs << " ids, " << r.gtDetections << " detections)\n"
		<< "Hypothesis:   " << hypFile << " (" << r.hypIDs << " ids, " << r.hypDetections << " detections)\n"
		<< "Frames: " << r.frames << ", match threshold: " << Number(threshold, 2) << " px\n\n"
		<< "CLEAR MOT\n"
		<< "  MOTA   " << Number(r.mota) << "\n"
		<< "  MOTP   " << Number(r.motp, 3) << " px (mean distance of matches)\n"
		<< "  FP     " << r.fp << "\n"
		<< "  FN     " << r.fn << "\n"
		<< "  IDSW   " << r.idsw << "\n"
		<< "  Frag   " << r.fragmentations << "\n"
		<< "  MT/PT/ML  " << r.mostlyTracked << " / " << r.partiallyTracked << " / " << r.mostlyLost << " (of " << r.gtIDs << ")\n"
		<< "Identity\n"
		<< "  IDF1   " << Number(r.idf1) << "\n"
		<< "  IDP    " << Number(r.idp) << "\n"
		<< "  IDR    " << Number(r.idr) << "\n"
		<< "HOTA (mean over alpha = 0.05..0.95)\n"
		<< "  HOTA   " << Number(r.hota) << "\n"
		<< "  DetA   " << Number(r.detA) << "  (DetRe " << Number(r.detRe) << ", DetPr " << Number(r.detPr) << ")\n"
		<< "  AssA   " << Number(r.assA) << "  (AssRe " << Number(r.assRe) << ", AssPr " << Number(r.assPr) << ")\n"
		<< "  LocA   " << Number(r.locA) << "\n";
	return 0;
}
