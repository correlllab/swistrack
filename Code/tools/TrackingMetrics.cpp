#include "TrackingMetrics.h"
#include "LinearAssignment.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace TrackingMetrics {

namespace {

const double NaN = std::numeric_limits<double>::quiet_NaN();

// Detections of one frame, as indices into the (contiguous) ID space.
struct FrameData {
	std::vector<int> gt;		// ground truth ID index
	std::vector<double> gtX, gtY;
	std::vector<int> hyp;		// hypothesis ID index
	std::vector<double> hypX, hypY;
};

std::vector<std::string> SplitCSVLine(const std::string & line) {
	std::vector<std::string> fields;
	std::string field;
	std::istringstream ss(line);
	while (std::getline(ss, field, ',')) {
		size_t a = field.find_first_not_of(" \t\r\"");
		size_t b = field.find_last_not_of(" \t\r\"");
		fields.push_back(a == std::string::npos ? std::string() : field.substr(a, b - a + 1));
	}
	return fields;
}

} // namespace

bool ReadCSV(const std::string & filename, std::vector<Detection> & detections, std::string & error) {
	detections.clear();
	std::ifstream file(filename);
	if (! file.is_open()) {
		error = "cannot open " + filename;
		return false;
	}

	std::string line;
	if (! std::getline(file, line)) {
		error = filename + ": empty file (a header line frame,id,x,y is required)";
		return false;
	}
	auto header = SplitCSVLine(line);
	int columns[4] = {-1, -1, -1, -1};
	const char * names[4] = {"frame", "id", "x", "y"};
	for (int c = 0; c < (int) header.size(); c++) {
		std::string name = header[c];
		std::transform(name.begin(), name.end(), name.begin(), [](unsigned char ch) { return (char) std::tolower(ch); });
		for (int k = 0; k < 4; k++)
			if (name == names[k] && columns[k] < 0) columns[k] = c;
	}
	for (int k = 0; k < 4; k++) {
		if (columns[k] < 0) {
			error = filename + ": missing column '" + names[k] + "' in the header";
			return false;
		}
	}

	int lineNumber = 1;
	while (std::getline(file, line)) {
		lineNumber++;
		if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
		auto fields = SplitCSVLine(line);
		try {
			Detection d;
			for (int k = 0; k < 4; k++)
				if (columns[k] >= (int) fields.size()) throw std::invalid_argument("missing field");
			d.frame = std::stoi(fields[columns[0]]);
			d.id = std::stoi(fields[columns[1]]);
			d.x = std::stod(fields[columns[2]]);
			d.y = std::stod(fields[columns[3]]);
			detections.push_back(d);
		} catch (const std::exception &) {
			error = filename + ": invalid line " + std::to_string(lineNumber) + ": " + line;
			return false;
		}
	}
	return true;
}

std::string CheckDuplicates(const std::vector<Detection> & detections) {
	std::set<std::pair<int, int>> seen;
	for (auto & d : detections)
		if (! seen.insert(std::make_pair(d.frame, d.id)).second)
			return "id " + std::to_string(d.id) + " occurs more than once in frame " + std::to_string(d.frame);
	return std::string();
}

Result Evaluate(const std::vector<Detection> & groundTruth, const std::vector<Detection> & hypothesis, double threshold) {
	Result r;

	// Contiguous ID indices
	std::map<int, int> gtIndex, hypIndex;
	for (auto & d : groundTruth) gtIndex.emplace(d.id, 0);
	for (auto & d : hypothesis) hypIndex.emplace(d.id, 0);
	int G = 0, H = 0;
	for (auto & e : gtIndex) e.second = G++;
	for (auto & e : hypIndex) e.second = H++;
	r.gtIDs = G;
	r.hypIDs = H;
	r.gtDetections = (long) groundTruth.size();
	r.hypDetections = (long) hypothesis.size();

	// Group by frame (sorted by frame number)
	std::map<int, FrameData> frames;
	for (auto & d : groundTruth) {
		auto & f = frames[d.frame];
		f.gt.push_back(gtIndex[d.id]);
		f.gtX.push_back(d.x);
		f.gtY.push_back(d.y);
	}
	for (auto & d : hypothesis) {
		auto & f = frames[d.frame];
		f.hyp.push_back(hypIndex[d.id]);
		f.hypX.push_back(d.x);
		f.hypY.push_back(d.y);
	}
	r.frames = (int) frames.size();

	auto distance = [](const FrameData & f, int i, int j) {
		return std::hypot(f.gtX[i] - f.hypX[j], f.gtY[i] - f.hypY[j]);
	};

	// ---------------------------------------------------------------- CLEAR MOT
	std::vector<int> lastMatch(G, -1);				// last hypothesis matched to each ground truth object
	std::vector<std::vector<char>> tracked(G);		// matched flag for each frame in which the object is present
	double distanceSum = 0;
	for (auto & entry : frames) {
		const FrameData & f = entry.second;
		int nG = (int) f.gt.size(), nH = (int) f.hyp.size();
		std::vector<double> dist((size_t) nG * nH);
		for (int i = 0; i < nG; i++)
			for (int j = 0; j < nH; j++)
				dist[(size_t) i * nH + j] = distance(f, i, j);

		std::vector<int> gtMatch(nG, -1);
		std::vector<char> hypUsed(nH, 0);

		// 1. Keep the previous correspondences that are still valid
		for (int i = 0; i < nG; i++) {
			int h = lastMatch[f.gt[i]];
			if (h < 0) continue;
			for (int j = 0; j < nH; j++) {
				if (f.hyp[j] != h || hypUsed[j]) continue;
				if (dist[(size_t) i * nH + j] <= threshold) {
					gtMatch[i] = j;
					hypUsed[j] = 1;
				}
				break;
			}
		}

		// 2. Optimal assignment of the remaining objects and hypotheses
		std::vector<int> rows, cols;
		for (int i = 0; i < nG; i++) if (gtMatch[i] < 0) rows.push_back(i);
		for (int j = 0; j < nH; j++) if (! hypUsed[j]) cols.push_back(j);
		if (! rows.empty() && ! cols.empty()) {
			LinearAssignment::tCostMatrix cost(rows.size() * cols.size());
			for (size_t a = 0; a < rows.size(); a++)
				for (size_t b = 0; b < cols.size(); b++)
					cost[a * cols.size() + b] = dist[(size_t) rows[a] * nH + cols[b]];
			for (auto & p : LinearAssignment::Solve(cost, (int) rows.size(), (int) cols.size(), threshold)) {
				gtMatch[rows[p.first]] = cols[p.second];
				hypUsed[cols[p.second]] = 1;
			}
		}

		// 3. Count
		int matched = 0;
		for (int i = 0; i < nG; i++) {
			int g = f.gt[i];
			int j = gtMatch[i];
			tracked[g].push_back(j >= 0 ? 1 : 0);
			if (j < 0) continue;
			matched++;
			distanceSum += dist[(size_t) i * nH + j];
			int h = f.hyp[j];
			if (lastMatch[g] >= 0 && lastMatch[g] != h) r.idsw++;
			lastMatch[g] = h;
		}
		r.matches += matched;
		r.fn += nG - matched;
		r.fp += nH - matched;
	}

	for (int g = 0; g < G; g++) {
		auto & seq = tracked[g];
		if (seq.empty()) continue;
		long count = std::count(seq.begin(), seq.end(), 1);
		double ratio = (double) count / (double) seq.size();
		if (ratio >= 0.8) r.mostlyTracked++;
		else if (ratio >= 0.2) r.partiallyTracked++;
		else r.mostlyLost++;

		// Fragmentations: tracked -> not tracked transitions between the first and the last tracked frame
		auto first = std::find(seq.begin(), seq.end(), 1);
		if (first == seq.end()) continue;
		auto last = std::find(seq.rbegin(), seq.rend(), 1).base();	// one past the last tracked frame
		for (auto it = first + 1; it < last; it++)
			if (*(it - 1) == 1 && *it == 0) r.fragmentations++;
	}

	r.mota = r.gtDetections > 0 ? 1.0 - (double) (r.fn + r.fp + r.idsw) / (double) r.gtDetections : NaN;
	r.motp = r.matches > 0 ? distanceSum / (double) r.matches : NaN;

	// ---------------------------------------------------------------- Identity metrics (IDF1)
	{
		// Number of frames in which ground truth g and hypothesis h are both present and within the threshold
		std::map<std::pair<int, int>, long> overlap;
		for (auto & entry : frames) {
			const FrameData & f = entry.second;
			for (int i = 0; i < (int) f.gt.size(); i++)
				for (int j = 0; j < (int) f.hyp.size(); j++)
					if (distance(f, i, j) <= threshold) overlap[std::make_pair(f.gt[i], f.hyp[j])]++;
		}

		// Only hypotheses that overlap with some ground truth can contribute to IDTP
		std::map<int, int> column;
		for (auto & e : overlap) column.emplace(e.first.second, 0);
		int nC = 0;
		for (auto & e : column) e.second = nC++;
		long maxOverlap = 0;
		for (auto & e : overlap) maxOverlap = std::max(maxOverlap, e.second);

		// Maximum weight assignment: on a complete bipartite graph with non-negative weights, it is attained by an
		// assignment of maximum cardinality, so minimizing (maxOverlap - overlap) is equivalent.
		if (G > 0 && nC > 0) {
			std::vector<long> weight((size_t) G * nC, 0);
			for (auto & e : overlap) weight[(size_t) e.first.first * nC + column[e.first.second]] = e.second;
			LinearAssignment::tCostMatrix cost(weight.size());
			for (size_t k = 0; k < weight.size(); k++) cost[k] = (double) (maxOverlap - weight[k]);
			for (auto & p : LinearAssignment::Solve(cost, G, nC))
				r.idtp += weight[(size_t) p.first * nC + p.second];
		}
		r.idfn = r.gtDetections - r.idtp;
		r.idfp = r.hypDetections - r.idtp;
		r.idp = r.hypDetections > 0 ? (double) r.idtp / (double) r.hypDetections : NaN;
		r.idr = r.gtDetections > 0 ? (double) r.idtp / (double) r.gtDetections : NaN;
		long denominator = r.gtDetections + r.hypDetections;
		r.idf1 = denominator > 0 ? 2.0 * (double) r.idtp / (double) denominator : NaN;
	}

	// ---------------------------------------------------------------- HOTA (as in TrackEval)
	{
		const int A = 19;
		std::vector<double> alphas(A);
		for (int a = 0; a < A; a++) alphas[a] = 0.05 + a * 0.05;	// = numpy.arange(0.05, 0.99, 0.05)

		auto similarity = [threshold, &distance](const FrameData & f, int i, int j) {
			return std::max(0.0, 1.0 - distance(f, i, j) / threshold);
		};

		// Global alignment score between all ground truth and hypothesis IDs
		std::vector<double> potential((size_t) G * H, 0.0);
		std::vector<double> gtCount(G, 0.0), hypCount(H, 0.0);
		for (auto & entry : frames) {
			const FrameData & f = entry.second;
			int nG = (int) f.gt.size(), nH = (int) f.hyp.size();
			for (int i = 0; i < nG; i++) gtCount[f.gt[i]] += 1;
			for (int j = 0; j < nH; j++) hypCount[f.hyp[j]] += 1;
			if (nG == 0 || nH == 0) continue;
			std::vector<double> sim((size_t) nG * nH), rowSum(nG, 0.0), colSum(nH, 0.0);
			for (int i = 0; i < nG; i++)
				for (int j = 0; j < nH; j++) {
					double s = similarity(f, i, j);
					sim[(size_t) i * nH + j] = s;
					rowSum[i] += s;
					colSum[j] += s;
				}
			for (int i = 0; i < nG; i++)
				for (int j = 0; j < nH; j++) {
					double s = sim[(size_t) i * nH + j];
					double denominator = rowSum[i] + colSum[j] - s;
					if (denominator > DBL_EPSILON) potential[(size_t) f.gt[i] * H + f.hyp[j]] += s / denominator;
				}
		}
		std::vector<double> globalScore((size_t) G * H, 0.0);
		for (int g = 0; g < G; g++)
			for (int h = 0; h < H; h++) {
				double p = potential[(size_t) g * H + h];
				if (p > 0) globalScore[(size_t) g * H + h] = p / (gtCount[g] + hypCount[h] - p);
			}

		// Per-frame matching, then counting per alpha
		std::vector<double> tp(A, 0.0), fn(A, 0.0), fp(A, 0.0), locSum(A, 0.0);
		std::vector<std::map<std::pair<int, int>, double>> matchCounts(A);
		for (auto & entry : frames) {
			const FrameData & f = entry.second;
			int nG = (int) f.gt.size(), nH = (int) f.hyp.size();
			if (nG == 0 || nH == 0) {
				for (int a = 0; a < A; a++) {
					fn[a] += nG;
					fp[a] += nH;
				}
				continue;
			}

			std::vector<double> sim((size_t) nG * nH), score((size_t) nG * nH);
			double maxScore = 0;
			for (int i = 0; i < nG; i++)
				for (int j = 0; j < nH; j++) {
					size_t k = (size_t) i * nH + j;
					sim[k] = similarity(f, i, j);
					score[k] = globalScore[(size_t) f.gt[i] * H + f.hyp[j]] * sim[k];
					maxScore = std::max(maxScore, score[k]);
				}
			LinearAssignment::tCostMatrix cost(score.size());
			for (size_t k = 0; k < score.size(); k++) cost[k] = maxScore - score[k];
			auto pairs = LinearAssignment::Solve(cost, nG, nH);

			for (int a = 0; a < A; a++) {
				int count = 0;
				for (auto & p : pairs) {
					double s = sim[(size_t) p.first * nH + p.second];
					if (s < alphas[a] - DBL_EPSILON) continue;
					count++;
					locSum[a] += s;
					matchCounts[a][std::make_pair(f.gt[p.first], f.hyp[p.second])] += 1;
				}
				tp[a] += count;
				fn[a] += nG - count;
				fp[a] += nH - count;
			}
		}

		r.hotaPerAlpha.assign(A, 0.0);
		for (int a = 0; a < A; a++) {
			double assA = 0, assRe = 0, assPr = 0;
			for (auto & e : matchCounts[a]) {
				double m = e.second;
				double gc = gtCount[e.first.first], hc = hypCount[e.first.second];
				assA += m * m / std::max(1.0, gc + hc - m);
				assRe += m * m / std::max(1.0, gc);
				assPr += m * m / std::max(1.0, hc);
			}
			assA /= std::max(1.0, tp[a]);
			assRe /= std::max(1.0, tp[a]);
			assPr /= std::max(1.0, tp[a]);
			double detA = tp[a] / std::max(1.0, tp[a] + fn[a] + fp[a]);
			double detRe = tp[a] / std::max(1.0, tp[a] + fn[a]);
			double detPr = tp[a] / std::max(1.0, tp[a] + fp[a]);
			double locA = std::max(1e-10, locSum[a]) / std::max(1e-10, tp[a]);
			double hota = std::sqrt(detA * assA);

			r.hotaPerAlpha[a] = hota;
			r.hota += hota / A;
			r.detA += detA / A;
			r.assA += assA / A;
			r.locA += locA / A;
			r.detRe += detRe / A;
			r.detPr += detPr / A;
			r.assRe += assRe / A;
			r.assPr += assPr / A;
		}
	}

	return r;
}

}
