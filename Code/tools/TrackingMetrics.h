#ifndef HEADER_TrackingMetrics
#define HEADER_TrackingMetrics

#include <string>
#include <vector>

//! Multi-object tracking metrics for point targets (CLEAR MOT, identity metrics, HOTA).
/*!
	Ground truth and hypothesis are lists of (frame, id, x, y) detections. Two points match if their Euclidean
	distance d is at most the threshold. HOTA uses the similarity max(0, 1 - d / threshold).

	CLEAR MOT (Bernardin & Stiefelhagen 2008), following py-motmetrics:
	  in each frame, the last correspondence of every ground truth object is kept if its hypothesis is present
	  and still within the threshold; the remaining objects are matched optimally (maximum number of matches,
	  then minimum total distance). A match is an ID switch if the object was last matched to a different hypothesis.
	  Fragmentations count the interruptions of tracking (matched -> unmatched -> matched) of a ground truth object.
	  MT / PT / ML: ground truth objects matched in >= 80%, [20%, 80%), < 20% of the frames they are present in.
	  MOTA = 1 - (FN + FP + IDSW) / #GT, MOTP = mean distance of the matches [pixels].
	Identity metrics (Ristani et al. 2016): optimal one-to-one assignment between ground truth and hypothesis
	  trajectories that maximizes the number of frames with matching (d <= threshold) points (IDTP).
	HOTA (Luiten et al. 2020), following TrackEval: per alpha in {0.05, ..., 0.95}, averaged over alpha.
*/
namespace TrackingMetrics {

struct Detection {
	int frame;
	int id;
	double x;
	double y;
};

struct Result {
	// Counts
	int frames = 0;					//!< Frames with at least one ground truth or hypothesis detection.
	int gtIDs = 0;
	int hypIDs = 0;
	long gtDetections = 0;
	long hypDetections = 0;

	// CLEAR MOT
	long matches = 0;
	long fp = 0;
	long fn = 0;
	long idsw = 0;
	long fragmentations = 0;
	int mostlyTracked = 0;
	int partiallyTracked = 0;
	int mostlyLost = 0;
	double mota = 0;
	double motp = 0;				//!< Mean distance of the matches [pixels] (NaN if there are no matches).

	// Identity metrics
	long idtp = 0;
	long idfp = 0;
	long idfn = 0;
	double idf1 = 0;
	double idp = 0;
	double idr = 0;

	// HOTA (averaged over alpha)
	double hota = 0;
	double detA = 0;
	double assA = 0;
	double locA = 0;
	double detRe = 0;
	double detPr = 0;
	double assRe = 0;
	double assPr = 0;
	std::vector<double> hotaPerAlpha;	//!< HOTA for alpha = 0.05, 0.10, ..., 0.95.
};

//! Computes all metrics. Each (frame, id) pair must occur at most once in each list.
Result Evaluate(const std::vector<Detection> & groundTruth, const std::vector<Detection> & hypothesis, double threshold);

//! Reads a CSV file with a header line containing (at least) the columns frame, id, x, y (in any order).
bool ReadCSV(const std::string & filename, std::vector<Detection> & detections, std::string & error);

//! Returns an error message if a (frame, id) pair occurs more than once, or an empty string.
std::string CheckDuplicates(const std::vector<Detection> & detections);

}

#endif
