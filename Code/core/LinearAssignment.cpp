#include "LinearAssignment.h"
#define THISCLASS LinearAssignment

#include <algorithm>
#include <cmath>

std::vector<THISCLASS::tPair> THISCLASS::Solve(const tCostMatrix & cost, int rows, int cols, double maxCost) {
	std::vector<int> unassignedRows, unassignedCols;
	return Solve(cost, rows, cols, maxCost, unassignedRows, unassignedCols);
}

std::vector<THISCLASS::tPair> THISCLASS::Solve(const tCostMatrix & cost, int rows, int cols, double maxCost, std::vector<int> & unassignedRows, std::vector<int> & unassignedCols) {
	std::vector<tPair> result;
	unassignedRows.clear();
	unassignedCols.clear();
	if (rows <= 0 || cols <= 0) {
		for (int r = 0; r < rows; r++) unassignedRows.push_back(r);
		for (int c = 0; c < cols; c++) unassignedCols.push_back(c);
		return result;
	}

	// Gated entries are replaced by a large (but finite) cost, so that the problem is always feasible.
	// Any finite cost above (sum of all admissible costs) works: an assignment using a gated entry can then never
	// be preferred over one that leaves the corresponding row unassigned in the final filtering step.
	double finiteMax = 0;
	for (double c : cost)
		if (std::isfinite(c) && c <= maxCost) finiteMax = std::max(finiteMax, std::fabs(c));
	const double gatedCost = (finiteMax + 1.0) * (double)(std::max(rows, cols) + 1);

	// The algorithm below requires n <= m (rows <= columns): transpose if necessary.
	const bool transposed = rows > cols;
	const int n = transposed ? cols : rows;
	const int m = transposed ? rows : cols;
	auto at = [&](int i, int j) -> double {
		double c = transposed ? cost[(size_t)j * cols + i] : cost[(size_t)i * cols + j];
		if (! std::isfinite(c) || c > maxCost) return gatedCost;
		return c;
	};

	// Shortest augmenting path with potentials (1-based indices, as in the classical formulation)
	const double inf = std::numeric_limits<double>::infinity();
	std::vector<double> u(n + 1, 0), v(m + 1, 0), minv(m + 1);
	std::vector<int> p(m + 1, 0), way(m + 1, 0);
	std::vector<char> used(m + 1);
	for (int i = 1; i <= n; i++) {
		p[0] = i;
		int j0 = 0;
		std::fill(minv.begin(), minv.end(), inf);
		std::fill(used.begin(), used.end(), 0);
		do {
			used[j0] = 1;
			int i0 = p[j0], j1 = 0;
			double delta = inf;
			for (int j = 1; j <= m; j++) {
				if (used[j]) continue;
				double cur = at(i0 - 1, j - 1) - u[i0] - v[j];
				if (cur < minv[j]) {
					minv[j] = cur;
					way[j] = j0;
				}
				if (minv[j] < delta) {
					delta = minv[j];
					j1 = j;
				}
			}
			for (int j = 0; j <= m; j++) {
				if (used[j]) {
					u[p[j]] += delta;
					v[j] -= delta;
				} else {
					minv[j] -= delta;
				}
			}
			j0 = j1;
		} while (p[j0] != 0);
		do {
			int j1 = way[j0];
			p[j0] = p[j1];
			j0 = j1;
		} while (j0);
	}

	// Collect the assignment, dropping gated pairs
	std::vector<char> rowAssigned(rows, 0), colAssigned(cols, 0);
	for (int j = 1; j <= m; j++) {
		if (p[j] == 0) continue;
		int r = transposed ? j - 1 : p[j] - 1;
		int c = transposed ? p[j] - 1 : j - 1;
		double original = cost[(size_t)r * cols + c];
		if (! std::isfinite(original) || original > maxCost) continue;
		result.push_back(tPair(r, c));
		rowAssigned[r] = 1;
		colAssigned[c] = 1;
	}
	std::sort(result.begin(), result.end());

	for (int r = 0; r < rows; r++)
		if (! rowAssigned[r]) unassignedRows.push_back(r);
	for (int c = 0; c < cols; c++)
		if (! colAssigned[c]) unassignedCols.push_back(c);
	return result;
}
