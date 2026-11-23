// Compares LinearAssignment against brute force on random rectangular, gated cost matrices.
#include "LinearAssignment.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>

// Brute force: maximize the number of admissible pairs, then minimize the total cost.
static void BruteForce(const std::vector<double> & cost, int rows, int cols, double maxCost, int & bestCount, double & bestCost) {
	bestCount = -1;
	bestCost = 0;
	std::vector<int> assignment(rows, -1);
	std::vector<char> usedCol(cols, 0);
	std::function<void(int, int, double)> rec = [&](int r, int count, double total) {
		if (r == rows) {
			if (count > bestCount || (count == bestCount && total < bestCost - 1e-9)) {
				bestCount = count;
				bestCost = total;
			}
			return;
		}
		rec(r + 1, count, total);
		for (int c = 0; c < cols; c++) {
			double v = cost[r * cols + c];
			if (usedCol[c] || ! std::isfinite(v) || v > maxCost) continue;
			usedCol[c] = 1;
			rec(r + 1, count + 1, total + v);
			usedCol[c] = 0;
		}
	};
	rec(0, 0, 0);
}

int main() {
	std::mt19937 rng(42);
	int failures = 0;
	for (int trial = 0; trial < 3000; trial++) {
		int rows = rng() % 7;
		int cols = rng() % 7;
		double maxCost = (rng() % 3 == 0) ? std::numeric_limits<double>::infinity() : 50.0;
		std::vector<double> cost(rows * cols);
		for (auto & c : cost) c = (double)(rng() % 100);
		if (trial % 5 == 0) for (auto & c : cost) if (rng() % 4 == 0) c = std::numeric_limits<double>::infinity();

		std::vector<int> ur, uc;
		auto pairs = LinearAssignment::Solve(cost, rows, cols, maxCost, ur, uc);
		double total = 0;
		std::vector<char> rowUsed(rows, 0), colUsed(cols, 0);
		bool valid = true;
		for (auto & p : pairs) {
			if (rowUsed[p.first] || colUsed[p.second] || cost[p.first * cols + p.second] > maxCost) valid = false;
			rowUsed[p.first] = colUsed[p.second] = 1;
			total += cost[p.first * cols + p.second];
		}
		if ((int)(pairs.size() + ur.size()) != rows || (int)(pairs.size() + uc.size()) != cols) valid = false;

		int bestCount;
		double bestCost;
		BruteForce(cost, rows, cols, maxCost, bestCount, bestCost);
		if (! valid || (int)pairs.size() != bestCount || std::fabs(total - bestCost) > 1e-6) {
			failures++;
			if (failures < 5) std::printf("trial %d (%dx%d): got %zu pairs cost %.1f, expected %d pairs cost %.1f\n", trial, rows, cols, pairs.size(), total, bestCount, bestCost);
		}
	}
	std::printf("LinearAssignment: %d failures\n", failures);
	return failures == 0 ? 0 : 1;
}
