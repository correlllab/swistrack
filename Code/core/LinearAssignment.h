#ifndef HEADER_LinearAssignment
#define HEADER_LinearAssignment

#include <vector>
#include <utility>
#include <limits>

//! Optimal linear assignment (Hungarian / Jonker-Volgenant shortest augmenting path), O(n^3).
/*!
	Solves the rectangular assignment problem: given a cost matrix with rows x cols entries,
	find the set of (row, col) pairs, each row and column used at most once, that minimizes the total cost.

	Entries larger than maxCost are considered forbidden ("gated"): such pairs are never returned.
	Rows or columns that cannot be assigned at a cost <= maxCost remain unassigned.
*/
class LinearAssignment {
public:
	//! Cost matrix type (row-major, rows x cols).
	typedef std::vector<double> tCostMatrix;
	//! An assigned (row, column) pair.
	typedef std::pair<int, int> tPair;

	//! Solves the assignment problem. Returns the assigned (row, column) pairs.
	static std::vector<tPair> Solve(const tCostMatrix & cost, int rows, int cols, double maxCost = std::numeric_limits<double>::infinity());

	//! Like Solve, but also returns the unassigned rows and columns.
	static std::vector<tPair> Solve(const tCostMatrix & cost, int rows, int cols, double maxCost, std::vector<int> & unassignedRows, std::vector<int> & unassignedCols);
};

#endif
