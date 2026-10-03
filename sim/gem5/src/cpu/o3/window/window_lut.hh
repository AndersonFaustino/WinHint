/**
 * @file
 * @brief Parser and lookup for the B5 runtime LUT (winhint::WindowLut).
 *
 * window_lut.hh - parser and lookup for the B5 runtime LUT
 * (docs/interfaces.md §6), shared by the gem5 `lut` window policy
 * (lut_policy.cc) and the host unit tests (tests/).
 *
 * Header-only, C++17, no gem5 dependency: errors are reported through the
 * return value and an error string, the caller decides whether to fatal().
 *
 * @code
 *   winhint::WindowLut lut;
 *   std::string err;
 *   if (!lut.load("lut.txt", err)) ...;
 *   double x[4] = {ipc, rob_occ, l1d_mpki, mlp};   // in lut.features() order
 *   unsigned cfg = lut.lookup(x);
 * @endcode
 *
 * File format:
 * @verbatim
 *  WINHINT_LUT 1
 *  features <F> name_1 ... name_F
 *  edges name_1 <n_1> e_1 ... e_n1      (one line per feature, same order)
 *  ...
 *  table <N>                            N = prod(n_i + 1)
 *  c_0 ... c_(N-1)                      config indices, row-major
 * @endverbatim
 * '#' starts a comment that runs to the end of the line. Tokens may be
 * split across lines arbitrarily.
 *
 * Bin rule (identical to numpy.searchsorted(edges, x, side="left") used by
 * sim/baselines/lut/whdata.py): bin = number of edges strictly below x, so
 * x == e_k falls in bin k-1 (edges are inclusive upper bin edges) and x
 * above the last edge goes to bin n. NaN goes to bin 0.
 * The table index is row-major with the first feature slowest.
 */
#ifndef WINHINT_WINDOW_LUT_HH
#define WINHINT_WINDOW_LUT_HH

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace winhint {

/**
 * B5 runtime lookup table: per-feature bin edges and a row-major table of
 * configuration indices (format in the file comment). A default-constructed
 * or failed-to-parse LUT is empty and lookup() returns 0.
 */
class WindowLut
{
  public:
    /**
     * @brief Parse from a file. Returns false and sets err on any error.
     * @param path LUT file path.
     * @param[out] err Error message (prefixed with path for parse errors).
     * @return true on success.
     */
    bool
    load(const std::string &path, std::string &err)
    {
        std::ifstream in(path);
        if (!in) {
            err = "cannot open '" + path + "'";
            return false;
        }
        std::stringstream buf;
        buf << in.rdbuf();
        if (!parse(buf.str(), err)) {
            err = path + ": " + err;
            return false;
        }
        return true;
    }

    /**
     * @brief Parse from a string (same format).
     *
     * Clears any previous content first. Rejects a bad header, zero
     * features, edges lines out of order or with non-ascending/NaN edges,
     * a table size other than prod(n_i + 1), non-integer or negative cells,
     * and trailing tokens. Cells are not range-checked against a config
     * table (see maxConfig()).
     *
     * @param text File contents.
     * @param[out] err Error message.
     * @return true on success.
     */
    bool
    parse(const std::string &text, std::string &err)
    {
        names_.clear();
        edges_.clear();
        table_.clear();

        std::vector<std::string> toks;
        {
            std::stringstream in(text);
            std::string line;
            while (std::getline(in, line)) {
                auto h = line.find('#');
                if (h != std::string::npos)
                    line.erase(h);
                std::stringstream ls(line);
                std::string t;
                while (ls >> t)
                    toks.push_back(t);
            }
        }
        size_t p = 0;
        auto word = [&](std::string &out) {
            if (p >= toks.size())
                return false;
            out = toks[p++];
            return true;
        };
        auto number = [&](double &out) {
            std::string t;
            if (!word(t))
                return false;
            char *end = nullptr;
            out = std::strtod(t.c_str(), &end);
            return end != t.c_str() && *end == '\0';
        };
        auto count = [&](size_t &out) {
            double v;
            if (!number(v) || v < 0 || v != std::floor(v))
                return false;
            out = (size_t)v;
            return true;
        };

        std::string w;
        double ver = 0;
        if (!word(w) || w != "WINHINT_LUT" || !number(ver) || ver != 1) {
            err = "bad header (expected 'WINHINT_LUT 1')";
            return false;
        }
        size_t nf = 0;
        if (!word(w) || w != "features" || !count(nf) || nf == 0) {
            err = "bad 'features' line";
            return false;
        }
        names_.resize(nf);
        for (auto &n : names_) {
            if (!word(n)) {
                err = "truncated feature list";
                return false;
            }
        }
        edges_.resize(nf);
        size_t expect = 1;
        for (size_t f = 0; f < nf; ++f) {
            std::string name;
            size_t n = 0;
            if (!word(w) || w != "edges" || !word(name) ||
                name != names_[f] || !count(n)) {
                err = "bad 'edges' line for feature '" + names_[f] + "'";
                return false;
            }
            edges_[f].resize(n);
            for (auto &e : edges_[f]) {
                if (!number(e) || std::isnan(e)) {
                    err = "bad or truncated edges for '" + names_[f] + "'";
                    return false;
                }
            }
            if (!std::is_sorted(edges_[f].begin(), edges_[f].end())) {
                err = "edges of '" + names_[f] + "' are not ascending";
                return false;
            }
            expect *= n + 1;
        }
        size_t n = 0;
        if (!word(w) || w != "table" || !count(n)) {
            err = "bad 'table' line";
            return false;
        }
        if (n != expect) {
            err = "table has " + std::to_string(n) + " cells, expected " +
                  std::to_string(expect);
            return false;
        }
        table_.resize(n);
        for (auto &c : table_) {
            size_t v;
            if (!count(v)) {
                err = "bad or truncated table";
                return false;
            }
            c = (unsigned)v;
        }
        if (p != toks.size()) {
            err = "trailing data after the table";
            return false;
        }
        return true;
    }

    /**
     * @brief Bin of value x for feature f (searchsorted side="left").
     * @param f Feature index (must be < features().size()).
     * @param x Feature value.
     * @return Number of edges strictly below x (0 for NaN).
     */
    size_t
    bin(size_t f, double x) const
    {
        if (std::isnan(x))
            return 0;
        const auto &e = edges_[f];
        return std::lower_bound(e.begin(), e.end(), x) - e.begin();
    }

    /**
     * @brief Flat table index for x (one value per feature, features()
     *        order), row-major with the first feature slowest.
     * @param x Array of features().size() values.
     * @return Index into the table.
     */
    size_t
    index(const double *x) const
    {
        size_t idx = 0;
        for (size_t f = 0; f < edges_.size(); ++f)
            idx = idx * (edges_[f].size() + 1) + bin(f, x[f]);
        return idx;
    }

    /**
     * @brief Config index for x (one value per feature, in features()
     *        order).
     * @param x Array of features().size() values.
     * @return The table cell, or 0 if the LUT is empty.
     */
    unsigned
    lookup(const double *x) const
    {
        return table_.empty() ? 0 : table_[index(x)];
    }

    /**
     * @brief Largest config index in the table (for range checks).
     * @return The maximum cell, or 0 if the LUT is empty.
     */
    unsigned
    maxConfig() const
    {
        return table_.empty()
                   ? 0 : *std::max_element(table_.begin(), table_.end());
    }

    /** @return Feature names, in file (lookup) order. */
    const std::vector<std::string> &features() const { return names_; }
    /** @return Per-feature ascending bin edges. */
    const std::vector<std::vector<double>> &edges() const { return edges_; }
    /** @return Number of table cells (0 if empty). */
    size_t size() const { return table_.size(); }

  private:
    std::vector<std::string> names_;          ///< feature names
    std::vector<std::vector<double>> edges_;  ///< per-feature bin edges
    std::vector<unsigned> table_;             ///< config index per cell
};

} // namespace winhint

#endif // WINHINT_WINDOW_LUT_HH
