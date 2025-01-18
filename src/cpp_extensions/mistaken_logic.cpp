#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <string>

namespace py = pybind11;

// This function replicates the "pivot stack" logic from apply_mistaken_logic.
// It expects 1D float arrays: pivotHighs, pivotLows (NaN if no pivot).
// It returns two 1D boolean arrays: pivotHighMistaken, pivotLowMistaken.
std::pair<py::array_t<bool>, py::array_t<bool>>
mark_mistakes(py::array_t<double> pivotHighs, py::array_t<double> pivotLows)
{
    auto bufH = pivotHighs.request(); // buffer info for pivotHighs
    auto bufL = pivotLows.request();  // buffer info for pivotLows

    // Basic checks
    if (bufH.ndim != 1 || bufL.ndim != 1) {
        throw std::runtime_error("Both pivotHighs and pivotLows must be 1D arrays.");
    }
    if (bufH.size != bufL.size) {
        throw std::runtime_error("pivotHighs and pivotLows must have the same length.");
    }

    // Data pointers
    double* ptrH = static_cast<double*>(bufH.ptr);
    double* ptrL = static_cast<double*>(bufL.ptr);
    size_t N = bufH.size; // number of rows

    // Prepare output arrays
    py::array_t<bool> outHigh(N);
    py::array_t<bool> outLow(N);

    auto bufOutHigh = outHigh.request();
    auto bufOutLow  = outLow.request();

    bool* ptrOutHigh = static_cast<bool*>(bufOutHigh.ptr);
    bool* ptrOutLow  = static_cast<bool*>(bufOutLow.ptr);

    // Initialize all false
    for (size_t i = 0; i < N; i++) {
        ptrOutHigh[i] = false;
        ptrOutLow[i]  = false;
    }

    // We'll store a stack of (index, 'high'/'low', pivot_value).
    // The logic is nearly the same as your Python version.
    std::vector<std::tuple<size_t, std::string, double>> pivotStack;

    for (size_t i = 0; i < N; i++) {
        double ph = ptrH[i]; // pivot high
        double pl = ptrL[i]; // pivot low

        // If pivot high is not NaN
        if (!std::isnan(ph)) {
            if (!pivotStack.empty() && std::get<1>(pivotStack.back()) == "high") {
                // compare with old pivot
                auto [old_idx, old_type, old_val] = pivotStack.back();
                pivotStack.pop_back(); // remove the old pivot for now
                if (ph > old_val) {
                    // Mark old pivot as mistaken
                    ptrOutHigh[old_idx] = true;
                    // push new pivot
                    pivotStack.push_back({i, "high", ph});
                } else {
                    // Keep old pivot, discard new pivot
                    pivotStack.push_back({old_idx, "high", old_val});
                    // In your Python code, you set the new pivot's "Pivot High Mistaken" to NaN
                    // We'll skip that here, but logically it's "discarded."
                }
            } else {
                // push new pivot
                pivotStack.push_back({i, "high", ph});
            }
        }

        // If pivot low is not NaN
        if (!std::isnan(pl)) {
            if (!pivotStack.empty() && std::get<1>(pivotStack.back()) == "low") {
                auto [old_idx, old_type, old_val] = pivotStack.back();
                pivotStack.pop_back();
                if (pl < old_val) {
                    // Mark old pivot as mistaken
                    ptrOutLow[old_idx] = true;
                    // push new pivot
                    pivotStack.push_back({i, "low", pl});
                } else {
                    // Keep old pivot, discard new pivot
                    pivotStack.push_back({old_idx, "low", old_val});
                }
            } else {
                pivotStack.push_back({i, "low", pl});
            }
        }
    }

    // ~ You can extend logic here for pivot weights if desired. ~

    return std::make_pair(outHigh, outLow);
}

// This macro creates a Python module called "mistaken_logic_ext" with a single function "mark_mistakes".
PYBIND11_MODULE(mistaken_logic_ext, m) {
    m.doc() = "C++ extension that marks pivot mistakes using pivot stack logic";
    m.def("mark_mistakes", &mark_mistakes,
          "mark_mistakes(pivotHighs: np.ndarray, pivotLows: np.ndarray) -> (mistakenHigh: np.ndarray[bool], mistakenLow: np.ndarray[bool])");
}