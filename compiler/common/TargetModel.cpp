//===- TargetModel.cpp - WinHint machine description ----------------------===//
/**
 * @file
 * @brief TargetModel defaults, machine-JSON loader and config-table lookups.
 */
#include "TargetModel.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

using namespace llvm;

namespace winhint {

/// Defaults of the builtin riscv_ooo machine (see TargetModel.h).
TargetModel::TargetModel() {
  Caches = {{"L1D", 32 * 1024, 4}, {"L2", 256 * 1024, 14}};
  Window = {{64, 32, 16, 16}, {128, 64, 32, 32}, {192, 96, 48, 48}, {256, 128, 64, 64}};
}

// Documented in TargetModel.h.
uint64_t parseSize(StringRef S, uint64_t Default) {
  S = S.trim();
  if (S.empty())
    return Default;
  size_t I = 0;
  while (I < S.size() && (isdigit((unsigned char)S[I]) || S[I] == '.'))
    ++I;
  double V;
  if (S.substr(0, I).getAsDouble(V))
    return Default;
  std::string L = S.substr(I).trim().lower();
  double M = 1;
  if (L == "" || L == "b")
    M = 1;
  else if (L == "kb" || L == "kib" || L == "k")
    M = 1024;
  else if (L == "mb" || L == "mib" || L == "m")
    M = 1024.0 * 1024;
  else if (L == "gb" || L == "gib" || L == "g")
    M = 1024.0 * 1024 * 1024;
  else
    return Default;
  return (uint64_t)(V * M);
}

// Documented in TargetModel.h.
double parseClockGHz(StringRef S, double Default) {
  S = S.trim();
  size_t I = 0;
  while (I < S.size() && (isdigit((unsigned char)S[I]) || S[I] == '.'))
    ++I;
  double V;
  if (S.substr(0, I).getAsDouble(V))
    return Default;
  std::string U = S.substr(I).trim().lower();
  if (U == "ghz" || U.empty())
    return V;
  if (U == "mhz")
    return V / 1000.0;
  return Default;
}

/// @brief Read a numeric member (floating or integer) of a JSON object.
/// @param O JSON object, may be null.
/// @param K Member key.
/// @return The value, or std::nullopt if O is null or K is missing / not a number.
static std::optional<double> getNum(const json::Object *O, StringRef K) {
  if (!O)
    return std::nullopt;
  if (auto V = O->getNumber(K))
    return *V;
  if (auto I = O->getInteger(K))
    return (double)*I;
  return std::nullopt;
}

/// @brief Read a string member of a JSON object.
/// @param O JSON object, may be null.
/// @param K Member key.
/// @return The string, or std::nullopt if O is null or K is missing / not a string.
static std::optional<std::string> getStr(const json::Object *O, StringRef K) {
  if (!O)
    return std::nullopt;
  if (auto V = O->getString(K))
    return V->str();
  return std::nullopt;
}

/// @brief Read a size member given as a number (bytes) or a string ("32kB").
/// @param O JSON object, may be null.
/// @param K Member key.
/// @return Size in bytes, or std::nullopt if missing or unparsable (or zero as a string).
static std::optional<uint64_t> getSize(const json::Object *O, StringRef K) {
  if (auto N = getNum(O, K))
    return (uint64_t)*N;
  if (auto S = getStr(O, K)) {
    uint64_t V = parseSize(*S, 0);
    if (V)
      return V;
  }
  return std::nullopt;
}

/// @brief Read an array member, keeping only its integer elements.
/// @param O JSON object, may be null.
/// @param K Member key.
/// @return The integers in order; empty if O is null or K is not an array.
static std::vector<unsigned> getUIntArray(const json::Object *O, StringRef K) {
  std::vector<unsigned> R;
  if (!O)
    return R;
  if (const json::Array *A = O->getArray(K))
    for (const json::Value &V : *A)
      if (auto I = V.getAsInteger())
        R.push_back((unsigned)*I);
  return R;
}

// Documented in TargetModel.h.
bool TargetModel::loadJSON(StringRef Path, std::string &Err) {
  auto BufOrErr = MemoryBuffer::getFile(Path);
  if (!BufOrErr) {
    Err = "cannot read " + Path.str() + ": " + BufOrErr.getError().message();
    return false;
  }
  Expected<json::Value> Parsed = json::parse((*BufOrErr)->getBuffer());
  if (!Parsed) {
    Err = "JSON parse error in " + Path.str() + ": " + toString(Parsed.takeError());
    return false;
  }
  const json::Object *Root = Parsed->getAsObject();
  if (!Root) {
    Err = Path.str() + ": top-level JSON value is not an object";
    return false;
  }
  SourcePath = Path.str();
  Name = Path.str();
  if (auto N = getStr(Root, "name"))
    Name = *N;

  // ---- cpu ---------------------------------------------------------------
  const json::Object *Cpu = Root->getObject("cpu");
  if (auto V = getNum(Cpu, "issue_width"))
    IssueWidth = (unsigned)*V;
  if (auto V = getNum(Cpu, "dispatch_width"))
    DispatchWidth = (unsigned)*V;
  else if (auto V = getNum(Cpu, "decode_width"))
    DispatchWidth = (unsigned)*V;
  else
    DispatchWidth = IssueWidth;
  if (auto V = getNum(Cpu, "commit_width"))
    CommitWidth = (unsigned)*V;
  if (auto S = getStr(Cpu, "clock"))
    ClockGHz = parseClockGHz(*S, ClockGHz);

  // ---- cache -------------------------------------------------------------
  const json::Object *Cache = Root->getObject("cache");
  if (auto V = getNum(Cache, "cache_line_size"))
    LineSize = (unsigned)*V;
  if (auto V = getNum(Cache, "line_size"))
    LineSize = (unsigned)*V;
  if (Cache) {
    std::vector<CacheLevel> Levels;
    struct Desc {
      const char *Name, *SizeKey, *LatKey, *LatKey2;
      unsigned DefLat;
    } Descs[] = {{"L1D", "l1d_size", "l1d_latency", "l1d_hit_latency", 4},
                 {"L2", "l2_size", "l2_latency", "l2_hit_latency", 14},
                 {"L3", "l3_size", "l3_latency", "l3_hit_latency", 40}};
    for (const Desc &D : Descs) {
      auto Sz = getSize(Cache, D.SizeKey);
      if (!Sz)
        continue;
      unsigned Lat = D.DefLat;
      if (auto L = getNum(Cache, D.LatKey))
        Lat = (unsigned)*L;
      else if (auto L = getNum(Cache, D.LatKey2))
        Lat = (unsigned)*L;
      // gem5 classic caches report tag+data+response latency separately.
      std::string Pfx = std::string(D.Name).substr(0, 2);
      std::transform(Pfx.begin(), Pfx.end(), Pfx.begin(), ::tolower);
      if (D.Name == std::string("L1D"))
        Pfx = "l1d";
      double Sum = 0;
      bool Any = false;
      for (const char *Part : {"_tag_latency", "_data_latency", "_response_latency"})
        if (auto L = getNum(Cache, Pfx + Part)) {
          Sum += *L;
          Any = true;
        }
      if (Any && !getNum(Cache, D.LatKey) && !getNum(Cache, D.LatKey2))
        Lat = (unsigned)std::max(1.0, Sum);
      Levels.push_back({D.Name, *Sz, Lat});
    }
    // Nested form (sim/machines/*.json): "l1d": {"size", "hit_latency_cycles", "mshrs"}.
    {
      std::vector<CacheLevel> Nested;
      for (const char *Key : {"l1d", "l2", "l3"}) {
        const json::Object *C = Cache->getObject(Key);
        if (!C)
          continue;
        auto Sz = getSize(C, "size");
        if (!Sz)
          continue;
        std::string N = Key;
        std::transform(N.begin(), N.end(), N.begin(), ::toupper);
        unsigned Lat = N == "L1D" ? 4 : N == "L2" ? 14 : 40;
        if (auto L = getNum(C, "hit_latency_cycles"))
          Lat = (unsigned)*L;
        else if (auto L = getNum(C, "latency"))
          Lat = (unsigned)*L;
        Nested.push_back({N, *Sz, Lat});
        if (N == "L1D")
          if (auto M = getNum(C, "mshrs"))
            L1DMSHRs = (unsigned)*M;
      }
      if (!Nested.empty())
        Levels = Nested;
    }
    if (!Levels.empty())
      Caches = Levels;
    for (const char *Key : {"l1d_prefetcher", "l2_prefetcher"})
      if (auto S = getStr(Cache, Key))
        if (!S->empty() && StringRef(*S).lower() != "none" && StringRef(*S).lower() != "null")
          StridePrefetcher = true;
    if (auto V = getNum(Cache, "l1d_mshrs"))
      L1DMSHRs = (unsigned)*V;
    if (auto V = getNum(Cache, "mshrs"))
      L1DMSHRs = (unsigned)*V;
    if (const json::Value *P = Cache->get("prefetcher")) {
      if (auto B = P->getAsBoolean())
        StridePrefetcher = *B;
      else if (auto S = P->getAsString())
        StridePrefetcher = !S->empty() && S->lower() != "none" && S->lower() != "null";
      else if (P->getAsObject())
        StridePrefetcher = true;
    }
    if (auto V = getNum(Cache, "effective_fraction"))
      CacheEffectiveFraction = *V;
  }

  // ---- memory ------------------------------------------------------------
  const json::Object *Mem = Root->getObject("memory");
  if (auto V = getNum(Mem, "latency_cycles"))
    MemLatency = (unsigned)*V;
  else if (auto V = getNum(Mem, "latency_ns"))
    MemLatency = (unsigned)std::lround(*V * ClockGHz);
  else if (auto V = getNum(Mem, "latency"))
    MemLatency = (unsigned)*V;

  // ---- latencies ---------------------------------------------------------
  const json::Object *LatObj = Root->getObject("latency");
  if (!LatObj && Cpu)
    LatObj = Cpu->getObject("latency_cycles");
  if (const json::Object *L = LatObj) {
    if (auto V = getNum(L, "load_hit"))
      if (!Caches.empty())
        Caches[0].Latency = (unsigned)*V;
    auto Set = [&](StringRef K, unsigned &F) {
      if (auto V = getNum(L, K))
        F = (unsigned)*V;
    };
    Set("int_alu", Lat.IntAlu);
    Set("int_mul", Lat.IntMul);
    Set("int_div", Lat.IntDiv);
    Set("fp_add", Lat.FpAdd);
    Set("fp_mul", Lat.FpMul);
    Set("fp_fma", Lat.FpFma);
    Set("fp_div", Lat.FpDiv);
    Set("fp_sqrt", Lat.FpSqrt);
    Set("fp_cvt", Lat.FpCvt);
    Set("store", Lat.Store);
    Set("branch", Lat.Branch);
    Set("call", Lat.Call);
  }

  // ---- functional units ----------------------------------------------------
  const json::Object *FU = Root->getObject("fu");
  if (!FU && Cpu)
    FU = Cpu->getObject("fu");
  if (auto V = getNum(FU, "fp_div_units"))
    FpDivUnits = std::max(1u, (unsigned)*V);
  if (auto V = getNum(FU, "int_div_units"))
    IntDivUnits = std::max(1u, (unsigned)*V);

  // ---- window (docs/interfaces.md §3) -------------------------------------
  if (const json::Value *WV = Root->get("window")) {
    std::vector<WindowConfig> W;
    auto FromObjArray = [&](const json::Array &A) {
      for (const json::Value &E : A) {
        const json::Object *O = E.getAsObject();
        if (!O)
          continue;
        auto R = getNum(O, "rob");
        if (!R)
          continue;
        unsigned ROB = (unsigned)*R;
        WindowConfig C{ROB, ROB / 2, ROB / 4, ROB / 4};
        if (auto V = getNum(O, "iq"))
          C.IQ = (unsigned)*V;
        if (auto V = getNum(O, "lq"))
          C.LQ = (unsigned)*V;
        if (auto V = getNum(O, "sq"))
          C.SQ = (unsigned)*V;
        W.push_back(C);
      }
    };
    if (const json::Object *O = WV->getAsObject()) {
      if (const json::Array *A = O->getArray("configs"))
        FromObjArray(*A);
      else {
        std::vector<unsigned> R = getUIntArray(O, "rob"), I = getUIntArray(O, "iq"),
                              LQ = getUIntArray(O, "lq"), SQ = getUIntArray(O, "sq");
        for (size_t K = 0; K < R.size(); ++K)
          W.push_back({R[K], K < I.size() ? I[K] : R[K] / 2, K < LQ.size() ? LQ[K] : R[K] / 4,
                       K < SQ.size() ? SQ[K] : R[K] / 4});
      }
    } else if (const json::Array *A = WV->getAsArray())
      FromObjArray(*A);
    // An empty result is kept (and rejected by getCachedTargetModel()).
    std::stable_sort(W.begin(), W.end(),
                     [](const WindowConfig &A, const WindowConfig &B) { return A.ROB < B.ROB; });
    Window = W;
  }

  // ---- winhint knobs -----------------------------------------------------
  if (const json::Object *K = Root->getObject("winhint")) {
    if (auto V = getNum(K, "mlp_target"))
      MLPTargetOverride = *V;
    if (auto V = getNum(K, "switch_cost_cycles"))
      SwitchCostCycles = *V;
    if (auto V = getNum(K, "memory_latency_cycles"))
      MemLatency = (unsigned)*V;
    if (auto V = getNum(K, "l1d_mshrs"))
      L1DMSHRs = (unsigned)*V;
  }
  return true;
}

// Documented in TargetModel.h.
const TargetModel &getCachedTargetModel(const std::string &Path, StringRef Tool) {
  static std::map<std::string, std::unique_ptr<TargetModel>> Cache;
  static std::mutex Mu;
  std::lock_guard<std::mutex> G(Mu);
  std::unique_ptr<TargetModel> &TM = Cache[Path];
  if (!TM) {
    auto New = std::make_unique<TargetModel>();
    if (!Path.empty()) {
      std::string Err;
      if (!New->loadJSON(Path, Err)) {
        errs() << Tool << ": warning: " << Err << "; using the built-in default machine\n";
        New = std::make_unique<TargetModel>();
      }
    }
    if (New->Window.empty())
      reportFatalUsageError(Twine(Tool) + ": machine description '" +
                            (Path.empty() ? StringRef("builtin-default") : StringRef(Path)) +
                            "' has an empty window table (the \"window\" section needs at "
                            "least one config with a \"rob\" size)");
    TM = std::move(New);
  }
  return *TM;
}

// Documented in TargetModel.h (as are the lookups below).
unsigned TargetModel::wMax() const { return Window.empty() ? 256 : Window.back().ROB; }

unsigned TargetModel::configForW(unsigned W) const {
  if (Window.empty())
    return 0;
  if (W == 0)
    return Window.size() - 1;
  for (unsigned I = 0; I < Window.size(); ++I)
    if (Window[I].ROB >= W)
      return I;
  return Window.size() - 1;
}

unsigned TargetModel::configCovering(unsigned ROB, unsigned IQ, unsigned LQ, unsigned SQ) const {
  for (unsigned I = 0; I < Window.size(); ++I) {
    const WindowConfig &C = Window[I];
    if (C.ROB >= ROB && C.IQ >= IQ && C.LQ >= LQ && C.SQ >= SQ)
      return I;
  }
  return Window.empty() ? 0 : Window.size() - 1;
}

unsigned TargetModel::configForIQ(unsigned IQ) const {
  for (unsigned I = 0; I < Window.size(); ++I)
    if (Window[I].IQ >= IQ)
      return I;
  return Window.empty() ? 0 : Window.size() - 1;
}

unsigned TargetModel::levelLatency(unsigned Level) const {
  if (Level < Caches.size())
    return Caches[Level].Latency;
  unsigned Last = Caches.empty() ? 0 : Caches.back().Latency;
  return Last + MemLatency;
}

std::string TargetModel::levelName(unsigned Level) const {
  if (Level < Caches.size())
    return Caches[Level].Name;
  return "MEM";
}

double TargetModel::gem5SwitchCost() const {
  if (SwitchCostCycles > 0)
    return SwitchCostCycles;
  // Shrinking gates dispatch until the occupancy drains below the new size:
  // on average half of the largest ROB at the commit width, plus pipeline
  // refill. Growing is nearly free; we charge the mean of both.
  double Drain = (double)wMax() / 2.0 / std::max(1u, CommitWidth);
  return 10.0 + Drain;
}

void TargetModel::print(raw_ostream &OS) const {
  OS << "target '" << Name << "': issue=" << IssueWidth << " dispatch=" << DispatchWidth
     << " commit=" << CommitWidth << " clock=" << format("%.3g", ClockGHz)
     << "GHz line=" << LineSize << " mem=" << MemLatency << "cyc mshrs=" << L1DMSHRs
     << " prefetch=" << (StridePrefetcher ? "stride" : "none")
     << " eff=" << format("%.3g", CacheEffectiveFraction) << "\n  caches:";
  for (const CacheLevel &C : Caches)
    OS << " " << C.Name << "=" << C.SizeBytes / 1024 << "KiB/" << C.Latency << "cyc";
  OS << "\n  latency: int_alu=" << Lat.IntAlu << " int_mul=" << Lat.IntMul
     << " int_div=" << Lat.IntDiv << " fp_add=" << Lat.FpAdd << " fp_mul=" << Lat.FpMul
     << " fp_fma=" << Lat.FpFma << " fp_div=" << Lat.FpDiv << " fp_sqrt=" << Lat.FpSqrt
     << " fp_cvt=" << Lat.FpCvt << " store=" << Lat.Store << " branch=" << Lat.Branch
     << " call=" << Lat.Call << "\n  fu: fp_div=" << FpDivUnits << " int_div=" << IntDivUnits
     << "\n  knobs: mlp_target=" << format("%.3g", MLPTargetOverride)
     << " switch_cost=" << format("%.3g", gem5SwitchCost()) << "\n  window:";
  for (const WindowConfig &C : Window)
    OS << " [" << C.ROB << "/" << C.IQ << "/" << C.LQ << "/" << C.SQ << "]";
  OS << "\n";
}

} // namespace winhint
