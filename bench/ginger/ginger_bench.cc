// Benchmark Ginger on the same dataset files the Postgres and FAISS runs use,
// appending to the same bench/results/<dataset>.jsonl schema, so one plotter
// reads every engine.
//
// Ginger is an in-process library: a query is a function call. Postgres
// crosses a socket, a planner and a buffer manager for every query. The raw
// QPS ratio therefore includes that boundary; run_experiment.sh's `overhead`
// phase measures it separately so it can be subtracted when reading charts.
//
// Usage:
//   ginger_bench --data bench/data --dataset dbpedia_openai_1m_1536_angular \
//                --config recommended --sweep 10,20,40,80,160,320 \
//                --k 10 --duration 60 --threads 1
//
// Configurations:
//   recommended   RecommendGingerConfig(d, SQUARED_L2, alpha): PiPNN graph,
//                 graph compression + reordering, 4-bit TurboQuant traversal,
//                 12-bit TurboQuant rerank, PCA when d >= 512, float data
//                 dropped. Ginger's own design point.
//   float-rerank  The same graph recipe, but no PCA, no extra bits, and the
//                 float dataset kept for an exact float rerank -- closer to
//                 what the Postgres indexes compute.
//   keep-floats   recommended, but the float dataset is kept, so a search may
//                 traverse or re-rank with floats (see --traversal, --rerank)
//   no-compression  recommended without graph compression
//   no-reorder    recommended without graph reordering or compression
//
// Search toggles (default: Ginger's DefaultSearchOptions for the index):
//   --traversal tq|float      score candidates with 4-bit codes or floats
//   --rerank tq|float|none    re-rank with 12-bit codes, floats, or not at all
//   --rerank-factor F         re-rank the top F*k; 0 re-ranks the whole beam
//   --label NAME              result rows are labelled ginger-NAME (default:
//                             the config name)
//
// Searching a graph built elsewhere (--graph FILE, a graph file written by
// pgvectorfast's tools/pipnn_export over this dataset's table):
//   Ginger searches that graph, from its entry point, instead of building or
//   loading an index. Node i must be dataset row i, as for a table loaded in
//   dataset order (recall shows it if not).
//   --search float   float distances, no re-rank: what pgvectorfast's float
//                    HNSW index computes on the same graph
//   --search native  Ginger's own search: 4-bit TurboQuant traversal, 12-bit
//                    re-rank of the top 6k (the recommended config's codes)
//   Rows are labelled ginger-graph-<float|native>.
//
// Instrumentation (off by default; neither changes the timed run):
//   --stats              after each timed point, one untimed single-thread
//                        pass over the queries with Ginger's search
//                        statistics (nodes expanded, candidates, distance
//                        computations, reranked points); their means go in
//                        the result row as "stats" and to stderr.
//   GINGER_PROGRESS=<f>  a thread rewrites <f> every 100 ms with
//                        "<queries answered> <unix time>", so a profiler
//                        attached for a window can compute its own QPS.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ginger/ginger.h"
#include "google/protobuf/text_format.h"

#include "npy.h"

using clk = std::chrono::steady_clock;
using ns = std::chrono::nanoseconds;

namespace {

// Queries answered so far, for GINGER_PROGRESS.
std::atomic<long> g_answered {0};

struct Options {
    std::string data_dir = "bench/data";
    std::string dataset;
    std::string config = "recommended";
    std::string results;  // defaults to bench/results/<dataset>.jsonl
    std::vector<int> sweep = {64};
    int k = 10;
    int num_queries = 1000;
    double duration = 60.0;
    int threads = 1;
    float alpha = 1.08f;
    bool rebuild = false;
    bool no_save = false;
    bool stats = false;
    std::string graph;            // --graph: search this graph file
    std::string search = "native";
    std::string traversal;        // --traversal: "", tq or float
    std::string rerank;           // --rerank: "", tq, float or none
    float rerank_factor = -1;     // --rerank-factor: < 0 keeps the default
    std::string label;            // --label: overrides the config name in labels
};

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "error: %s\n", msg.c_str());
    std::exit(1);
}

void check(const absl::Status& s, const std::string& what) {
    if (!s.ok()) die(what + ": " + std::string(s.message()));
}

std::vector<int> parse_sweep(const std::string& s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        out.push_back(std::stoi(s.substr(i, j - i)));
        i = j + 1;
    }
    return out;
}

std::string iso_now() {
    const std::time_t t = std::time(nullptr);
    std::tm tm {};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S+00:00", &tm);
    return buf;
}

long file_size(const std::string& p) {
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0 ? long(st.st_size) : -1L;
}

bool exists(const std::string& p) {
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0;
}

double percentile(const std::vector<double>& sorted_ms, double q) {
    if (sorted_ms.empty()) return 0.0;
    // numpy's linear interpolation, so the numbers line up with ann_bench.py
    const double pos = q / 100.0 * double(sorted_ms.size() - 1);
    const size_t lo = size_t(std::floor(pos));
    const size_t hi = size_t(std::ceil(pos));
    return sorted_ms[lo] + (sorted_ms[hi] - sorted_ms[lo]) * (pos - double(lo));
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (c == '\n') { out += "\\n"; continue; }
        out += c;
    }
    return out;
}

std::string one_line(const google::protobuf::Message& m) {
    google::protobuf::TextFormat::Printer p;
    p.SetSingleLineMode(true);
    std::string out;
    if (!p.PrintToString(m, &out)) die("cannot print a proto as text");
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

// ---- memory guard ---------------------------------------------------------
// The comparison is defined as fully in memory. These mirror ann_bench.py's
// checks: no swap device at all when REQUIRE_NO_SWAP=1, and no swapping or
// major faults inside any measured window.

struct VmStat {
    long pswpin = 0, pswpout = 0, pgmajfault = 0;
    bool ok = false;
};

VmStat vmstat() {
    VmStat v;
    FILE* f = std::fopen("/proc/vmstat", "r");
    if (!f) return v;
    char key[64];
    long val;
    while (std::fscanf(f, "%63s %ld", key, &val) == 2) {
        if (!std::strcmp(key, "pswpin")) v.pswpin = val;
        else if (!std::strcmp(key, "pswpout")) v.pswpout = val;
        else if (!std::strcmp(key, "pgmajfault")) v.pgmajfault = val;
    }
    std::fclose(f);
    v.ok = true;
    return v;
}

// A field of /proc/meminfo or /proc/self/status, in kB (-1 if absent).
long proc_kb(const char* path, const char* field) {
    FILE* f = std::fopen(path, "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    const size_t n = std::strlen(field);
    while (std::fgets(line, sizeof line, f)) {
        if (!std::strncmp(line, field, n) && line[n] == ':') {
            kb = std::atol(line + n + 1);
            break;
        }
    }
    std::fclose(f);
    return kb;
}

void require_no_swap() {
    const char* req = std::getenv("REQUIRE_NO_SWAP");
    if (!req || std::strcmp(req, "1") != 0) return;
    FILE* f = std::fopen("/proc/swaps", "r");
    if (!f) return;
    char line[512];
    int lines = 0;
    while (std::fgets(line, sizeof line, f)) lines++;
    std::fclose(f);
    if (lines > 1) die("swap is enabled; run 'sudo swapoff -a' (REQUIRE_NO_SWAP=1)");
}

ginger::GingerSearcherConfig make_config(const Options& o, int d) {
    absl::StatusOr<ginger::GingerSearcherConfig> c =
            ginger::RecommendGingerConfig(d, ginger::SQUARED_L2, o.alpha);
    check(c.status(), "RecommendGingerConfig");
    ginger::GingerSearcherConfig config = *std::move(c);
    if (o.config == "recommended") return config;
    if (o.config == "float-rerank") {
        config.clear_pca();
        config.mutable_turbo_quant()->set_extra_bits(0);
        config.set_drop_dataset(false);
        return config;
    }
    if (o.config == "keep-floats") {
        config.set_drop_dataset(false);
        return config;
    }
    if (o.config == "no-compression") {
        config.set_use_graph_compression(false);
        return config;
    }
    if (o.config == "no-reorder") {
        config.set_use_graph_compression(false);
        config.set_reordering_algorithm(ginger::REORDERING_ALGORITHM_NONE);
        return config;
    }
    die("unknown --config " + o.config +
        " (want recommended, float-rerank, keep-floats, no-compression or no-reorder)");
}

// One sweep point's raw measurements.
struct Run {
    std::vector<std::vector<ginger::DistanceAndId>> results;  // last answer per query
    std::vector<double> lat_ms;
    double wall_s = 0;
};

// `threads` workers split the query set by stride. Each worker repeats full
// passes over its share and checks the deadline only BETWEEN passes, so every
// query is answered at least once. A partial pass would leave queries
// unanswered while recall is divided by the whole set -- the bug that made
// pgvector IVFFlat report 0.485 instead of 0.990 at probes=100.
Run run_point(const ginger::GingerSearcher& s, const ginger::SearchOptions& opts,
              const float* xq, int nq, int d, int threads, double duration) {
    Run run;
    run.results.resize(size_t(nq));
    std::vector<std::vector<double>> lat(static_cast<size_t>(threads));
    std::atomic<bool> failed {false};

    // Warm up outside the clock, as ann_bench.py's query_worker does.
    (void)s.FindNeighbors(absl::MakeConstSpan(xq, size_t(d)), opts);

    const auto start = clk::now();
    const auto deadline = start + ns(int64_t(duration * 1e9));
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) {
        pool.emplace_back([&, t] {
            std::vector<double>& mine = lat[size_t(t)];
            while (true) {
                for (int i = t; i < nq; i += threads) {
                    const auto t0 = clk::now();
                    absl::StatusOr<std::vector<ginger::DistanceAndId>> r = s.FindNeighbors(
                            absl::MakeConstSpan(xq + size_t(i) * size_t(d), size_t(d)), opts);
                    g_answered.fetch_add(1, std::memory_order_relaxed);
                    mine.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0)
                                           .count());
                    if (!r.ok()) {
                        std::fprintf(stderr, "query %d failed: %s\n", i,
                                     std::string(r.status().message()).c_str());
                        failed = true;
                        return;
                    }
                    run.results[size_t(i)] = *std::move(r);
                }
                if (duration <= 0 || clk::now() >= deadline) break;
            }
        });
    }
    for (auto& th : pool) th.join();
    run.wall_s = std::chrono::duration<double>(clk::now() - start).count();
    if (failed) die("a query failed; see above");
    for (auto& v : lat) run.lat_ms.insert(run.lat_ms.end(), v.begin(), v.end());
    return run;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "--data") o.data_dir = next();
        else if (a == "--dataset") o.dataset = next();
        else if (a == "--config") o.config = next();
        else if (a == "--results") o.results = next();
        else if (a == "--sweep") o.sweep = parse_sweep(next());
        else if (a == "--k") o.k = std::stoi(next());
        else if (a == "--num-queries") o.num_queries = std::stoi(next());
        else if (a == "--duration") o.duration = std::stod(next());
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--alpha") o.alpha = std::stof(next());
        else if (a == "--rebuild") o.rebuild = true;
        else if (a == "--no-save") o.no_save = true;
        else if (a == "--stats") o.stats = true;
        else if (a == "--graph") o.graph = next();
        else if (a == "--search") o.search = next();
        else if (a == "--traversal") o.traversal = next();
        else if (a == "--rerank") o.rerank = next();
        else if (a == "--rerank-factor") o.rerank_factor = std::stof(next());
        else if (a == "--label") o.label = next();
        else if (a == "--version") {
#ifdef __AVX512F__
            std::puts("isa: AVX-512");
#elif defined(__AVX2__)
            std::puts("isa: AVX2");
#else
            std::puts("isa: baseline (no AVX2!)");
#endif
#ifdef __AVXVNNI__
            std::puts("vnni: AVX-VNNI");
#elif defined(__AVX512VNNI__)
            std::puts("vnni: AVX512-VNNI");
#else
            std::puts("vnni: none");
#endif
            std::printf("compiler: %s\n", __VERSION__);
            return 0;
        } else die("unknown argument " + a);
    }
    if (o.dataset.empty()) die("--dataset is required");
    require_no_swap();
    if (o.threads < 1) die("--threads must be >= 1");
    if (o.results.empty()) o.results = "bench/results/" + o.dataset + ".jsonl";

    // ---- dataset ----------------------------------------------------------
    const std::string train_path = o.data_dir + "/" + o.dataset + "_train.npy";
    const std::string meta_path = o.data_dir + "/" + o.dataset + "_meta.npz";
    npy::Array train = npy::mmap_npy(train_path);
    npy::Array test = npy::load_npz(meta_path, "test");
    npy::Array truth = npy::load_npz(meta_path, "neighbors");
    train.expect("<f4", train_path);
    test.expect("<f4", meta_path + ":test");
    truth.expect("<i8", meta_path + ":neighbors");

    const int d = int(train.cols());
    const long nb = long(train.rows());
    const int nq = std::min<int>(o.num_queries, int(test.rows()));
    const int gt_k = int(truth.cols());
    if (o.k > gt_k) die("k exceeds the stored ground-truth neighbours");
    if (int(test.cols()) != d) die("query dimension does not match the train set");

    const float* xb = reinterpret_cast<const float*>(train.data);
    const float* xq = reinterpret_cast<const float*>(test.data);
    const int64_t* gt = reinterpret_cast<const int64_t*>(truth.data);
    std::fprintf(stderr, "dataset %s: %ld x %d, %d queries, k=%d, config %s\n",
                 o.dataset.c_str(), nb, d, nq, o.k, o.config.c_str());

    // ---- index ------------------------------------------------------------
    const std::string index_dir = o.data_dir + "/ginger";
    ::mkdir(index_dir.c_str(), 0775);
    char alpha_s[16];
    std::snprintf(alpha_s, sizeof alpha_s, "%.2f", o.alpha);
    const bool graph_mode = !o.graph.empty();
    if (graph_mode && o.search != "float" && o.search != "native")
        die("--search must be float or native");
    const std::string name = graph_mode ? "ginger_graph_" + o.search
                                        : "ginger_" + o.config + "_a" + alpha_s;
    const std::string index_path =
            graph_mode ? o.graph : index_dir + "/" + o.dataset + "_" + name + ".ginger";
    const std::string label = !o.label.empty() ? o.label : graph_mode ? "graph-" + o.search : o.config;
    if (!o.traversal.empty() && o.traversal != "tq" && o.traversal != "float")
        die("--traversal must be tq or float");
    if (!o.rerank.empty() && o.rerank != "tq" && o.rerank != "float" && o.rerank != "none")
        die("--rerank must be tq, float or none");

    const char* pnt = std::getenv("PARLAY_NUM_THREADS");
    const int build_threads = pnt ? std::atoi(pnt) : int(std::thread::hardware_concurrency());
    std::unique_ptr<ginger::GingerSearcher> searcher;
    double build_s = 0;

    if (graph_mode) {
        // Header (see pipnn_export): magic, version, max degree, nodes, entry,
        // dimensions; then a TID per node; then (max degree + 1) uint32 per
        // node, degree first -- FlatGraph's own layout.
        FILE* f = std::fopen(o.graph.c_str(), "rb");
        if (!f) die("cannot open " + o.graph);
        char magic[8];
        uint32_t version = 0, max_degree = 0, dims = 0, reserved = 0;
        uint64_t num_nodes = 0, entry = 0;
        if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "PIPNNGR1", 8) != 0 ||
            std::fread(&version, 4, 1, f) != 1 || version != 1 ||
            std::fread(&max_degree, 4, 1, f) != 1 || std::fread(&num_nodes, 8, 1, f) != 1 ||
            std::fread(&entry, 8, 1, f) != 1 || std::fread(&dims, 4, 1, f) != 1 ||
            std::fread(&reserved, 4, 1, f) != 1)
            die(o.graph + " is not a graph file written by pipnn_export");
        // Bit 0 of the last header field: nodes in Ginger's reordered order.
        // Here node i must be dataset row i, so such a file can't be searched.
        if (reserved & 1u)
            die(o.graph + " was written with --reorder: its node i is not dataset row i");
        if (long(num_nodes) != nb || int(dims) != d)
            die("the graph file has " + std::to_string(num_nodes) + " nodes of " +
                std::to_string(dims) + " dimensions; the dataset " + std::to_string(nb) +
                " of " + std::to_string(d));
        if (std::fseek(f, long(40 + num_nodes * 8), SEEK_SET) != 0) die("cannot seek in " + o.graph);
        ginger::FlatGraph g(static_cast<ginger::GingerNodeId>(num_nodes),
                            static_cast<ginger::GingerNodeId>(max_degree));
        absl::Span<ginger::GingerNodeId> data = g.MutableData();
        if (std::fread(data.data(), sizeof(ginger::GingerNodeId), data.size(), f) != data.size())
            die(o.graph + " ends early");
        std::fclose(f);
        for (size_t u = 0; u < num_nodes; u++)
            if (uint32_t(data[u * (max_degree + 1)]) > max_degree) die("bad degree in " + o.graph);

        // The floats in anonymous memory, not the mmapped file (see below).
        auto ds = std::make_shared<ginger::SpanDataset>(ginger::GingerNodeId(nb), size_t(d));
        for (long i = 0; i < nb; i++)
            ds->Set<float>(ginger::GingerNodeId(i), absl::MakeConstSpan(xb + size_t(i) * size_t(d), size_t(d)));

        ginger::GingerSearcherConfig config;
        if (o.search == "native") {
            config = make_config(o, d);  // the recommended config: its TurboQuant codes
            config.set_reordering_algorithm(ginger::REORDERING_ALGORITHM_NONE);
            config.set_use_graph_compression(false);
        } else {
            config.set_distance_metric(ginger::SQUARED_L2);
            config.set_drop_dataset(false);
        }
        std::fprintf(stderr, "searching %s (%llu nodes, max degree %u, entry %llu), %s search:\n  %s\n",
                     o.graph.c_str(), (unsigned long long) num_nodes, max_degree,
                     (unsigned long long) entry, o.search.c_str(), one_line(config).c_str());
        const auto t0 = clk::now();
        absl::StatusOr<std::unique_ptr<ginger::GingerSearcher>> s =
                ginger::GingerSearcherBuilder(config)
                        .AddDataset(ds)
                        .AddIndex(std::make_unique<const ginger::FlatGraph>(std::move(g)))
                        .SetEntryPoint(ginger::GingerNodeId(entry))
                        .Build();
        check(s.status(), "Build");
        std::fprintf(stderr, "prepared in %.1fs\n", std::chrono::duration<double>(clk::now() - t0).count());
        searcher = *std::move(s);
    } else if (exists(index_path) && !o.rebuild) {
        std::fprintf(stderr, "reusing %s (pass --rebuild to time a fresh build)\n",
                     index_path.c_str());
        absl::StatusOr<std::unique_ptr<ginger::GingerSearcher>> s =
                ginger::io::LoadGingerSearcher(index_path);
        check(s.status(), "LoadGingerSearcher(" + index_path + ")");
        searcher = *std::move(s);
    } else {
        const ginger::GingerSearcherConfig config = make_config(o, d);
        std::fprintf(stderr, "building %s with %d threads:\n  %s\n", name.c_str(),
                     build_threads, one_line(config).c_str());
        // Wrap the mmapped train set: no copy. It outlives the searcher
        // because `train` lives until main returns.
        absl::StatusOr<std::shared_ptr<const ginger::SpanDataset>> ds =
                ginger::SpanDataset::Wrap(absl::MakeConstSpan(xb, size_t(nb) * size_t(d)),
                                          size_t(d));
        check(ds.status(), "SpanDataset::Wrap");
        const auto t0 = clk::now();
        absl::StatusOr<std::unique_ptr<ginger::GingerSearcher>> s =
                ginger::GingerSearcherBuilder(config).AddDataset(*ds).Build();
        check(s.status(), "Build");
        build_s = std::chrono::duration<double>(clk::now() - t0).count();
        std::fprintf(stderr, "built in %.1fs\n", build_s);
        check(ginger::io::SaveGingerSearcher(**s, index_path), "SaveGingerSearcher");
        // Query from a copy loaded out of the saved file, never from the
        // just-built searcher. A searcher built over the Wrap()ed train set
        // may still reference the mmapped .npy, whose pages the kernel can
        // drop and re-read from disk mid-measurement; a loaded searcher holds
        // everything in anonymous memory, which with swap off cannot leave RAM.
        // It also makes the measured searcher identical whether or not this
        // run did the build.
        s->reset();
        absl::StatusOr<std::unique_ptr<ginger::GingerSearcher>> loaded =
                ginger::io::LoadGingerSearcher(index_path);
        check(loaded.status(), "LoadGingerSearcher(" + index_path + ")");
        searcher = *std::move(loaded);
    }
    // The train set is not needed at query time; release the mapping so
    // nothing in the measured process is file-backed except the binary.
    train = npy::Array {};
    const long index_bytes = file_size(index_path);
    const std::string config_text = one_line(searcher->config());

    char host[256] = "unknown";
    ::gethostname(host, sizeof host - 1);

    // Everything the measurement touches is now resident: the index was read
    // into anonymous memory and the train mapping released. Leave a margin so
    // the query threads' own allocations cannot push the box into reclaim.
    const long rss_kb = proc_kb("/proc/self/status", "VmRSS");
    const long avail_kb = proc_kb("/proc/meminfo", "MemAvailable");
    std::fprintf(stderr, "resident: %.2f GiB, MemAvailable: %.2f GiB\n", rss_kb / 1048576.0,
                 avail_kb / 1048576.0);
    if (avail_kb >= 0 && avail_kb < 2L * 1024 * 1024) {
        die("under 2 GiB of MemAvailable with the index loaded; the run would not be in memory");
    }

    // ---- sweep ------------------------------------------------------------
    std::atomic<bool> progress_stop {false};
    std::thread progress;
    if (const char* pf = std::getenv("GINGER_PROGRESS"); pf && *pf) {
        const std::string path = pf, tmp = path + ".tmp";
        progress = std::thread([&progress_stop, path, tmp] {
            while (!progress_stop.load()) {
                if (FILE* f = std::fopen(tmp.c_str(), "w")) {
                    const double now = std::chrono::duration<double>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
                    std::fprintf(f, "%ld %.3f\n", g_answered.load(), now);
                    std::fclose(f);
                    std::rename(tmp.c_str(), path.c_str());
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }
    std::printf("\n%16s %8s %10s %9s %9s %9s\n", "beam_size", "recall", "QPS", "p50 ms",
                "p95 ms", "p99 ms");
    std::string lines;
    for (int beam : o.sweep) {
        if (beam < o.k) die("beam_size " + std::to_string(beam) + " < k");
        ginger::SearchOptions opts = searcher->DefaultSearchOptions(o.k, beam);
        if (graph_mode && o.search == "float") {
            // pgvectorfast's float index: float distances, no re-rank
            opts.set_search_space(ginger::SEARCH_SPACE_FLOAT);
            opts.set_reranking_strategy(ginger::RERANKING_STRATEGY_NONE);
            opts.clear_rerank_factor();
        }
        if (o.traversal == "tq") opts.set_search_space(ginger::SEARCH_SPACE_TURBO_QUANT);
        if (o.traversal == "float") opts.set_search_space(ginger::SEARCH_SPACE_FLOAT);
        if (o.rerank == "tq") opts.set_reranking_strategy(ginger::RERANKING_STRATEGY_TURBO_QUANT_EXTRA_BITS);
        if (o.rerank == "float") opts.set_reranking_strategy(ginger::RERANKING_STRATEGY_FLOAT);
        if (o.rerank == "none") {
            opts.set_reranking_strategy(ginger::RERANKING_STRATEGY_NONE);
            opts.clear_rerank_factor();
        }
        if (o.rerank_factor == 0) opts.clear_rerank_factor();
        if (o.rerank_factor > 0) opts.set_rerank_factor(o.rerank_factor);
        const VmStat vm0 = vmstat();
        Run run = run_point(*searcher, opts, xq, nq, d, o.threads, o.duration);
        const VmStat vm1 = vmstat();
        std::string stats_json;
        if (o.stats) {
            double v = 0, c = 0, dc = 0, fc = 0, rr = 0;
            for (int i = 0; i < nq; i++) {
                ginger::BeamSearchResult::Statistics st;
                absl::StatusOr<std::vector<ginger::DistanceAndId>> r = searcher->FindNeighbors(
                        absl::MakeConstSpan(xq + size_t(i) * size_t(d), size_t(d)), opts, &st);
                check(r.status(), "FindNeighbors with statistics");
                v += double(st.num_nodes_visited);
                c += double(st.num_candidates_considered);
                dc += double(st.num_distance_computations);
                fc += double(st.num_filter_computations);
                rr += double(st.num_reranked);
            }
            char sb[512];
            std::snprintf(sb, sizeof sb,
                          "\"stats\":{\"nodes_visited\":%.2f,\"candidates_considered\":%.2f,"
                          "\"distance_computations\":%.2f,\"filter_computations\":%.2f,"
                          "\"reranked\":%.2f},",
                          v / nq, c / nq, dc / nq, fc / nq, rr / nq);
            stats_json = sb;
            std::fprintf(stderr, "beam_size %d per query: %.1f nodes expanded, %.1f candidates, "
                         "%.1f distance computations, %.1f reranked\n",
                         beam, v / nq, c / nq, dc / nq, rr / nq);
        }
        const long swapped = (vm1.pswpin - vm0.pswpin) + (vm1.pswpout - vm0.pswpout);
        const long majflt = vm1.pgmajfault - vm0.pgmajfault;
        if (vm0.ok && swapped > 0) {
            die("the machine swapped during beam_size " + std::to_string(beam) + " (" +
                std::to_string(swapped) + " pages); the run is not in memory");
        }

        long hits = 0;
        for (int i = 0; i < nq; i++) {
            const auto& got = run.results[size_t(i)];
            if (got.empty()) die("query " + std::to_string(i) + " returned nothing");
            for (const ginger::DistanceAndId& r : got) {
                for (int b = 0; b < o.k; b++) {
                    if (int64_t(r.id) == gt[size_t(i) * size_t(gt_k) + size_t(b)]) {
                        hits++;
                        break;
                    }
                }
            }
        }
        const double recall = double(hits) / (double(nq) * o.k);
        const double qps = double(run.lat_ms.size()) / run.wall_s;
        std::vector<double> sorted = run.lat_ms;
        std::sort(sorted.begin(), sorted.end());
        const double mean =
                std::accumulate(sorted.begin(), sorted.end(), 0.0) / double(sorted.size());

        std::printf("%16d %8.4f %10.1f %9.3f %9.3f %9.3f\n", beam, recall, qps,
                    percentile(sorted, 50), percentile(sorted, 95), percentile(sorted, 99));
        std::fflush(stdout);

        char buf[32768];
        const int len = std::snprintf(
                buf, sizeof buf,
                "{\"dataset\":\"%s\",\"rows\":%ld,\"dim\":%d,\"metric\":\"euclidean\","
                "\"num_queries\":%d,\"k\":%d,\"clients\":%d,\"duration_s\":%g,"
                "\"index\":\"%s\",\"engine\":\"ginger\",\"label\":\"ginger-%s\","
                "\"ginger_config\":\"%s\",\"ginger_options\":\"%s\",\"threads\":%d,"
                "\"build_threads\":%d,\"build_s\":%s,\"index_bytes\":%ld,"
                "\"build_params\":{\"alpha\":%.2f},"
                "\"search_param\":\"beam_size\",\"search_value\":%d,"
                "\"recall\":%.5f,\"qps\":%.2f,\"queries_executed\":%zu,"
                "\"latency_ms\":{\"mean\":%.4f,\"p50\":%.4f,\"p95\":%.4f,\"p99\":%.4f},"
                "\"memory\":{\"pswpin\":%ld,\"pswpout\":%ld,\"pgmajfault\":%ld,\"rss_kb\":%ld},"
                "%s\"host\":\"%s\",\"timestamp\":\"%s\"}\n",
                json_escape(o.dataset).c_str(), nb, d, nq, o.k, o.threads,
                o.duration > 0 ? o.duration : 0.0, name.c_str(), label.c_str(),
                json_escape(config_text).c_str(), json_escape(one_line(opts)).c_str(),
                o.threads, build_threads,
                build_s > 0 ? std::to_string(build_s).c_str() : "null", index_bytes, o.alpha,
                beam, recall, qps, run.lat_ms.size(), mean, percentile(sorted, 50),
                percentile(sorted, 95), percentile(sorted, 99), vm1.pswpin - vm0.pswpin,
                vm1.pswpout - vm0.pswpout, majflt, proc_kb("/proc/self/status", "VmRSS"),
                stats_json.c_str(), json_escape(host).c_str(),
                iso_now().c_str());
        if (len < 0 || size_t(len) >= sizeof buf) die("result row too long for the buffer");
        lines += buf;
    }

    progress_stop = true;
    if (progress.joinable()) progress.join();
    if (!o.no_save) {
        FILE* f = std::fopen(o.results.c_str(), "a");
        if (!f) die("cannot append to " + o.results);
        std::fwrite(lines.data(), 1, lines.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "appended %zu rows to %s\n", o.sweep.size(), o.results.c_str());
    }
    return 0;
}
