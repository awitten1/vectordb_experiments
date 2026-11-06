// Benchmark FAISS index types on the same dataset the Postgres benchmarks use,
// writing the same bench/results/<dataset>.jsonl schema so one plotter reads
// both.
//
// The point of this driver is attribution: FAISS runs the same algorithms
// in-process, so "FAISS IVFFlat with the same lists" against "pgvector IVFFlat"
// is the algorithm either side of the Postgres boundary. It is NOT a fair
// head-to-head on its own -- FAISS calls a function where pgvector crosses a
// socket, a planner and a buffer manager. See the overhead floor in
// bench/FINDINGS.md.
//
// Usage:
//   faiss_bench --data bench/data --dataset bigann_18000000 \
//               --factory "IVF4242,Flat" --sweep 1,2,5,10,20,50,100 \
//               --k 10 --duration 60 --threads 1

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <omp.h>
#include <sys/stat.h>

#include <faiss/AutoTune.h>
#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <faiss/utils/utils.h>

#include "npy.h"

using clk = std::chrono::steady_clock;
using ns = std::chrono::nanoseconds;

namespace {

struct Options {
    std::string data_dir = "bench/data";
    std::string dataset = "bigann_18000000";
    std::string factory = "Flat";
    std::string name;      // defaults to a sanitized factory string
    std::string param;     // defaults to auto-detected (nprobe / efSearch)
    std::string results;   // defaults to <repo>/bench/results/<dataset>.jsonl
    std::vector<int> sweep;
    int k = 10;
    int num_queries = 1000;
    double duration = 60.0;
    int threads = 1;
    long train_size = 0;   // 0 = auto
    int ef_construction = 64;
    bool rebuild = false;
    bool no_save = false;
};

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "error: %s\n", msg.c_str());
    std::exit(1);
}

std::string sanitize(std::string s) {
    for (char& c : s) {
        c = std::isalnum(static_cast<unsigned char>(c))
                    ? char(std::tolower(static_cast<unsigned char>(c)))
                    : '_';
    }
    // collapse runs of '_'
    std::string out;
    for (char c : s) {
        if (c == '_' && !out.empty() && out.back() == '_') continue;
        out += c;
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out;
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

double percentile(std::vector<double>& sorted_ms, double q) {
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
        out += c;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    std::string sweep_arg;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "--data") o.data_dir = next();
        else if (a == "--dataset") o.dataset = next();
        else if (a == "--factory") o.factory = next();
        else if (a == "--name") o.name = next();
        else if (a == "--param") o.param = next();
        else if (a == "--results") o.results = next();
        else if (a == "--sweep") sweep_arg = next();
        else if (a == "--k") o.k = std::stoi(next());
        else if (a == "--num-queries") o.num_queries = std::stoi(next());
        else if (a == "--duration") o.duration = std::stod(next());
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--train-size") o.train_size = std::stol(next());
        else if (a == "--ef-construction") o.ef_construction = std::stoi(next());
        else if (a == "--rebuild") o.rebuild = true;
        else if (a == "--no-save") o.no_save = true;
        else if (a == "--version") {
            std::printf("faiss %s\ncompile options: %s\n", VERSION_STRING,
                        faiss::get_compile_options().c_str());
            return 0;
        } else die("unknown argument " + a);
    }
    if (!sweep_arg.empty()) o.sweep = parse_sweep(sweep_arg);
    if (o.name.empty()) o.name = "faiss_" + sanitize(o.factory);
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
    const faiss::idx_t nb = faiss::idx_t(train.rows());
    const int nq = std::min<int>(o.num_queries, int(test.rows()));
    const int gt_k = int(truth.cols());
    if (o.k > gt_k) die("k exceeds the stored ground-truth neighbours");
    if (int(test.cols()) != d) die("query dimension does not match the train set");

    const float* xb = reinterpret_cast<const float*>(train.data);
    const float* xq = reinterpret_cast<const float*>(test.data);
    const int64_t* gt = reinterpret_cast<const int64_t*>(truth.data);

    std::fprintf(stderr, "dataset %s: %lld x %d, %d queries, k=%d\n",
                 o.dataset.c_str(), (long long)nb, d, nq, o.k);

    // ---- index ------------------------------------------------------------
    const std::string index_dir = o.data_dir + "/faiss";
    ::mkdir(index_dir.c_str(), 0775);
    const std::string index_path = index_dir + "/" + o.dataset + "_" + o.name + ".index";

    const int build_threads = omp_get_max_threads();
    std::unique_ptr<faiss::Index> index;
    double build_s = 0;

    if (exists(index_path) && !o.rebuild) {
        std::fprintf(stderr, "reusing %s (pass --rebuild to time a fresh build)\n",
                     index_path.c_str());
        index.reset(faiss::read_index(index_path.c_str()));
    } else {
        std::fprintf(stderr, "building %s (%s, %d threads) ...\n",
                     o.name.c_str(), o.factory.c_str(), build_threads);
        const auto t0 = clk::now();
        index.reset(faiss::index_factory(d, o.factory.c_str(), faiss::METRIC_L2));

        if (auto* h = dynamic_cast<faiss::IndexHNSW*>(index.get())) {
            h->hnsw.efConstruction = o.ef_construction;
        }

        if (!index->is_trained) {
            long ntrain = o.train_size;
            if (ntrain <= 0) {
                long nlist = 1;
                if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index.get())) {
                    nlist = long(ivf->nlist);
                }
                // FAISS wants >= 39*nlist and is happy at 256*nlist; cap it so
                // k-means on a large nlist does not dominate the build.
                ntrain = std::min<long>(nb, std::min<long>(256 * nlist, 4000000));
                ntrain = std::max<long>(ntrain, std::min<long>(nb, 100000));
            }
            // Random sample rather than a prefix: the prefix of a real dataset
            // is not guaranteed to be an unbiased sample of it. A seen-bitmap
            // with rejection costs 2 MB here, where shuffling an index vector
            // of 18M idx_t would cost 144 MB.
            std::vector<bool> seen(static_cast<size_t>(nb), false);
            std::vector<float> sample(static_cast<size_t>(ntrain) * size_t(d));
            std::mt19937_64 rng(42);
            std::uniform_int_distribution<faiss::idx_t> pick(0, nb - 1);
            for (long i = 0; i < ntrain; i++) {
                faiss::idx_t r;
                do {
                    r = pick(rng);
                } while (seen[static_cast<size_t>(r)]);
                seen[static_cast<size_t>(r)] = true;
                std::memcpy(&sample[static_cast<size_t>(i) * d],
                            xb + static_cast<size_t>(r) * d,
                            sizeof(float) * size_t(d));
            }
            std::fprintf(stderr, "  training on %ld vectors ...\n", ntrain);
            index->train(faiss::idx_t(ntrain), sample.data());
        }

        // Add in chunks so transient memory stays bounded and progress shows.
        const faiss::idx_t chunk = 1000000;
        for (faiss::idx_t i = 0; i < nb; i += chunk) {
            const faiss::idx_t n = std::min(chunk, nb - i);
            index->add(n, xb + i * d);
            std::fprintf(stderr, "\r  added %lld / %lld", (long long)(i + n),
                         (long long)nb);
            std::fflush(stderr);
        }
        std::fprintf(stderr, "\n");
        build_s = std::chrono::duration<double>(clk::now() - t0).count();
        std::fprintf(stderr, "built in %.1fs\n", build_s);
        faiss::write_index(index.get(), index_path.c_str());
    }

    const long index_bytes = file_size(index_path);

    // Which knob this index type sweeps. ParameterSpace dispatches both names
    // uniformly, so the only per-type logic in this program is picking it.
    std::string param = o.param;
    if (param.empty()) {
        if (dynamic_cast<faiss::IndexIVF*>(index.get())) param = "nprobe";
        else if (dynamic_cast<faiss::IndexHNSW*>(index.get())) param = "efSearch";
    }
    if (param.empty() && !o.sweep.empty()) {
        die("index type has no sweep parameter; drop --sweep or pass --param");
    }
    std::vector<int> sweep = o.sweep;
    if (sweep.empty()) sweep.push_back(0);  // a single point, e.g. Flat

    // Single-threaded search, so these numbers share an axis with the
    // single-client Postgres figures. Builds above used every core.
    omp_set_num_threads(o.threads);

    std::printf("\n%16s %8s %10s %9s %9s %9s\n",
                param.empty() ? "(none)" : param.c_str(),
                "recall", "QPS", "p50 ms", "p95 ms", "p99 ms");

    faiss::ParameterSpace ps;
    std::vector<faiss::idx_t> ids(size_t(nq) * size_t(o.k));
    std::vector<float> dis(size_t(nq) * size_t(o.k));
    std::string lines;

    for (int value : sweep) {
        if (!param.empty()) ps.set_index_parameter(index.get(), param, value);

        std::fill(ids.begin(), ids.end(), faiss::idx_t(-1));
        std::vector<double> lat_ms;

        // Warm up outside the clock, exactly as query_worker does.
        index->search(1, xq, o.k, dis.data(), ids.data());

        const auto start = clk::now();
        const auto deadline = start + ns(int64_t(o.duration * 1e9));
        while (true) {
            for (int i = 0; i < nq; i++) {
                const auto t0 = clk::now();
                index->search(1, xq + size_t(i) * d, o.k,
                              dis.data() + size_t(i) * o.k,
                              ids.data() + size_t(i) * o.k);
                lat_ms.push_back(
                        std::chrono::duration<double, std::milli>(clk::now() - t0)
                                .count());
            }
            // Checked BETWEEN passes, never inside one. A partial pass leaves
            // queries unanswered while recall is divided by the whole set,
            // which silently scales recall by coverage -- the bug that made
            // pgvector IVFFlat report 0.485 instead of 0.990 at probes=100.
            if (o.duration <= 0 || clk::now() >= deadline) break;
        }
        const double wall_s = std::chrono::duration<double>(clk::now() - start).count();

        long hits = 0;
        for (int i = 0; i < nq; i++) {
            const faiss::idx_t* got = &ids[size_t(i) * o.k];
            bool answered = false;
            for (int a = 0; a < o.k; a++) {
                if (got[a] < 0) continue;
                answered = true;
                for (int b = 0; b < o.k; b++) {
                    if (got[a] == gt[size_t(i) * gt_k + b]) {
                        hits++;
                        break;
                    }
                }
            }
            if (!answered) die("query " + std::to_string(i) + " returned nothing");
        }
        const double recall = double(hits) / (double(nq) * o.k);
        const double qps = double(lat_ms.size()) / wall_s;

        std::vector<double> sorted = lat_ms;
        std::sort(sorted.begin(), sorted.end());
        const double mean =
                std::accumulate(lat_ms.begin(), lat_ms.end(), 0.0) / double(lat_ms.size());

        std::printf("%16d %8.4f %10.1f %9.3f %9.3f %9.3f\n", value, recall, qps,
                    percentile(sorted, 50), percentile(sorted, 95),
                    percentile(sorted, 99));
        std::fflush(stdout);

        char buf[2048];
        std::snprintf(buf, sizeof buf,
                      "{\"dataset\":\"%s\",\"rows\":%lld,\"dim\":%d,\"metric\":\"euclidean\","
                      "\"num_queries\":%d,\"k\":%d,\"clients\":1,\"duration_s\":%g,"
                      "\"index\":\"%s\",\"engine\":\"faiss\",\"faiss_factory\":\"%s\","
                      "\"faiss_version\":\"%s\",\"threads\":%d,\"build_threads\":%d,"
                      "\"build_s\":%s,\"index_bytes\":%ld,\"build_params\":{},"
                      "\"search_param\":%s%s%s,\"search_value\":%s,"
                      "\"recall\":%.5f,\"qps\":%.2f,\"queries_executed\":%zu,"
                      "\"latency_ms\":{\"mean\":%.4f,\"p50\":%.4f,\"p95\":%.4f,\"p99\":%.4f},"
                      "\"timestamp\":\"%s\"}\n",
                      json_escape(o.dataset).c_str(), (long long)nb, d, nq, o.k,
                      o.duration > 0 ? o.duration : 0.0,
                      json_escape(o.name).c_str(), json_escape(o.factory).c_str(),
                      VERSION_STRING, o.threads, build_threads,
                      build_s > 0 ? std::to_string(build_s).c_str() : "null",
                      index_bytes,
                      param.empty() ? "" : "\"", param.empty() ? "null" : param.c_str(),
                      param.empty() ? "" : "\"",
                      param.empty() ? "null" : std::to_string(value).c_str(),
                      recall, qps, lat_ms.size(), mean, percentile(sorted, 50),
                      percentile(sorted, 95), percentile(sorted, 99), iso_now().c_str());
        lines += buf;
    }

    if (!o.no_save) {
        FILE* f = std::fopen(o.results.c_str(), "a");
        if (!f) die("cannot append to " + o.results);
        std::fwrite(lines.data(), 1, lines.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "appended %zu rows to %s\n", sweep.size(),
                     o.results.c_str());
    }
    return 0;
}
