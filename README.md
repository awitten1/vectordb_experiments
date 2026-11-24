# vectordb_experiments

Benchmarks of vector search in Postgres (stock pgvector and pgvectorfast)
against Ginger and other engines. `bench/cloudlab/` holds the scripts behind
the Hacker News study: stock pgvector,
pgvectorfast and Ginger on 27.4 million 384-dimensional MiniLM vectors,
measured on two CloudLab c220g2 machines. The results and their explanation are
in [`bench/FINDINGS-hackernews.md`](bench/FINDINGS-hackernews.md). This page
explains how the experiments are run and what each script does. `bench/README.md`
covers the benchmark client itself and the earlier studies' write-ups.

Some scripts are from earlier studies on smaller machines (DBpedia-OpenAI and
SIFT on m510 nodes). They are listed at the end.

## The setup

Three machines take part:

- **The workstation** (your laptop) runs the driver scripts. It never measures
  anything. It starts steps on the two nodes over SSH and waits for them.
- **The server node** runs Postgres, with the table and index held entirely in
  shared memory.
- **The client node** runs the benchmark clients, which send queries to the
  server over the 10 GbE experiment network. Ginger also runs here, inside its
  own process with no network in between, with Postgres stopped on this node.

The two nodes are named in [`bench/cloudlab/nodes-wisc.sh`](bench/cloudlab/nodes-wisc.sh): host names, the
server's address on the experiment network, and the CloudLab domain. A new
allocation means editing that file.

All the code under test is in this repository or its submodules. Clone with
`git clone --recursive` to get all of it:

| Folder | What it is | Pinned to |
|---|---|---|
| `pgvector` | Stock pgvector | v0.8.7 |
| `pgvectorfast` | pgvectorfast, as the main study measured it | an early master commit |
| `pgvectorfast_pipnn` | pgvectorfast that can import a graph file into an HNSW index | branch `pipnn-graph-import` of awitten1/pgvectorfast |
| `pgvectorfast_ntq` | pgvectorfast with 4-bit TurboQuant indexes (`quantizer = 'tq4'`) | branch `pgvectorfast-ntq` |
| `pgvector_ntq` | pgvector with TurboQuant navigation codes, used for the early tq4 arm | branch `hnsw-tq4-nav` of awitten1/pgvector |
| `ginger` | Ginger, including its PiPNN graph builder | main |
| `faiss`, `pg_vectorsearch` | Other engines, used in the earlier studies | |

The first pgvectorfast + TurboQuant prototype is not a submodule. It lives on
branch `tq4-prototype` of awitten1/pgvectorfast. It holds the optimized AVX2
kernels, Calico's translation arrays on 2 MB pages, and the pipelined hop,
which the later port doesn't have. To rerun the scripts that used it
(`hn_fastntq*.sh`, `hn_calico_thp.sh`, `hn_pipe_hop.sh`, `hn_tlb_diag.sh`),
check that branch out in the `pgvectorfast_ntq` submodule first. The comment in
[`build_fastntq.sh`](build_fastntq.sh) has the commands.

## Preparing a new pair of nodes

1. Put the new host names and the server's experiment-network address in
   `nodes-wisc.sh`.
2. Create `/mydata` on each node's SSD. The repository, Postgres's data and the
   dataset all live there.
   ```
   . bench/cloudlab/nodes-wisc.sh
   ssh <node> 'bash -s -- disk' < bench/cloudlab/setup_node.sh
   ```
3. Copy the repository to both nodes: `bench/cloudlab/sync.sh push`. The nodes
   get the working tree as it is, uncommitted changes included. Nothing built
   or measured is copied.
4. Run `bench/cloudlab/hn_qps.sh`. Its first stage runs `setup_node.sh` on both
   nodes and builds everything:
   - `setup_node.sh` installs packages and turns swap off. It pins every core at
     the base clock (2.6 GHz on the c220g2) with turbo and deep sleep states
     off, and records the machine's details in `bench/results/env-*.txt`.
   - Postgres and both pgvector builds go on the server, Ginger on the client.
   - The dataset is downloaded and prepared on both nodes, which must produce
     byte-identical copies.

`SMOKE=1` in front of any driver switches to a 1-million-row subset with
5-second points, to check a pipeline end to end before committing hours to it.

## How a driver works

Every `bench/cloudlab/hn_*.sh` script is a driver. It runs on the workstation, typically as
`caffeinate -i bench/cloudlab/<driver>.sh`, and is a list of steps. The
machinery is in [`bench/cloudlab/lib.sh`](bench/cloudlab/lib.sh).

- **Steps run detached on the nodes.** Each step runs under `nohup` on its
  node and writes `/mydata/steps/<step>.log`, then `/mydata/steps/<step>.done`
  with its exit status. The driver only starts steps and checks for those
  files. If the laptop sleeps or loses its connection, nothing is left half
  done: the driver just stops advancing.
- **Drivers can be resumed.** Running a driver again skips every step that
  already succeeded and waits for any that are still running. To redo
  finished steps, change the run number (`RUN=2`), which changes the step
  names.
- **SSH trouble stops the driver.** After two failed SSH attempts in a row the
  driver gives up rather than retrying. Steps already running on the nodes
  continue.
- **Don't edit a driver while it runs.** Bash reads a script as it goes, so
  editing a running script corrupts what it does next. To change the plan,
  stop the driver (its detached steps keep going), edit, and start it again.

Every step loads the settings in [`bench/cloudlab/hn.env`](bench/cloudlab/hn.env) on its node: the dataset,
the search widths to sweep (ef_search 10 to 1000), 60 seconds per point, 1,000
queries, 112 GB of shared_buffers, and the index build settings.

### Guards that make a measurement valid

- `require_measurement_state` refuses to start unless both nodes have their
  clock pinned and swap off.
- Every server start prewarms the table and index and checks they fit in
  shared_buffers (`prewarm_or_die` in `run_experiment.sh`), so no query ever
  reads from disk.
- Postgres runs with its memory interleaved across both sockets.
- Before Ginger or a client sweep starts, the client node is checked to be
  idle.

### One measurement

A measurement is two steps:

1. **On the server:** `bench/run_experiment.sh serve`. This starts Postgres
   with one engine:
   - `SERVE_ARM=stock` serves stock pgvector;
   - `SERVE_ARM=fast` serves pgvectorfast, and `FAST_SO` picks which
     pgvectorfast build;
   - `SERVE_TABLE` picks the table, and `EXTRA_PG_OPTS` adds server settings
     such as `-c huge_pages=on`.
2. **On the client:** `bench/ann_bench.py`. It runs the search-width sweep with
   N concurrent connections (`--clients`) for 60 s per width. It writes one
   JSON line per width to `bench/results/hackernews_minilm_384.jsonl`: recall@10,
   QPS, latency percentiles and the settings. `BENCH_LABEL` tags each row, and
   these labels are what the findings, tables and dashboard refer to (for
   example `pippipnn-fork` or `sat60n-pgvectorfast-tq4`).

### Indexes

- `run_experiment.sh build` builds an HNSW index with stock pgvector's
  `CREATE INDEX`. `M`, `EF_CONSTRUCTION` and `BUILD_MWM` set the parameters,
  and the log says whether the build stayed in memory ("no spill").
- `run_experiment.sh build_pipnn` builds a PiPNN graph with Ginger's builder
  (exported by `pgvectorfast_pipnn/tools/pipnn_export`). It then imports that
  graph into an HNSW index with `CREATE INDEX ... WITH (m = 32)` and
  `hnsw.build_graph_file`. With `PIPNN_QUANTIZER=tq4` the same graph is
  imported with 4-bit codes instead of floats.
- The PiPNN table (`PIPNN_TABLE`, `items_hn_pipnn`) is reused for the float
  PiPNN index, the tq4 index and HNSW m = 32. There is room on the disk for
  only one at a time.
- Indexes are never dropped by accident. A script must name the index to drop
  (`BUILD_DROP`, `PIPNN_DROP`), or ask to rebuild one in place
  (`PIPNN_REBUILD=1`).
- After the float PiPNN index is put back, `recall_gate.py` checks that it
  gives exactly the same recall at every search width as the original
  (`pippipnn-fork`).

### Huge pages

[`bench/cloudlab/hp_lib.sh`](bench/cloudlab/hp_lib.sh) has three commands:

- **Reserve:** stops Postgres, frees memory, and reserves as many 2 MB pages as
  Postgres reports it needs.
- **Check:** confirms the running server really uses them.
- **Release:** gives them back.

Index builds need the memory the reserved pages hold, so drivers release the
pages before a build and reserve them again after it.

### Getting results back

`bench/cloudlab/sync.sh pull` copies both nodes' `bench/results/` to
`bench/results/<host>/` on the workstation. Most drivers also copy their own
outputs into `bench/profiles/<study>/` at the end. Both folders are ignored by
git.

Drivers that measure the saturated server also sample both nodes' CPU and the
server's link every 2 seconds. `saturation_report.py` then shows, for every
point, how busy the server was, whether it ever went idle, and whether the
link dropped.

## Profiling

The profiling drivers use one method. A client holds a single search width for
130 s. On the server, [`bench/cloudlab/perf_window.sh`](bench/cloudlab/perf_window.sh) waits 30 s for the load
to settle, then takes four 20 s windows in turn:

1. hardware counters, with memory-controller traffic;
2. Top-Down pipeline breakdown;
3. call-graph samples on every CPU;
4. TLB and page-walk counters.

[`perf_summary.py`](bench/cloudlab/perf_summary.py) turns these into one row per
configuration: QPS, IPC, DRAM traffic, CPU per query and page-walk share.
[`perf_buckets.py`](bench/cloudlab/perf_buckets.py) splits server CPU per query into buckets
(buffer manager, graph walk, distance, everything else) and lists the hottest
functions. [`explain_buffers.py`](bench/cloudlab/explain_buffers.py) counts the pages each
query touches, for stock pgvector.

These are runs of their own, so the measured sweeps carry no profiling
overhead. Other tools look at one part in more detail:

- `tlb_window.sh` and `tlb_attrib.py`: which data structures the TLB misses
  land on.
- `trav_window.sh` and `trav_attrib.py`: inside pgvectorfast's graph walk,
  which instructions stall and where slow loads are served from.
- `ginger_window.sh` and `ginger_attrib.py`: the same for Ginger.
- `pg_counts.sh`: how often key functions run per query, from uprobes.

## The scripts

All of these are in `bench/cloudlab/`.

### Plumbing

| Script | What it does |
|---|---|
| `lib.sh` | Detached, resumable steps; SSH settings; helpers used by every driver |
| `nodes-wisc.sh` | The current pair of nodes |
| `hn.env` | Settings for the Hacker News study (dataset, sweep, memory, build) |
| `hp_lib.sh` | Reserve, check and release huge pages |
| `setup_node.sh` | Prepare a fresh node: disk, packages, clock pinning, swap off, Python environment |
| `pin_cpu.sh` | Pin the clock (`check` verifies it) |
| `sync.sh` | Copy the repository to the nodes, or results back |

### The main study, in the order it was run

| Script | What it measures | Labels |
|---|---|---|
| `hn_qps.sh` | Stock pgvector, pgvectorfast and Ginger on HNSW m = 16, at 1, 10, 20 and 40 clients. Also builds everything and prepares the dataset | `pvf-*`, `sc-*`, `rtt`, `ginger-*` |
| `hn_tq4.sh` | pgvector_ntq: HNSW with 4-bit TurboQuant navigation codes | |
| `hn_hugepages.sh` | Stock pgvector with shared_buffers on huge pages | `hpsc-*` |
| `hn_fast_hp.sh` | pgvectorfast on huge pages | |
| `hn_fastntq.sh`, `hn_fastntq_k2.sh` | The tq4 prototype, then with optimized kernels (prototype branch) | |
| `hn_calico_thp.sh`, `hn_pipe_hop.sh` | Prototype: Calico arrays on 2 MB pages; the pipelined hop | |
| `hn_pipnn.sh` | **Is it the graph?** Imports Ginger's PiPNN graph into pgvectorfast and compares it with HNSW | `pippipnn-*` |
| `hn_pipnn_ginger.sh` | Ginger searching that same PiPNN graph | |
| `hn_pipnn_stock.sh` | Stock pgvector searching the imported PiPNN graph | `pipstock-*` |
| `hn_hnsw_m32.sh` | HNSW built with m = 32, PiPNN's degree, for a like-for-like graph comparison | `pipm32-*` |
| `hn_pipnn_hp.sh` | The PiPNN comparison on huge pages | `piphp-*` |
| `hn_pipnn_tq4.sh` | pgvectorfast with tq4 codes on the PiPNN graph | `piptq4-*` |
| `hn_ginger_ablation.sh` | Ginger with one feature turned off at a time | `ginger-abl-*` |
| `hn_ginger_thp.sh` | Ginger on transparent huge pages | `ginger-thp-*` |
| `hn_saturation_probe.sh` | How many clients it takes to keep the server busy (40, 60, 80) | |
| `hn_saturated.sh`, `hn_saturated_rerun.sh` | Every Postgres arm at 60 clients on the first pair of nodes. Spoiled by that pair's network link dropping | |
| `hn_new_nodes_check.sh` | After moving to a new pair: the data arrived whole and recall matches | `move-*` |
| `hn_saturated_m32.sh` | Every Postgres arm at 60 clients on the second pair, HNSW at m = 32, plus a 40-client check that the two pairs agree | `sat60n-*` |
| `hn_saturated_hp.sh` | The same on huge pages. **Not run yet**: stopped at the first sweep when the nodes were returned | `sathp60-*` |

### Profiling drivers

| Script | What it profiles |
|---|---|
| `hn_profile_tq4.sh` | Stock pgvector, pgvectorfast and pgvector_ntq on HNSW m = 16 |
| `hn_profile_sat.sh` | Stock pgvector and pgvectorfast on HNSW m = 32, at 1 and 60 clients |
| `hn_profile_pipnn_sat.sh` | The same on the PiPNN graph |
| `hn_tlb_diag.sh` | Where pgvectorfast + tq4's remaining page walks come from |
| `hn_trav_prof.sh` | What pgvectorfast + tq4's graph walk is made of |
| `hn_ginger_prof.sh` | Why Ginger is faster: the work per query and what it costs |

### Analysis

| Script | What it produces |
|---|---|
| `hn_fourway.py` | The four-way tables and charts in the findings |
| `pipnn_compare.py` | QPS at matched recall for the graph comparison |
| `compare_qps.py` | Recall and QPS of two labels side by side, per search width |
| `compare_recall.py`, `recall_gate.py` | Fail unless two labels give the same recall at every search width |
| `saturation_report.py` | Per point: server and client CPU busy, idle stalls, link drops |
| `perf_summary.py`, `perf_buckets.py`, `perf_categories.py` | Profile summaries (see Profiling) |
| `tlb_attrib.py`, `trav_attrib.py`, `ginger_attrib.py` | Detailed attribution from the window scripts |
| `ginger_decomp.py` | Breaks Ginger's lead over pgvectorfast + tq4 into parts |
| `ginger_thp_promote.py` | Checks a Ginger run really did (or did not) use huge pages |
| `graph_layout.py` | How many index pages expanding one node touches, for a graph file |

### Microbenchmarks

| File | What it measures |
|---|---|
| `distbench.c` | Whether pgvector's L2 kernel is limited by arithmetic or by memory |
| `membw.c` | The machine's memory-bandwidth ceilings, by thread count |
| `tq_kernel_bench.c`, `tq_kernel_opt_bench.c` | The TurboQuant kernels, plain and optimized |

### Earlier studies (m510 nodes)

| Script | What it is |
|---|---|
| `dbpedia.env`, `sift.env` | Settings for the DBpedia-OpenAI (1536-d) and SIFT1M (128-d) studies |
| `sift_qps.sh` | Every system on SIFT1M |
| `halfvec.sh` | halfvec against float HNSW on both datasets |
| `profile_pgvector.sh` | Where stock pgvector spends its time on DBpedia-OpenAI |
| `scaling.sh` | 12 and 16 concurrent clients |
| `remote_client.sh` | What running the client on the server's own machine costs |

## Practical notes

- **Disk.** The server's `/mydata` was 93% full with the PiPNN table holding
  one index. Anything that needs two indexes at once needs a bigger disk.
- **Step commands are run as `( settings && command ); echo $?`.** A heredoc
  inside a step's command breaks this wrapper; put the logic in a script
  instead.
- **Inside a step or over SSH, never `pkill -f` a pattern that also appears in
  your own command line.** It kills its own shell. Write the pattern so it
  cannot match itself, for example `[a]nn_bench.py`.
- **`ssh` inside a script that is piped over SSH must use `-n`.** Otherwise it
  reads the rest of the script as its input.
- **Moving to new nodes.** Copy `/mydata` across node to node, then run
  `hn_new_nodes_check.sh`. The 4 KB 60-client pass reran pgvectorfast + PiPNN
  at 40 clients (`sat60n-anchor40`) to confirm the new pair measures like the
  old one.
