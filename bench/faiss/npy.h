// Minimal readers for the .npy / .npz files bench/datasets.py writes, so the
// C++ driver consumes exactly the same dataset the Postgres benchmarks did
// with no conversion step.
//
// Only what we need: C-order arrays, and .npz members that are STORED (which
// np.savez always is -- np.savez_compressed is what deflates). Anything else
// throws rather than guessing, because a silently misread dtype produces
// plausible wrong numbers rather than an error.
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace npy {

struct Array {
    std::string dtype;            // e.g. "<f4", "<i8"
    std::vector<size_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;

    Array() = default;
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    Array(Array&& o) noexcept { *this = std::move(o); }
    Array& operator=(Array&& o) noexcept {
        if (this != &o) {
            release();
            dtype = std::move(o.dtype);
            shape = std::move(o.shape);
            data = o.data;
            nbytes = o.nbytes;
            owned_ = std::move(o.owned_);
            map_ = o.map_;
            map_len_ = o.map_len_;
            o.data = nullptr;
            o.map_ = nullptr;
            o.map_len_ = 0;
        }
        return *this;
    }
    ~Array() { release(); }

    size_t rows() const { return shape.empty() ? 0 : shape[0]; }
    size_t cols() const { return shape.size() > 1 ? shape[1] : 1; }

    // Fail loudly on an unexpected dtype: a float64 array read as float32 is
    // not a crash, it is wrong answers.
    void expect(const std::string& want, const std::string& what) const {
        if (dtype != want) {
            throw std::runtime_error(what + ": expected dtype " + want +
                                     ", file says " + dtype);
        }
    }

    std::vector<uint8_t> owned_;  // backing store for .npz members
    void* map_ = nullptr;
    size_t map_len_ = 0;

  private:
    void release() {
        if (map_) {
            ::munmap(map_, map_len_);
            map_ = nullptr;
        }
    }
};

// Pull one quoted or bare value out of the .npy header dict.
inline std::string header_field(const std::string& h, const std::string& key) {
    size_t k = h.find("'" + key + "'");
    if (k == std::string::npos) {
        throw std::runtime_error(".npy header has no '" + key + "'");
    }
    size_t colon = h.find(':', k);
    size_t start = h.find_first_not_of(" \t", colon + 1);
    size_t end = h.find_first_of(",}", start);
    std::string v = h.substr(start, end - start);
    while (!v.empty() && (v.back() == ' ' || v.back() == ')')) v.pop_back();
    if (v.size() >= 2 && (v.front() == '\'' || v.front() == '"')) {
        v = v.substr(1, v.size() - 2);
    }
    return v;
}

// Parse the header that prefixes every .npy payload; returns the data offset.
inline size_t parse_header(const uint8_t* p, size_t len, Array& out) {
    if (len < 10 || std::memcmp(p, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("not a .npy file (bad magic)");
    }
    const int major = p[6];
    size_t hlen, off;
    if (major == 1) {
        hlen = p[8] | (size_t(p[9]) << 8);
        off = 10;
    } else {  // v2/v3 use a 4-byte header length
        hlen = p[8] | (size_t(p[9]) << 8) | (size_t(p[10]) << 16) |
               (size_t(p[11]) << 24);
        off = 12;
    }
    if (off + hlen > len) throw std::runtime_error(".npy header runs past EOF");

    const std::string header(reinterpret_cast<const char*>(p + off), hlen);
    out.dtype = header_field(header, "descr");
    if (header_field(header, "fortran_order") != "False") {
        throw std::runtime_error(".npy is Fortran-ordered; only C order is read");
    }

    // shape is "(a, b)" or "(a,)"
    size_t s = header.find("'shape'");
    size_t open = header.find('(', s);
    size_t close = header.find(')', open);
    std::string dims = header.substr(open + 1, close - open - 1);
    out.shape.clear();
    for (size_t i = 0; i < dims.size();) {
        while (i < dims.size() && !isdigit(dims[i])) i++;
        if (i >= dims.size()) break;
        size_t j = i;
        while (j < dims.size() && isdigit(dims[j])) j++;
        out.shape.push_back(std::stoull(dims.substr(i, j - i)));
        i = j;
    }
    return off + hlen;
}

// mmap a .npy file. Used for the 9.2 GB train array, which must never be read
// into the heap.
inline Array mmap_npy(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("cannot stat " + path);
    }
    void* m = ::mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("cannot mmap " + path);

    Array a;
    a.map_ = m;
    a.map_len_ = st.st_size;
    const uint8_t* p = static_cast<const uint8_t*>(m);
    size_t off = parse_header(p, st.st_size, a);
    a.data = p + off;
    a.nbytes = st.st_size - off;
    // Random access across 9.2 GB; tell the kernel not to read ahead.
    ::madvise(m, st.st_size, MADV_RANDOM);
    return a;
}

inline uint32_t le32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
           (uint32_t(p[3]) << 24);
}
inline uint16_t le16(const uint8_t* p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

// Read one member out of an uncompressed .npz (a zip of .npy files). np.savez
// stores members verbatim, so the bytes are the .npy file itself; only
// np.savez_compressed would need zlib.
inline Array load_npz(const std::string& path, const std::string& member) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::vector<uint8_t> buf(size);
    std::fseek(f, 0, SEEK_SET);
    if (std::fread(buf.data(), 1, size, f) != size_t(size)) {
        std::fclose(f);
        throw std::runtime_error("short read on " + path);
    }
    std::fclose(f);

    // End-of-central-directory record, scanning back over any trailing comment.
    long eocd = -1;
    for (long i = size - 22; i >= 0 && i > size - 22 - 65536; i--) {
        if (std::memcmp(&buf[i], "PK\x05\x06", 4) == 0) {
            eocd = i;
            break;
        }
    }
    if (eocd < 0) throw std::runtime_error(path + " is not a zip archive");

    const uint16_t entries = le16(&buf[eocd + 10]);
    uint32_t cd = le32(&buf[eocd + 16]);
    const std::string want = member.size() > 4 && member.substr(member.size() - 4) == ".npy"
                                 ? member
                                 : member + ".npy";

    for (uint16_t e = 0; e < entries; e++) {
        if (std::memcmp(&buf[cd], "PK\x01\x02", 4) != 0) {
            throw std::runtime_error("corrupt central directory in " + path);
        }
        const uint16_t method = le16(&buf[cd + 10]);
        const uint16_t nlen = le16(&buf[cd + 28]);
        const uint16_t elen = le16(&buf[cd + 30]);
        const uint16_t clen = le16(&buf[cd + 32]);
        const uint32_t lho = le32(&buf[cd + 42]);
        const std::string name(reinterpret_cast<char*>(&buf[cd + 46]), nlen);

        if (name == want) {
            if (method != 0) {
                throw std::runtime_error(path + ":" + name +
                                         " is deflated; only stored members are "
                                         "read (np.savez, not savez_compressed)");
            }
            // Local header repeats the name/extra lengths, and they can differ
            // from the central directory's, so re-read them here.
            if (std::memcmp(&buf[lho], "PK\x03\x04", 4) != 0) {
                throw std::runtime_error("corrupt local header in " + path);
            }
            const uint32_t data_off =
                    lho + 30 + le16(&buf[lho + 26]) + le16(&buf[lho + 28]);
            Array a;
            size_t off = parse_header(&buf[data_off], size - data_off, a);
            a.owned_.assign(buf.begin() + data_off + off, buf.end());
            a.data = a.owned_.data();
            a.nbytes = a.owned_.size();
            return a;
        }
        cd += 46 + nlen + elen + clen;
    }
    throw std::runtime_error(path + " has no member " + want);
}

}  // namespace npy
