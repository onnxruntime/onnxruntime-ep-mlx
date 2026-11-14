#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct Options {
    int64_t rows = 0;
    int64_t inner = 0;
    int64_t outer = 0;
    int64_t experts = 0;
    int64_t selections = 0;
    int warmup = 10;
    int repeats = 30;
    bool time_io = false;
    std::string format;
    std::string ids_type;
    fs::path weights;
    fs::path activations;
    fs::path ids;
    fs::path output_json;
    fs::path output_raw;
    fs::path metal_plugin = "/opt/homebrew/opt/ggml/libexec/libggml-metal.so";
};

struct Format {
    const char * name;
    enum ggml_type type;
    int64_t block_elements;
    size_t block_bytes;
};

constexpr Format kFormats[] = {
    {"gguf:q4_0", GGML_TYPE_Q4_0, 32, 18},
    {"gguf:q4_1", GGML_TYPE_Q4_1, 32, 20},
    {"gguf:q8_0", GGML_TYPE_Q8_0, 32, 34},
    {"gguf:q4_k", GGML_TYPE_Q4_K, 256, 144},
    {"gguf:q5_k", GGML_TYPE_Q5_K, 256, 176},
    {"gguf:q6_k", GGML_TYPE_Q6_K, 256, 210},
};

struct RegistryHandle {
    ggml_backend_reg_t value = nullptr;
    ~RegistryHandle() {
        if (value != nullptr) {
            ggml_backend_unload(value);
        }
    }
};

struct BackendHandle {
    ggml_backend_t value = nullptr;
    ~BackendHandle() {
        if (value != nullptr) {
            ggml_backend_free(value);
        }
    }
};

struct ContextHandle {
    ggml_context * value = nullptr;
    ~ContextHandle() {
        if (value != nullptr) {
            ggml_free(value);
        }
    }
};

struct AllocatorHandle {
    ggml_gallocr_t value = nullptr;
    ~AllocatorHandle() {
        if (value != nullptr) {
            ggml_gallocr_free(value);
        }
    }
};

[[noreturn]] void usage(const char * argv0, const std::string & error = {}) {
    if (!error.empty()) {
        std::cerr << "error: " << error << "\n\n";
    }
    std::cerr
        << "Usage:\n  " << argv0 << " --format gguf:q4_0 --rows M --inner K --outer N"
        << " --experts E --selections S\n"
        << "    --weights weights.bin --x x.f32 --ids ids.bin --ids-type i32|i64\n"
        << "    --warmup W --repeats R --time-io 0|1\n"
        << "    --output-json result.json --output-raw result.f32\n"
        << "    [--metal-plugin /opt/homebrew/opt/ggml/libexec/libggml-metal.so]\n\n"
        << "Raw layouts are weights [E,N,K] in GGUF blocks along K, x [M,K] float32,\n"
        << "ids [M,S] int32/int64, and output [M,S,N] float32. Bias is not applied.\n";
    throw std::runtime_error(error.empty() ? "help requested" : error);
}

int64_t parse_i64(const std::string & text, const std::string & option) {
    size_t used = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &used);
    } catch (const std::exception &) {
        throw std::runtime_error(option + " requires an integer, got '" + text + "'");
    }
    if (used != text.size()) {
        throw std::runtime_error(option + " requires an integer, got '" + text + "'");
    }
    return static_cast<int64_t>(value);
}

int parse_count(const std::string & text, const std::string & option) {
    const int64_t value = parse_i64(text, option);
    if (value < 0 || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(option + " is outside the supported integer range");
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char ** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            usage(argv[0]);
        }
        if (i + 1 >= argc) {
            usage(argv[0], "missing value for " + key);
        }
        const std::string value = argv[++i];
        if (key == "--format") {
            result.format = value;
        } else if (key == "--rows") {
            result.rows = parse_i64(value, key);
        } else if (key == "--inner") {
            result.inner = parse_i64(value, key);
        } else if (key == "--outer") {
            result.outer = parse_i64(value, key);
        } else if (key == "--experts") {
            result.experts = parse_i64(value, key);
        } else if (key == "--selections") {
            result.selections = parse_i64(value, key);
        } else if (key == "--warmup") {
            result.warmup = parse_count(value, key);
        } else if (key == "--repeats") {
            result.repeats = parse_count(value, key);
        } else if (key == "--time-io") {
            if (value != "0" && value != "1") {
                usage(argv[0], "--time-io must be 0 or 1");
            }
            result.time_io = value == "1";
        } else if (key == "--ids-type") {
            result.ids_type = value;
        } else if (key == "--weights") {
            result.weights = value;
        } else if (key == "--x") {
            result.activations = value;
        } else if (key == "--ids") {
            result.ids = value;
        } else if (key == "--output-json") {
            result.output_json = value;
        } else if (key == "--output-raw") {
            result.output_raw = value;
        } else if (key == "--metal-plugin") {
            result.metal_plugin = value;
        } else {
            usage(argv[0], "unknown option " + key);
        }
    }

    if (result.rows <= 0 || result.inner <= 0 || result.outer <= 0 ||
        result.experts <= 0 || result.selections <= 0) {
        usage(argv[0], "M, K, N, E, and S must all be positive");
    }
    if (result.repeats <= 0) {
        usage(argv[0], "repeats must be positive");
    }
    if (result.warmup > std::numeric_limits<int>::max() - result.repeats) {
        usage(argv[0], "warmup + repeats is too large");
    }
    if (result.format.empty() || result.ids_type.empty() || result.weights.empty() ||
        result.activations.empty() || result.ids.empty() || result.output_json.empty() ||
        result.output_raw.empty()) {
        usage(argv[0], "format, input paths, ids type, and output paths are required");
    }
    if (result.ids_type != "i32" && result.ids_type != "i64") {
        usage(argv[0], "--ids-type must be i32 or i64");
    }
    return result;
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

const Format & find_format(std::string name) {
    name = lowercase(std::move(name));
    if (name.rfind("gguf:", 0) != 0) {
        name = "gguf:" + name;
    }
    for (const Format & format : kFormats) {
        if (name == format.name) {
            return format;
        }
    }
    throw std::runtime_error(
        "unsupported format '" + name +
        "'; expected q4_0, q4_1, q8_0, q4_k, q5_k, or q6_k");
}

size_t checked_mul(size_t lhs, size_t rhs, const std::string & what) {
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
        throw std::runtime_error(what + " size overflows size_t");
    }
    return lhs * rhs;
}

size_t checked_extent(int64_t value, const std::string & what) {
    if (value <= 0 || static_cast<uint64_t>(value) > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(what + " is outside the supported host size range");
    }
    return static_cast<size_t>(value);
}

std::vector<uint8_t> read_file_exact(
    const fs::path & path, size_t expected, const std::string & label) {
    std::error_code ec;
    const uintmax_t actual = fs::file_size(path, ec);
    if (ec) {
        throw std::runtime_error(
            "cannot stat " + label + " file '" + path.string() + "': " + ec.message());
    }
    if (actual != expected) {
        std::ostringstream message;
        message << label << " file '" << path.string() << "' is " << actual
                << " bytes; expected exactly " << expected;
        throw std::runtime_error(message.str());
    }
    if (expected > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error(label + " file is too large for one host read");
    }
    std::vector<uint8_t> data(expected);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + label + " file '" + path.string() + "'");
    }
    if (expected != 0) {
        input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(expected));
    }
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("short or inconsistent read from " + label + " file '" +
                                 path.string() + "'");
    }
    return data;
}

void ensure_parent(const fs::path & path) {
    const fs::path parent = path.parent_path();
    if (parent.empty()) {
        return;
    }
    std::error_code ec;
    fs::create_directories(parent, ec);
    if (ec) {
        throw std::runtime_error(
            "cannot create output directory '" + parent.string() + "': " + ec.message());
    }
}

std::string json_string(const std::string & value) {
    std::ostringstream result;
    result << '"';
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': result << "\\\""; break;
            case '\\': result << "\\\\"; break;
            case '\b': result << "\\b"; break;
            case '\f': result << "\\f"; break;
            case '\n': result << "\\n"; break;
            case '\r': result << "\\r"; break;
            case '\t': result << "\\t"; break;
            default:
                if (ch < 0x20) {
                    result << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<int>(ch) << std::dec;
                } else {
                    result << static_cast<char>(ch);
                }
        }
    }
    result << '"';
    return result.str();
}

const char * device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU: return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accelerator";
        case GGML_BACKEND_DEVICE_TYPE_META: return "meta";
    }
    return "unknown";
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    if (values.size() == 1) {
        return values[0];
    }
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    const double part = position - static_cast<double>(lower);
    return values[lower] + (values[upper] - values[lower]) * part;
}

void require_little_endian() {
    const uint16_t value = 1;
    if (*reinterpret_cast<const uint8_t *>(&value) != 1) {
        throw std::runtime_error("raw fixture format requires a little-endian host");
    }
}

int run(const Options & options) {
    require_little_endian();
    const Format & format = find_format(options.format);

    for (const auto & dimension : std::vector<std::pair<const char *, int64_t>>{
             {"M", options.rows},
             {"K", options.inner},
             {"N", options.outer},
             {"E", options.experts},
             {"S", options.selections},
         }) {
        if (dimension.second > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error(
                std::string(dimension.first) +
                " exceeds the int32 dimension range used by the installed Metal kernels");
        }
    }
    if (options.inner % format.block_elements != 0) {
        std::ostringstream message;
        message << format.name << " requires K divisible by " << format.block_elements
                << ", got K=" << options.inner;
        throw std::runtime_error(message.str());
    }
    if (ggml_blck_size(format.type) != format.block_elements ||
        ggml_type_size(format.type) != format.block_bytes) {
        throw std::runtime_error(
            "installed ggml ABI block geometry does not match the requested GGUF format");
    }

    const size_t m = checked_extent(options.rows, "M");
    const size_t k = checked_extent(options.inner, "K");
    const size_t n = checked_extent(options.outer, "N");
    const size_t e = checked_extent(options.experts, "E");
    const size_t s = checked_extent(options.selections, "S");
    const size_t weight_rows = checked_mul(e, n, "weight row count");
    const size_t weight_bytes = checked_mul(
        weight_rows, ggml_row_size(format.type, options.inner), "weight");
    const size_t x_elements = checked_mul(m, k, "activation element count");
    const size_t x_bytes = checked_mul(x_elements, sizeof(float), "activation");
    const size_t id_elements = checked_mul(m, s, "ID element count");
    const size_t id_scalar_bytes = options.ids_type == "i32" ? sizeof(int32_t) : sizeof(int64_t);
    const size_t ids_bytes = checked_mul(id_elements, id_scalar_bytes, "ID");
    const size_t output_elements = checked_mul(checked_mul(m, s, "output rows"), n, "output");
    const size_t output_bytes = checked_mul(output_elements, sizeof(float), "output");

    std::vector<uint8_t> weight_data =
        read_file_exact(options.weights, weight_bytes, "packed weight");
    std::vector<uint8_t> x_data =
        read_file_exact(options.activations, x_bytes, "activation");
    std::vector<uint8_t> ids_input = read_file_exact(options.ids, ids_bytes, "ID");
    if (!ggml_validate_row_data(format.type, weight_data.data(), weight_data.size())) {
        throw std::runtime_error(
            "packed weight bytes fail ggml_validate_row_data for " +
            std::string(format.name) + "; check block layout and finite scale fields");
    }

    std::vector<int32_t> ids_i32(id_elements);
    std::vector<bool> used_experts(e, false);
    bool duplicate_per_token = false;
    for (size_t row = 0; row < m; ++row) {
        std::set<int32_t> row_ids;
        for (size_t selection = 0; selection < s; ++selection) {
            const size_t index = row * s + selection;
            int64_t value = 0;
            if (options.ids_type == "i32") {
                int32_t narrow = 0;
                std::memcpy(&narrow, ids_input.data() + index * sizeof(narrow), sizeof(narrow));
                value = narrow;
            } else {
                std::memcpy(&value, ids_input.data() + index * sizeof(value), sizeof(value));
            }
            if (value < 0 || value >= options.experts) {
                std::ostringstream message;
                message << "ids[" << row << "," << selection << "]=" << value
                        << " is outside [0," << options.experts << ")";
                throw std::runtime_error(message.str());
            }
            ids_i32[index] = static_cast<int32_t>(value);
            used_experts[static_cast<size_t>(value)] = true;
            duplicate_per_token = !row_ids.insert(static_cast<int32_t>(value)).second ||
                                  duplicate_per_token;
        }
    }
    const size_t active_experts =
        static_cast<size_t>(std::count(used_experts.begin(), used_experts.end(), true));
    if (duplicate_per_token) {
        throw std::runtime_error(
            "GGML MUL_MAT_ID leaves duplicate expert IDs per token undefined; "
            "export unique routes for this comparison (MLX GroupedMatMul supports repeats)");
    }

    std::error_code ec;
    const fs::path plugin = fs::canonical(options.metal_plugin, ec);
    if (ec) {
        throw std::runtime_error(
            "cannot resolve Metal plugin '" + options.metal_plugin.string() + "': " +
            ec.message());
    }

    RegistryHandle registry;
    registry.value = ggml_backend_load(plugin.c_str());
    if (registry.value == nullptr) {
        throw std::runtime_error(
            "ggml_backend_load failed for Metal plugin '" + plugin.string() + "'");
    }
    const std::string registry_name = ggml_backend_reg_name(registry.value);
    const std::string registry_name_lower = lowercase(registry_name);
    if (registry_name_lower != "mtl" &&
        registry_name_lower.find("metal") == std::string::npos) {
        throw std::runtime_error(
            "loaded backend registry is '" + registry_name + "', not Metal");
    }
    const size_t device_count = ggml_backend_reg_dev_count(registry.value);
    if (device_count == 0) {
        throw std::runtime_error("Metal backend registry exposes no devices");
    }
    ggml_backend_dev_t device = ggml_backend_reg_dev_get(registry.value, 0);
    if (device == nullptr) {
        throw std::runtime_error("Metal backend returned a null device");
    }
    const enum ggml_backend_dev_type device_type = ggml_backend_dev_type(device);
    if (device_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
        device_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        throw std::runtime_error(
            "Metal device has unexpected type '" + std::string(device_type_name(device_type)) +
            "'");
    }

    BackendHandle backend;
    backend.value = ggml_backend_dev_init(device, nullptr);
    if (backend.value == nullptr) {
        throw std::runtime_error("ggml_backend_dev_init failed for Metal device");
    }

    ContextHandle context;
    context.value = ggml_init({
        /* .mem_size = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc = */ true,
    });
    if (context.value == nullptr) {
        throw std::runtime_error("ggml_init failed for graph metadata");
    }

    ggml_tensor * weights = ggml_new_tensor_3d(
        context.value, format.type, options.inner, options.outer, options.experts);
    ggml_tensor * activations =
        ggml_new_tensor_3d(context.value, GGML_TYPE_F32, options.inner, 1, options.rows);
    ggml_tensor * ids =
        ggml_new_tensor_2d(context.value, GGML_TYPE_I32, options.selections, options.rows);
    if (weights == nullptr || activations == nullptr || ids == nullptr) {
        throw std::runtime_error("failed to create ggml input tensors");
    }
    ggml_set_name(weights, "weights_[K,N,E]");
    ggml_set_name(activations, "x_[K,1,M]");
    ggml_set_name(ids, "ids_[S,M]");
    ggml_set_input(weights);
    ggml_set_input(activations);
    ggml_set_input(ids);

    ggml_tensor * output = ggml_mul_mat_id(context.value, weights, activations, ids);
    if (output == nullptr) {
        throw std::runtime_error("ggml_mul_mat_id failed to construct the routed product");
    }
    ggml_set_name(output, "output_[N,S,M]");
    ggml_set_output(output);

    if (ggml_nbytes(weights) != weight_bytes || ggml_nbytes(activations) != x_bytes ||
        ggml_nbytes(ids) != id_elements * sizeof(int32_t) ||
        ggml_nbytes(output) != output_bytes) {
        throw std::runtime_error(
            "ggml tensor byte sizes do not match the documented raw layouts");
    }
    const bool device_supports = ggml_backend_dev_supports_op(device, output);
    const bool backend_supports = ggml_backend_supports_op(backend.value, output);
    if (!device_supports || !backend_supports) {
        throw std::runtime_error(
            "installed Metal backend reports supports_op=false for GGML_OP_MUL_MAT_ID "
            "with the requested types and dimensions; no CPU fallback is attempted");
    }

    ggml_cgraph * graph = ggml_new_graph_custom(context.value, 8, false);
    if (graph == nullptr) {
        throw std::runtime_error("failed to create ggml compute graph");
    }
    ggml_build_forward_expand(graph, output);
    if (ggml_graph_n_nodes(graph) != 1 ||
        ggml_graph_node(graph, 0)->op != GGML_OP_MUL_MAT_ID) {
        throw std::runtime_error(
            "unexpected ggml graph: expected one direct GGML_OP_MUL_MAT_ID node");
    }

    AllocatorHandle allocator;
    allocator.value = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.value));
    if (allocator.value == nullptr || !ggml_gallocr_alloc_graph(allocator.value, graph)) {
        throw std::runtime_error("Metal graph allocation failed");
    }

    ggml_backend_tensor_set(weights, weight_data.data(), 0, weight_data.size());
    if (!options.time_io) {
        ggml_backend_tensor_set(activations, x_data.data(), 0, x_data.size());
        ggml_backend_tensor_set(ids, ids_i32.data(), 0, ids_i32.size() * sizeof(int32_t));
    }
    ggml_backend_synchronize(backend.value);

    std::vector<float> output_data(output_elements);
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(options.repeats));
    const int iterations = options.warmup + options.repeats;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        ggml_backend_synchronize(backend.value);
        const auto start = std::chrono::steady_clock::now();
        if (options.time_io) {
            ggml_backend_tensor_set(activations, x_data.data(), 0, x_data.size());
            ggml_backend_tensor_set(ids, ids_i32.data(), 0, ids_i32.size() * sizeof(int32_t));
        }
        const enum ggml_status status = ggml_backend_graph_compute(backend.value, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(
                "Metal graph compute failed: " + std::string(ggml_status_to_string(status)));
        }
        ggml_backend_synchronize(backend.value);
        if (options.time_io) {
            ggml_backend_tensor_get(output, output_data.data(), 0, output_bytes);
            ggml_backend_synchronize(backend.value);
        }
        const auto stop = std::chrono::steady_clock::now();
        if (iteration >= options.warmup) {
            samples.push_back(
                std::chrono::duration<double, std::milli>(stop - start).count());
        }
    }
    if (!options.time_io) {
        ggml_backend_tensor_get(output, output_data.data(), 0, output_bytes);
        ggml_backend_synchronize(backend.value);
    }

    ensure_parent(options.output_raw);
    std::ofstream raw_output(options.output_raw, std::ios::binary);
    if (!raw_output) {
        throw std::runtime_error(
            "cannot open raw output '" + options.output_raw.string() + "'");
    }
    raw_output.write(
        reinterpret_cast<const char *>(output_data.data()),
        static_cast<std::streamsize>(output_bytes));
    if (!raw_output) {
        throw std::runtime_error(
            "failed writing raw output '" + options.output_raw.string() + "'");
    }

    const std::string device_name = ggml_backend_dev_name(device);
    const std::string device_description = ggml_backend_dev_description(device);
    const std::string backend_name = ggml_backend_name(backend.value);
    const double median_ms = percentile(samples, 0.5);
    const double p95_ms = percentile(samples, 0.95);

    ensure_parent(options.output_json);
    std::ofstream json(options.output_json);
    if (!json) {
        throw std::runtime_error(
            "cannot open JSON output '" + options.output_json.string() + "'");
    }
    json << std::setprecision(17);
    json << "{\n";
    json << "  \"format\": " << json_string(format.name) << ",\n";
    json << "  \"op\": \"GGML_OP_MUL_MAT_ID\",\n";
    json << "  \"transB\": 1,\n";
    json << "  \"bias\": false,\n";
    json << "  \"shape\": {\"M\": " << options.rows << ", \"K\": " << options.inner
         << ", \"N\": " << options.outer << ", \"E\": " << options.experts
         << ", \"S\": " << options.selections << "},\n";
    json << "  \"layout\": {\"weights\": \"[E,N,K] GGUF blocks along K / ggml ne=[K,N,E]\", "
         << "\"x\": \"[M,K] float32 / ggml ne=[K,1,M]\", "
         << "\"ids\": \"[M,S] " << options.ids_type
         << " converted losslessly to ggml int32 ne=[S,M]\", "
         << "\"output\": \"[M,S,N] float32 / ggml ne=[N,S,M]\"},\n";
    json << "  \"routing\": {\"active_experts\": " << active_experts
         << ", \"unused_experts\": " << (e - active_experts)
         << ", \"duplicate_expert_per_token\": "
         << (duplicate_per_token ? "true" : "false") << "},\n";
    json << "  \"warmup\": " << options.warmup << ",\n";
    json << "  \"repeats\": " << options.repeats << ",\n";
    json << "  \"samples_ms\": [";
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i != 0) {
            json << ", ";
        }
        json << samples[i];
    }
    json << "],\n";
    json << "  \"median_ms\": " << median_ms << ",\n";
    json << "  \"p95_ms\": " << p95_ms << ",\n";
    json << "  \"timing\": {\"setup\": false, \"weight_upload\": false, "
         << "\"activation_and_id_upload\": " << (options.time_io ? "true" : "false")
         << ", \"output_download\": " << (options.time_io ? "true" : "false")
         << ", \"explicit_backend_sync\": true},\n";
    json << "  \"backend_proof\": {\"plugin\": " << json_string(plugin.string())
         << ", \"registry\": " << json_string(registry_name)
         << ", \"backend\": " << json_string(backend_name)
         << ", \"device\": " << json_string(device_name)
         << ", \"description\": " << json_string(device_description)
         << ", \"device_type\": " << json_string(device_type_name(device_type))
         << ", \"device_supports_op\": " << (device_supports ? "true" : "false")
         << ", \"backend_supports_op\": " << (backend_supports ? "true" : "false")
         << ", \"direct_backend_compute\": true, \"scheduler_used\": false, "
         << "\"cpu_backend_loaded\": false},\n";
    json << "  \"ggml\": {\"version\": " << json_string(ggml_version())
         << ", \"commit\": " << json_string(ggml_commit()) << "},\n";
    json << "  \"bytes\": {\"weights\": " << weight_bytes << ", \"x\": " << x_bytes
         << ", \"ids_input\": " << ids_bytes << ", \"output\": " << output_bytes
         << ", \"graph_compute_buffer\": "
         << ggml_gallocr_get_buffer_size(allocator.value, 0) << "},\n";
    json << "  \"numeric_scope\": "
         << json_string(
                "Output is float32. In ggml v0.29.0 Metal, the M<32 routed vector path "
                "consumes float32 activations and accumulates float32; the M>=32 SIMD-group "
                "matrix path stages decoded weights and activations through FP16 tiles before "
                "float32 accumulation. This is not an exact ONNX Float32-decode baseline.")
         << ",\n";
    json << "  \"route_condition\": "
         << json_string(
                "v0.29.0 selects mul_mm_id only when simdgroup_mm && K>=64 && E<=1024 && "
                "M>=32; otherwise it selects mul_mv_id.")
         << ",\n";
    json << "  \"duplicate_id_contract\": "
         << json_string(
                "ggml v0.29.0 declares duplicate experts within one token undefined across "
                "backends; this helper rejects duplicate IDs within each token.")
         << "\n";
    json << "}\n";
    if (!json) {
        throw std::runtime_error(
            "failed writing JSON output '" + options.output_json.string() + "'");
    }

    std::cout << "{\"output\":" << json_string(options.output_json.string())
              << ",\"median_ms\":" << std::setprecision(10) << median_ms
              << ",\"backend\":" << json_string(backend_name)
              << ",\"supports_op\":true,\"scheduler_used\":false}\n";
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = parse_options(argc, argv);
        return run(options);
    } catch (const std::exception & error) {
        if (std::string(error.what()) != "help requested") {
            std::cerr << "ggml_static_weight_decode: " << error.what()
                      << "\nCheck input dimensions/files and the Metal-enabled GGML build; "
                      << "use --help for the required layouts and inspect backend logs for compute failures.\n";
        }
        return std::string(error.what()) == "help requested" ? 0 : 1;
    }
}
