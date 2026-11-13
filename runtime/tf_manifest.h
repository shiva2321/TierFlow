#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tierflow {

enum class ObjectRole : uint8_t {
    ALWAYS_ON   = 0,
    ATTN_BLOCK  = 1,
    EXPERT      = 2,
    EMBEDDING   = 3,
    OUTPUT_HEAD = 4,
    UNKNOWN     = 5,
};

struct ManifestObject {
    uint32_t    object_id          = 0;
    std::string name;
    ObjectRole  role               = ObjectRole::UNKNOWN;
    int32_t     layer              = -1;
    int32_t     expert_idx         = -1;
    int32_t     ggml_type          = 0;
    std::string ggml_type_name;
    uint64_t    tfpack_offset      = 0;
    uint64_t    exact_bytes        = 0;
    uint64_t    page_aligned_bytes = 0;
    double      freq               = 0.0; // calibration activation probability
    std::string sha256;
};

struct TfManifest {
    std::string                 format;
    std::string                 source_gguf;
    std::string                 tfpack_path;
    uint64_t                    tfpack_size_bytes = 0;
    uint32_t                    page_size_bytes   = 4096;
    std::string                 architecture;
    int32_t                     block_count       = 0;
    int32_t                     embedding_length  = 0;
    int32_t                     ffn_length        = 0;
    int32_t                     head_count        = 0;
    int32_t                     head_count_kv     = 0;
    std::vector<ManifestObject> objects;

    uint64_t total_exact_bytes() const;
    uint64_t total_aligned_bytes() const;
};

bool load_tf_manifest(const std::string& json_path, TfManifest& out, std::string* err_out = nullptr);

} // namespace tierflow

