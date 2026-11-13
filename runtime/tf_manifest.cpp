#include "tf_manifest.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace tierflow {

uint64_t TfManifest::total_exact_bytes() const {
    uint64_t s = 0;
    for (const auto& o : objects) s += o.exact_bytes;
    return s;
}

uint64_t TfManifest::total_aligned_bytes() const {
    uint64_t s = 0;
    for (const auto& o : objects) s += o.page_aligned_bytes;
    return s;
}

namespace {

ObjectRole parse_role(const std::string& r) {
    if (r == "always_on") return ObjectRole::ALWAYS_ON;
    if (r == "attn_block") return ObjectRole::ATTN_BLOCK;
    if (r == "expert" || r == "moe_expert") return ObjectRole::EXPERT;
    if (r == "embedding") return ObjectRole::EMBEDDING;
    if (r == "output_head") return ObjectRole::OUTPUT_HEAD;
    return ObjectRole::UNKNOWN;
}

// Minimal recursive-descent JSON reader sufficient for manifest.json
struct JsonCursor {
    const std::string& s;
    size_t pos = 0;

    void skip_ws() {
        while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    }

    bool consume(char c) {
        skip_ws();
        if (pos < s.size() && s[pos] == c) {
            ++pos;
            return true;
        }
        return false;
    }

    bool parse_string(std::string& out) {
        skip_ws();
        if (pos >= s.size() || s[pos] != '"') return false;
        ++pos;
        out.clear();
        while (pos < s.size()) {
            char c = s[pos++];
            if (c == '"') return true;
            if (c == '\\' && pos < s.size()) {
                char e = s[pos++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default: out.push_back(e); break;
                }
            } else {
                out.push_back(c);
            }
        }
        return false;
    }

    bool parse_number_token(std::string& out) {
        skip_ws();
        size_t start = pos;
        while (pos < s.size()) {
            char c = s[pos];
            if (std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
                ++pos;
            } else {
                break;
            }
        }
        if (pos == start) return false;
        out.assign(s.data() + start, pos - start);
        return true;
    }

    bool skip_value() {
        skip_ws();
        if (pos >= s.size()) return false;
        char c = s[pos];
        if (c == '"') {
            std::string dummy;
            return parse_string(dummy);
        }
        if (c == '{') {
            ++pos;
            if (consume('}')) return true;
            while (pos < s.size()) {
                std::string k;
                if (!parse_string(k) || !consume(':') || !skip_value()) return false;
                if (consume('}')) return true;
                if (!consume(',')) return false;
            }
            return false;
        }
        if (c == '[') {
            ++pos;
            if (consume(']')) return true;
            while (pos < s.size()) {
                if (!skip_value()) return false;
                if (consume(']')) return true;
                if (!consume(',')) return false;
            }
            return false;
        }
        if (s.compare(pos, 4, "true") == 0) { pos += 4; return true; }
        if (s.compare(pos, 5, "false") == 0) { pos += 5; return true; }
        if (s.compare(pos, 4, "null") == 0) { pos += 4; return true; }
        std::string num;
        return parse_number_token(num);
    }

    bool parse_int64(int64_t& val) {
        std::string num;
        if (!parse_number_token(num)) return false;
        val = std::strtoll(num.c_str(), nullptr, 10);
        return true;
    }

    bool parse_uint64(uint64_t& val) {
        std::string num;
        if (!parse_number_token(num)) return false;
        val = std::strtoull(num.c_str(), nullptr, 10);
        return true;
    }

    bool parse_double(double& val) {
        std::string num;
        if (!parse_number_token(num)) return false;
        val = std::strtod(num.c_str(), nullptr);
        return true;
    }
};

bool parse_gguf_metadata(JsonCursor& cur, TfManifest& out) {
    if (!cur.consume('{')) return false;
    if (cur.consume('}')) return true;
    while (cur.pos < cur.s.size()) {
        std::string k;
        if (!cur.parse_string(k) || !cur.consume(':')) return false;
        if (k == "general.architecture") {
            if (!cur.parse_string(out.architecture)) return false;
        } else if (k.size() >= 12 && k.compare(k.size() - 12, 12, ".block_count") == 0) {
            int64_t v = 0; if (!cur.parse_int64(v)) return false; out.block_count = static_cast<int32_t>(v);
        } else if (k.size() >= 17 && k.compare(k.size() - 17, 17, ".embedding_length") == 0) {
            int64_t v = 0; if (!cur.parse_int64(v)) return false; out.embedding_length = static_cast<int32_t>(v);
        } else if (k.size() >= 20 && k.compare(k.size() - 20, 20, ".feed_forward_length") == 0) {
            int64_t v = 0; if (!cur.parse_int64(v)) return false; out.ffn_length = static_cast<int32_t>(v);
        } else if (k.size() >= 21 && k.compare(k.size() - 21, 21, ".attention.head_count") == 0) {
            int64_t v = 0; if (!cur.parse_int64(v)) return false; out.head_count = static_cast<int32_t>(v);
        } else if (k.size() >= 24 && k.compare(k.size() - 24, 24, ".attention.head_count_kv") == 0) {
            int64_t v = 0; if (!cur.parse_int64(v)) return false; out.head_count_kv = static_cast<int32_t>(v);
        } else {
            if (!cur.skip_value()) return false;
        }
        if (cur.consume('}')) return true;
        if (!cur.consume(',')) return false;
    }
    return false;
}

bool parse_manifest_object(JsonCursor& cur, ManifestObject& obj) {
    if (!cur.consume('{')) return false;
    if (cur.consume('}')) return true;
    while (cur.pos < cur.s.size()) {
        std::string k;
        if (!cur.parse_string(k) || !cur.consume(':')) return false;
        if (k == "object_id") {
            uint64_t v = 0; if (!cur.parse_uint64(v)) return false; obj.object_id = static_cast<uint32_t>(v);
        } else if (k == "name") {
            if (!cur.parse_string(obj.name)) return false;
        } else if (k == "role") {
            std::string r; if (!cur.parse_string(r)) return false; obj.role = parse_role(r);
        } else if (k == "layer") {
            int64_t v = -1; if (!cur.parse_int64(v)) return false; obj.layer = static_cast<int32_t>(v);
        } else if (k == "expert" || k == "expert_idx") {
            int64_t v = -1; if (!cur.parse_int64(v)) return false; obj.expert_idx = static_cast<int32_t>(v);
        } else if (k == "ggml_type") {
            int64_t v = 0; if (!cur.parse_int64(v)) return false; obj.ggml_type = static_cast<int32_t>(v);
        } else if (k == "ggml_type_name") {
            if (!cur.parse_string(obj.ggml_type_name)) return false;
        } else if (k == "tfpack_offset") {
            if (!cur.parse_uint64(obj.tfpack_offset)) return false;
        } else if (k == "exact_bytes") {
            if (!cur.parse_uint64(obj.exact_bytes)) return false;
        } else if (k == "page_aligned_bytes") {
            if (!cur.parse_uint64(obj.page_aligned_bytes)) return false;
        } else if (k == "freq") {
            if (!cur.parse_double(obj.freq)) return false;
        } else if (k == "sha256") {
            if (!cur.parse_string(obj.sha256)) return false;
        } else {
            if (!cur.skip_value()) return false;
        }
        if (cur.consume('}')) return true;
        if (!cur.consume(',')) return false;
    }
    return false;
}

} // namespace

bool load_tf_manifest(const std::string& json_path, TfManifest& out, std::string* err_out) {
    std::ifstream f(json_path, std::ios::binary);
    if (!f) {
        if (err_out) *err_out = "Cannot open manifest file: " + json_path;
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();

    out = TfManifest{};
    JsonCursor cur{text, 0};
    if (!cur.consume('{')) {
        if (err_out) *err_out = "Manifest root must be JSON object";
        return false;
    }
    if (cur.consume('}')) return true;

    while (cur.pos < cur.s.size()) {
        std::string k;
        if (!cur.parse_string(k) || !cur.consume(':')) {
            if (err_out) *err_out = "Malformed JSON key near offset " + std::to_string(cur.pos);
            return false;
        }
        if (k == "format") {
            if (!cur.parse_string(out.format)) return false;
        } else if (k == "source_gguf") {
            if (!cur.parse_string(out.source_gguf)) return false;
        } else if (k == "tfpack_path") {
            if (!cur.parse_string(out.tfpack_path)) return false;
        } else if (k == "tfpack_size_bytes") {
            if (!cur.parse_uint64(out.tfpack_size_bytes)) return false;
        } else if (k == "page_size_bytes") {
            uint64_t v = 4096; if (!cur.parse_uint64(v)) return false; out.page_size_bytes = static_cast<uint32_t>(v);
        } else if (k == "gguf_metadata") {
            if (!parse_gguf_metadata(cur, out)) return false;
        } else if (k == "objects") {
            if (!cur.consume('[')) return false;
            if (!cur.consume(']')) {
                while (cur.pos < cur.s.size()) {
                    ManifestObject obj{};
                    if (!parse_manifest_object(cur, obj)) {
                        if (err_out) *err_out = "Malformed object entry";
                        return false;
                    }
                    out.objects.push_back(std::move(obj));
                    if (cur.consume(']')) break;
                    if (!cur.consume(',')) {
                        if (err_out) *err_out = "Missing comma in objects array";
                        return false;
                    }
                }
            }
        } else {
            if (!cur.skip_value()) {
                if (err_out) *err_out = "Failed to skip JSON key " + k;
                return false;
            }
        }
        if (cur.consume('}')) break;
        if (!cur.consume(',')) {
            if (err_out) *err_out = "Missing comma near offset " + std::to_string(cur.pos);
            return false;
        }
    }
    return true;
}

} // namespace tierflow

