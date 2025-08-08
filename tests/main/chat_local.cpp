#include "chat_local.h"

#include <sstream>

#include "ggml.h"

struct common_chat_local_templates {
    std::string template_default_source;
};

void common_chat_local_templates_free(struct common_chat_local_templates * tmpls) {
    delete tmpls;
}

common_chat_local_templates_ptr common_chat_local_templates_init(const struct llama_model * model) {
    std::string default_template_src;
    GGML_ASSERT(model != nullptr);
    const auto * str = llama_model_chat_template(model, /* name */ nullptr);
    GGML_ASSERT(str);
    default_template_src = str;
    common_chat_local_templates_ptr tmpls(new common_chat_local_templates());
    tmpls->template_default_source = default_template_src;
    return tmpls;
}

// Legacy template route (adhoc C++ implementation of known templates), forward to llama_chat_apply_template.
static common_chat_local_params common_chat_local_templates_apply_legacy(
    const struct common_chat_local_templates * tmpls, const struct common_chat_local_templates_inputs & inputs) {
    int                             alloc_size = 0;
    std::vector<llama_chat_message> chat;
    std::vector<std::string>        contents;
    for (const auto & msg : inputs.messages) {
        auto content = msg.content;
        for (const auto & part : msg.content_parts) {
            if (part.type != "text") {
                fprintf(stderr, "%s: Ignoring non-text content part: %s\n", __func__, part.type.c_str());
                continue;
            }
            if (!content.empty()) {
                content += "\n";
                ;
            }
            content += part.text;
        }
        contents.emplace_back(std::move(content));
    }
    for (size_t i = 0; i < contents.size(); ++i) {
        const auto & msg     = inputs.messages[i];
        const auto & content = contents[i];
        chat.push_back({ msg.role.c_str(), content.c_str() });
        alloc_size += (msg.role.size() + content.size()) * 1.25;
    }

    std::vector<char> buf(alloc_size);

    // run the first time to get the total output length
    const auto & src = tmpls->template_default_source;
    int32_t      res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt,
                                                 buf.data(), buf.size());

    // error: chat template is not supported
    if (res < 0) {
        // if the custom "tmpl" is not supported, we throw an error
        // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
        throw std::runtime_error("this custom template is not supported");
    }

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt, buf.data(),
                                        buf.size());
    }

    common_chat_local_params params;
    params.prompt = std::string(buf.data(), res);
    GGML_ASSERT(inputs.json_schema.empty());
    params.grammar = inputs.grammar;
    return params;
}

common_chat_local_params common_chat_local_templates_apply(const struct common_chat_local_templates *        tmpls,
                                                           const struct common_chat_local_templates_inputs & inputs) {
    GGML_ASSERT(tmpls != nullptr);
    GGML_ASSERT(!inputs.use_jinja);
    return common_chat_local_templates_apply_legacy(tmpls, inputs);
}

std::string common_chat_local_format_single(const struct common_chat_local_templates * tmpls,
                                            const std::vector<common_chat_local_msg> & past_msg,
                                            const common_chat_local_msg & new_msg, bool add_ass, bool use_jinja) {
    common_chat_local_templates_inputs inputs;
    inputs.use_jinja = use_jinja;

    std::string fmt_past_msg;
    if (!past_msg.empty()) {
        inputs.messages              = past_msg;
        inputs.add_generation_prompt = false;
        fmt_past_msg                 = common_chat_local_templates_apply(tmpls, inputs).prompt;
    }
    std::ostringstream ss;
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    inputs.messages.push_back(new_msg);
    inputs.add_generation_prompt = add_ass;
    auto fmt_new_msg             = common_chat_local_templates_apply(tmpls, inputs).prompt;
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}
