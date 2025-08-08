// Chat support (incl. tool call grammar constraining & output parsing) w/ generic & custom template handlers.

#pragma once

#include <string>
#include <vector>

#include "common_local.h"

struct common_chat_templates;

struct common_chat_tool_call {
    std::string name;
    std::string arguments;
    std::string id;
};

struct common_chat_msg_content_part {
    std::string type;
    std::string text;
};

struct common_chat_msg {
    std::string                               role;
    std::string                               content;
    std::vector<common_chat_msg_content_part> content_parts = {};
    std::vector<common_chat_tool_call>        tool_calls    = {};
    std::string                               reasoning_content;
    std::string                               tool_name;
};

struct common_chat_tool {
    std::string name;
    std::string description;
    std::string parameters;
};

enum common_chat_tool_choice {
    COMMON_CHAT_TOOL_CHOICE_AUTO,
    COMMON_CHAT_TOOL_CHOICE_REQUIRED,
    COMMON_CHAT_TOOL_CHOICE_NONE,
};

enum common_chat_format {
    COMMON_CHAT_FORMAT_CONTENT_ONLY,
    COMMON_CHAT_FORMAT_GENERIC,
};

struct common_chat_templates_inputs {
    std::vector<common_chat_msg>  messages;
    std::string                   grammar;
    std::string                   json_schema;
    bool                          add_generation_prompt = true;
    std::vector<common_chat_tool> tools;
};

struct common_chat_params {
    common_chat_format                  format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::string                         prompt;
    std::string                         grammar;
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string>            preserved_tokens;
    std::vector<std::string>            additional_stops;
};

void common_chat_templates_free(struct common_chat_templates * tmpls);

struct common_chat_templates_deleter {
    void operator()(common_chat_templates * tmpls) { common_chat_templates_free(tmpls); }
};

typedef std::unique_ptr<struct common_chat_templates, common_chat_templates_deleter> common_chat_templates_ptr;

common_chat_templates_ptr common_chat_templates_init(const struct llama_model * model,
                                                     const std::string &        chat_template_override,
                                                     const std::string &        bos_token_override = "",
                                                     const std::string &        eos_token_override = "");

struct common_chat_params common_chat_templates_apply(const struct common_chat_templates *        tmpls,
                                                      const struct common_chat_templates_inputs & inputs);

// Returns an example of formatted chat
std::string common_chat_format_example(const struct common_chat_templates * tmpls);

common_chat_msg common_chat_parse(const std::string & input, common_chat_format format);

// Parses a JSON array of messages in OpenAI's chat completion API format.
// T can be std::string containing JSON or nlohmann::ordered_json
template <class T> std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const T & messages);
template <class T>
T common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text = false);

// Parses a JSON array of tools in OpenAI's chat completion tool call API format.
// T can be std::string containing JSON or nlohmann::ordered_json
template <class T> std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const T & tools);
template <class T> T common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools);
