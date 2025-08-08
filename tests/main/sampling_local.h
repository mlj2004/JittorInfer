#pragma once

#include "llama.h"

// common_sampler_local extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler_local also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler_local also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler_local;

// llama_sampler API overloads

struct common_sampler_local * common_sampler_local_init(const struct llama_model *                  model,
                                                        const struct common_params_local_sampling & params);

void common_sampler_local_free(struct common_sampler_local * gsmpl);

// if accept_grammar is true, the token is accepted both by the sampling chain and the grammar
void common_sampler_accept(struct common_sampler_local * gsmpl, llama_token token, bool accept_grammar);
void common_sampler_reset(struct common_sampler_local * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler_local * gsmpl, struct llama_context * ctx, int idx,
                                  bool grammar_first = false);

// get the last accepted token
llama_token common_sampler_local_last(const struct common_sampler_local * gsmpl);

// performance metrics
void common_perf_print(const struct llama_context * ctx, const struct common_sampler_local * gsmpl);
