#include <cstring>
#include <vector>

#include "llama-impl.h"
#include "llama-model-loader.h"
#include "llama-model.h"
#include "llama.h"

static void add_default_devices_into_backend(llama_model * model);
static int  llama_model_load(const std::string & fname, std::vector<std::string> & splits, llama_model & model,
                             llama_model_params & params);

struct llama_model * llama_model_load_from_file(const char * path_model, struct llama_model_params params) {
    std::vector<std::string> splits = {};

    ggml_time_init();

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback           = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned   percentage       = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    llama_model * model = new llama_model(params);

    // create list of devices to use with this model
    if (params.devices) {
        for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
            model->devices.push_back(*dev);
        }
    } else {
        add_default_devices_into_backend(model);
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
        if (params.main_gpu < 0 || params.main_gpu >= (int) model->devices.size()) {
            LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %d)\n", __func__, params.main_gpu,
                            (int) model->devices.size());
            llama_model_free(model);
            return nullptr;
        }
        ggml_backend_dev_t main_gpu = model->devices[params.main_gpu];
        model->devices.clear();
        model->devices.push_back(main_gpu);
    }

    const int status = llama_model_load(path_model, splits, *model, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        llama_model_free(model);
        return nullptr;
    }

    return model;
}

void add_default_devices_into_backend(llama_model * model) {
    std::vector<ggml_backend_dev_t> rpc_servers;
    // use all available devices
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:
            case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                // skip CPU backends since they are handled separately
                break;

            case GGML_BACKEND_DEVICE_TYPE_GPU:
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                    rpc_servers.push_back(dev);
                } else {
                    model->devices.push_back(dev);
                }
                break;
        }
    }
    // add RPC servers at the front of the list
    if (!rpc_servers.empty()) {
        model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());
    }
}

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
int llama_model_load(const std::string & fname, std::vector<std::string> & splits, llama_model & model,
                     llama_model_params & params) {
    // loading time will be recalculated after the first eval, so
    // we take page faults deferred by mmap() into consideration

    model.t_load_us = 0;
    time_meas tm(model.t_load_us);

    model.t_start_us = tm.t_start_us;

    try {
        llama_model_loader ml(fname, splits, params.use_mmap, params.check_tensors, params.kv_overrides);

        ml.print_info();

        model.hparams.vocab_only = params.vocab_only;

        try {
            model.load_arch(ml);
        } catch (const std::exception & e) {
            throw std::runtime_error("error loading model architecture: " + std::string(e.what()));
        }
        try {
            model.load_hparams(ml);
        } catch (const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        try {
            model.load_vocab(ml);
        } catch (const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model.load_stats(ml);
        // model.print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return 0;
        }

        if (!model.load_tensors(ml)) {
            return -2;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return -1;
    }

    return 0;
}
