#ifndef GLOBAL_STATE_HPP
#define GLOBAL_STATE_HPP

#include "MobileNetModel.hpp"
#include "Classifier.hpp"
#include "inference_mngt.h"
#include "ClassificationResult.hpp"
#include "ImgClassProcessing.hpp"
#include "imlib.h"
#include "framebuffer.h"
#include "LogConfig.hpp"

#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
#include "ZOTrainer.hpp"
#endif

/* ------------------------------------------------------------------ */
/*  Global variables shared between main.cpp and uart/uart_cmd.cpp    */
/* ------------------------------------------------------------------ */

/* Model & Inference Objects */
extern arm::app::MobileNetModel model;
extern arm::app::Classifier classifier;
extern bool modelLoaded;

#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
extern arm::app::MobileNetModel extractorModel;
extern arm::app::MobileNetModel classifierModel;
extern bool splitModelLoaded;
#endif

/* Tensor Info (valid only after model load) */
extern TfLiteTensor *inputTensor;
extern int inputImgCols;
extern int inputImgRows;
extern uint32_t inputChannels;

/* Labels & Results */
extern std::vector<std::string> labels;
extern std::vector<arm::app::ClassificationResult> results;

/* Processing Objects (heap allocated) */
extern arm::app::ImgClassPreProcess  *preProcess;
extern arm::app::ImgClassPostProcess *postProcess;

/* Framebuffer & ROI */
/* Note: fb_array is static in main.cpp/omv_init, but we access via frameBuffer struct */
extern image_t frameBuffer;
extern rectangle_t roi;

/* Frame synchronisation: ensures inference and display use consistent frame */
extern uint32_t inferenceFrameCount;      /* Increments on each inference start */
extern bool frameBufferFrozen;            /* When true, skip camera capture updates */

#if defined(__USE_CCAP__)
    /* Camera capture function prototype from ImageSensor.h/main.cpp context */
    /* Since ImageSensor.h is C, we can just include it in cpp, but here we declare what we need */
    // extern void ImageSensor_WaitCaptureDone(void); // Defined in ImageSensor.h
#endif

/* ------------------------------------------------------------------ */
/*  User Interface State                                             */
/* ------------------------------------------------------------------ */
extern std::string lastInferenceResult;

/* ------------------------------------------------------------------ */
/*  ZO Trainer State (split model only)                              */
/* ------------------------------------------------------------------ */
#if defined(USE_SPLIT_MODEL) && (USE_SPLIT_MODEL == 1)
extern ZOTrainer* zoTrainer;
extern int        zoTargetLabel;
extern float      zoLearningRate;
extern int        zoNumPerturbations;

/* ZO gradient-estimation method: 0 = Node Perturbation (NP, default),
 * 1 = Weight Perturbation (WP). Set once via `zo_set_method` BEFORE the first
 * training step; locked thereafter until `zo_reset` to keep a comparison run on
 * a single method. */
enum ZOMethod { ZO_METHOD_NP = 0, ZO_METHOD_WP = 1 };
extern int        zoMethod;
#endif

/* ------------------------------------------------------------------ */
/*  Helper functions or definitions                                   */
/* ------------------------------------------------------------------ */
/* You might want to expose some helper functions if needed */

#endif /* GLOBAL_STATE_HPP */
