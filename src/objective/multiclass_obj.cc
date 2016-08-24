/*!
 * Copyright 2015 by Contributors
 * \file multi_class.cc
 * \brief Definition of multi-class classification objectives.
 * \author Tianqi Chen
 */
#include <dmlc/omp.h>
#include <dmlc/parameter.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <algorithm>
#include <utility>
#include "../common/math.h"

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(multiclass_obj);

struct SoftmaxMultiClassParam : public dmlc::Parameter<SoftmaxMultiClassParam> {
  int num_class;
  // declare parameters
  DMLC_DECLARE_PARAMETER(SoftmaxMultiClassParam) {
    DMLC_DECLARE_FIELD(num_class).set_lower_bound(1)
        .describe("Number of output class in the multi-class classification.");
  }
};

class SoftmaxMultiClassObj : public ObjFunction {
 public:
  explicit SoftmaxMultiClassObj(bool output_prob)
      : output_prob_(output_prob) {
  }
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }
  void GetGradient(const std::vector<float>& preds,
                   const MetaInfo& info,
                   int iter,
                   std::vector<bst_gpair>* out_gpair) override {
    CHECK_NE(info.labels.size(), 0) << "label set cannot be empty";
    CHECK(preds.size() == (static_cast<size_t>(param_.num_class) * info.labels.size()))
        << "SoftmaxMultiClassObj: label size and pred size does not match";
    out_gpair->resize(preds.size());
    const int nclass = param_.num_class;
    const omp_ulong ndata = static_cast<omp_ulong>(preds.size() / nclass);

    int label_error = 0;
    #pragma omp parallel
    {
      std::vector<float> rec(nclass);
      #pragma omp for schedule(static)
      for (omp_ulong i = 0; i < ndata; ++i) {
        for (int k = 0; k < nclass; ++k) {
          rec[k] = preds[i * nclass + k];
        }
        common::Softmax(&rec);
        int label = static_cast<int>(info.labels[i]);
        if (label < 0 || label >= nclass)  {
          label_error = label; label = 0;
        }
        const float wt = info.GetWeight(i);
        for (int k = 0; k < nclass; ++k) {
          float p = rec[k];
          const float h = 2.0f * p * (1.0f - p) * wt;
          if (label == k) {
            out_gpair->at(i * nclass + k) = bst_gpair((p - 1.0f) * wt, h);
            //std::cout << (p - 1.0f) * wt << " " << h << " " << p << std::endl;
          } else {
            out_gpair->at(i * nclass + k) = bst_gpair(p* wt, h);
            //std::cout << p * wt << " " << h << " " << p << std::endl;
          }
        }
      }
    }
    CHECK(label_error >= 0 && label_error < nclass)
        << "SoftmaxMultiClassObj: label must be in [0, num_class),"
        << " num_class=" << nclass
        << " but found " << label_error << " in label.";
  }
  void PredTransform(std::vector<float>* io_preds) override {
    this->Transform(io_preds, output_prob_);
  }
  void EvalTransform(std::vector<float>* io_preds) override {
    this->Transform(io_preds, true);
  }
  const char* DefaultEvalMetric() const override {
    return "merror";
  }

 private:
  inline void Transform(std::vector<float> *io_preds, bool prob) {
    std::vector<float> &preds = *io_preds;
    std::vector<float> tmp;
    const int nclass = param_.num_class;
    const omp_ulong ndata = static_cast<omp_ulong>(preds.size() / nclass);
    if (!prob) tmp.resize(ndata);

    #pragma omp parallel
    {
      std::vector<float> rec(nclass);
      #pragma omp for schedule(static)
      for (omp_ulong j = 0; j < ndata; ++j) {
        for (int k = 0; k < nclass; ++k) {
          rec[k] = preds[j * nclass + k];
        }
        if (!prob) {
          tmp[j] = static_cast<float>(
              common::FindMaxIndex(rec.begin(), rec.end()) - rec.begin());
        } else {
          common::Softmax(&rec);
          for (int k = 0; k < nclass; ++k) {
            preds[j * nclass + k] = rec[k];
          }
        }
      }
    }
    if (!prob) preds = tmp;
  }
  // output probability
  bool output_prob_;
  // parameter
  SoftmaxMultiClassParam param_;
};


/********************************/
/** Implement Brier objective ***/
/********************************/

struct BrierMultiClassParam : public dmlc::Parameter<BrierMultiClassParam> {
  int num_class;
  //float * class_weights;
  // declare parameters
  DMLC_DECLARE_PARAMETER(BrierMultiClassParam) {
    DMLC_DECLARE_FIELD(num_class).set_lower_bound(1)
        .describe("Number of output class in the multi-class classification.");
  }
};

class BrierMultiClassObj : public ObjFunction {
 public:
  explicit BrierMultiClassObj(bool output_prob)
      : output_prob_(output_prob) {
  }
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }
  void GetGradient(const std::vector<float>& preds,
                   const MetaInfo& info,
                   int iter,
                   std::vector<bst_gpair>* out_gpair) override {
    CHECK_NE(info.labels.size(), 0) << "label set cannot be empty";
    CHECK(preds.size() == info.labels.size())
        << "BrierMultiClassObj: label size and pred size does not match" << std::endl
        << "Labels shape must be (n_samples, n_classes)";
    out_gpair->resize(preds.size());
    const int nclass = param_.num_class;
    const omp_ulong ndata = static_cast<omp_ulong>(preds.size() / nclass);
    const float class_weights[] = {1.35298455691, 1.38684574053, 1.59587388404,
        1.35318713948, 0.347783666015, 0.661081706198, 1.04723628621,
        0.398865222651, 0.207586320237, 1.50578335208, 0.110181365961,
        1.07803284435, 1.36560417316, 1.17024113802, 1.1933637414, 
        1.1803704493, 1.34414875433, 1.11683830693, 1.08083910312, 0.503152249073};

    // TODO Check that: true probas == labels ??
    // const std::vector<float> true_probas = info.labels;
    //std::cout << "size: " << info.labels.size() << std::endl;

    #pragma omp parallel
    {
      //std::cout << "nclass: "<< nclass << std::endl;
      std::vector<float> rec(nclass);
      std::vector<float> true_p(nclass);
      #pragma omp for schedule(static)
      for (omp_ulong i = 0; i < ndata; ++i) {
        // load predictions for i-th example into rec
        for (int k = 0; k < nclass; ++k) {
          rec[k] = preds[i * nclass + k];
          true_p[k] = info.labels[i * nclass + k];
          //std::cout << true_p[k] << " ";
        }
        //std::cout << std::endl;
        // apply softmax to rec
        common::Softmax(&rec);
        //int label = static_cast<int>(info.labels[i]); -> we don't care about labels anymore
        // weight of the example
        const float wt = info.GetWeight(i);
        for (int k = 0; k < nclass; ++k) {
          float grad = 0;
          float hess = 0;
          float p_hat_k = rec[k];
          float p_true_k = true_p[k];
          grad += -class_weights[k] * (p_true_k - p_hat_k);
          //TODO: Hessian: ATTENTION AUX SIGNES: + ou - hessienne? 

          hess = class_weights[k] * ( - p_true_k*p_hat_k + 2*p_true_k*p_hat_k*p_hat_k - 2*p_hat_k*p_hat_k - 4*p_hat_k*p_hat_k*p_hat_k );
 

          for (int c = 0; c < nclass; ++c){
            float p_hat_c = rec[c];
            float p_true_c = true_p[c];
            grad += class_weights[c] * (p_true_c - p_hat_c) * p_hat_c;
            hess += class_weights[c] * ( - p_true_c*p_hat_k + 3*p_hat_k*p_hat_c - p_hat_c) * p_hat_c * p_hat_k;

          }
          grad *= 2 * p_hat_k;
          hess *= 2 ;

          // CONSTANT HESSIAN
          hess = info.hessian[0];

          // LOGS
          //std::cout << grad << " " << hess << " " << p_hat_k << " " << p_true_k << std::endl;

          out_gpair->at(i * nclass + k) = bst_gpair(grad * wt, hess * wt);
        }
      }
    }
    //CHECK(true_probas_error >= 0 && label_error < nclass)
    //    << "BrierMultiClassObj: probas must be in (0, 1) and sum to 1,"
    //    << " num_class=" << nclass
    //    << " but found " << true_probas_error << " in label.";
  }
  void PredTransform(std::vector<float>* io_preds) override {
    this->Transform(io_preds, output_prob_);
  }
  void EvalTransform(std::vector<float>* io_preds) override {
    this->Transform(io_preds, true);
  }
  const char* DefaultEvalMetric() const override {
    return "merror";
  }

 private:
  inline void Transform(std::vector<float> *io_preds, bool prob) {
    std::vector<float> &preds = *io_preds;
    std::vector<float> tmp;
    const int nclass = param_.num_class;
    const omp_ulong ndata = static_cast<omp_ulong>(preds.size() / nclass);
    if (!prob) tmp.resize(ndata);

    #pragma omp parallel
    {
      std::vector<float> rec(nclass);
      #pragma omp for schedule(static)
      for (omp_ulong j = 0; j < ndata; ++j) {
        for (int k = 0; k < nclass; ++k) {
          rec[k] = preds[j * nclass + k];
        }
        if (!prob) {
          tmp[j] = static_cast<float>(
              common::FindMaxIndex(rec.begin(), rec.end()) - rec.begin());
        } else {
          common::Softmax(&rec);
          for (int k = 0; k < nclass; ++k) {
            preds[j * nclass + k] = rec[k];
          }
        }
      }
    }
    if (!prob) preds = tmp;
  }
  // output probability
  bool output_prob_;
  // parameter
  BrierMultiClassParam param_;
};

// register the ojective functions
DMLC_REGISTER_PARAMETER(SoftmaxMultiClassParam);
DMLC_REGISTER_PARAMETER(BrierMultiClassParam);

XGBOOST_REGISTER_OBJECTIVE(SoftmaxMultiClass, "multi:softmax")
.describe("Softmax for multi-class classification, output class index.")
.set_body([]() { return new SoftmaxMultiClassObj(false); });

XGBOOST_REGISTER_OBJECTIVE(SoftprobMultiClass, "multi:softprob")
.describe("Softmax for multi-class classification, output probability distribution.")
.set_body([]() { return new SoftmaxMultiClassObj(true); });

XGBOOST_REGISTER_OBJECTIVE(BrierMultiClass, "multi:brier")
.describe("Brier for multi-class classification, output probability distribution.")
.set_body([]() { return new BrierMultiClassObj(true); });

}  // namespace obj
}  // namespace xgboost
