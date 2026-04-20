# Model Version History

## Overview
This document tracks all trained models for the drowsiness detection system, including architecture changes, datasets, and performance metrics.

---

## Model Versions

### drowsiness_model_V2
**Date:** [w/c 10/11/2025]  
**Status:** ⚠️ Overwritten by retraining  
**Architecture:** Basic neural network  
**Dataset:** Video frames (original dataset)  

**Description:**  
- Overwritten initial training of V1
- Second training attempt seeking better results
- Performance superior to V1, hence overwrote it

Note: no results available

---

### drowsiness_model_V2_pi
**Date:** [w/c 24/11/2025]  
**Status:** ✅ Deployed on Raspberry Pi  
**Architecture:** Basic neural network (TFLite optimized)  
**Dataset:** Same as V2 

**Description:**  
- Converted from drowsiness_model_V2
- Optimized for edge deployment using TensorFlow Lite
- Designed for real-time inference on Raspberry Pi
- Note: despite increased accuracy over individual frame (for both datasets and myself), entire videos are still skewing results (due to large average disparity in datasets)

Note: no results available

---

### drowsiness_model_V3
**Date:** [18/12/2025]  
**Status:** Complete  
**Architecture:** Basic neural network  
**Dataset:** BINA dataset (new source)

**Description:**  
- Attempted to use different dataset (BINA) instead of video-sourced images
- Goal: Improve generalization with external dataset

**Results:**
- Improved reliability, but lead to some strange results.
- Some case lables were entirely wrong, with others for the same image being correct

---

### xception_drowsiness_model
**Date:** Christmas Break ~ [26/12/2025]  
**Status:** ⚠️ Deprecated - unreliable  
**Architecture:** Xception (transfer learning)  
**Dataset:** BINA dataset 

**Description:**  
- Initial Xception model implementation
- Used foreign language labels
- Mismatched reasoning between labels and training

**Results**
- **Accuracy: ~65-70%** (over Validation data for definative event identification)
- this was worse than the Simple Nueral Network from before that used self-made thresholds
**Issues:**
- Limited reliability due to label inconsistencies
- Superseded by V2

---

### xception_drowsiness_model_v2
**Date:** [11/02/2026]  
**Status:** Improved baseline  
**Architecture:** Xception (transfer learning)  
**Dataset:** BINA dataset

**Description:**  
- English-checked labels
- Code-based label assignment for consistency
- Fixed labeling issues from V1

**Results**
- **Accuracy: ~78%** (over validation data)

**Improvements:**
- Corrected label mapping
- More reliable than V1

---

### xception_drowsiness_model_v3_imageChanges
**Date:** [14/02/2026]  
**Status:** Experimental  
**Architecture:** Xception (transfer learning)  
**Dataset:** Same as V2 with image preprocessing

**Description:**  
- Based on xception_drowsiness_model_v2
- Added image preprocessing/augmentation changes (V3 did not have this image variation)

**Results:**
- **Accuravy: 75.91%** (over validaiton data)
- General accuracy depreciated due to variations in the dataset being implementd

---

### xception_drowsiness_model_v4 (2 classifications)
**Date:** [17/02/2026]  
**Status:** Best performing  
**Architecture:** Xception (transfer learning)  
**Dataset:** Bina - Opinionated binary classifier (Tired vs Alert) 

**Description:**  
- Simplified to 2-class problem (tired/alert)
- Used opinionated classification criteria
- Clear decision boundaries

**Results:**
- **Accuracy: 65.91%**
- Better than multi-class approaches

---

### xception_drowsiness_model_v5 (2 classifications)
**Date:** [17/02/2026]  
**Status:** Retraining attempt  
**Architecture:** Xception (transfer learning)  
**Dataset:** Bina

**Description:**  
- Retrained same architecture as V4
- Goal: Improve upon 65.91% accuracy
- Different random initialization/training run

**Results:**
- **Accuracy: 63.50%**
- Slightly worse than original V4 training
- Suggests V4 was near optimal for this approach

---

### xception_drowsiness_model_v6 (2 classifications)
**Date:** [17/02/2026]  
**Status:** Best performing for single classification
**Architecture:** Xception (transfer learning)  
**Dataset:** Bina Uni

**Description:**  
- restructured to give one final classification (tired or alert) only
- uses opinion based learning, applies my opinion to a dataset

**Results:**
- **Accuracy on validation: 73.72%**
- **Accuracy on me: 63.27%** (for frames taking in my room - an environment completely different to the training data)
- full in-depth review available in the Report, in the Evaluation Section
- Over all test cases and **Accuracy: 77.34%** which is great considering this model is 100% opinion based and removes the need for a threshold to be determined (which can vary between users)

---

## Performance Comparison

| Model | Date | Architecture | Classes | Accuracy | Status |
|-------|------|-------------|---------|----------|--------|
| drowsiness_model_V2 | - | Basic NN | [tired, awake] | N/A | Overwritten |
| drowsiness_model_V2_pi | - | Basic NN (TFLite) | [tired, awake] | N/A | Deployed |
| drowsiness_model_V3 | - | Basic NN | [open, closed, yawn, no yawn] | N/A | Experimental |
| xception_drowsiness_model | 26/12/2025 | Xception | [open, closed, yawn, no yawn] | 65-70% | Deprecated |
| xception_drowsiness_model_v2 | 11/02/2026 | Xception | [open, closed, yawn, no yawn] | ~78% | Baseline |
| xception_drowsiness_model_v3_imageChanges | 14/02/2026 | Xception | [open, closed, yawn, no yawn] | 75.91% | Experimental |
| xception_drowsiness_model_v4 | 17/02/2026 | Xception | [tired, alert] | 65.91% | Medium |
| xception_drowsiness_model_v5 | 17/02/2026 | Xception | [tired, alert] | 63.50% | Lower |
| xception_drowsiness_model_v6 | 17/02/2026 | Xception | [tired, alert] | **77.34%** | ✅ Best - Integrated to Webpage on the Pi to form a final result |


---

## Key Insights

1. **Xception outperforms basic NN**: Transfer learning approach shows promise
2. **Binary classification works better**: 2-class (tired/alert) achieves 73.72% vs unclear multi-class performance
3. **Label quality matters**: V2 English labels significantly improved reliability
4. **Training variance**: V4 retrain shows ~2.4% accuracy variance, suggesting training stability issues or near-optimal convergence

<!-- General structure was formed by Gemini.ai becuase I wanted to look nice-->