# Model Version History

## Overview
This document tracks all trained models for the drowsiness detection system, including architecture changes, datasets, and performance metrics.

---

## Model Versions

### drowsiness_model_V2
**Date:** [Add date]  
**Status:** ⚠️ Overwritten by retraining  
**Architecture:** Basic neural network  
**Dataset:** Video frames (original dataset)  
**Description:**  
- Overwritten initial training of V1
- Second training attempt seeking better results
- Performance superior to V1, hence overwrote it

**Results:**
- Accuracy: [Add if available]
- Notes: Improved over V1

---

### drowsiness_model_V2_pi
**Date:** [Add date]  
**Status:** ✅ Deployed on Raspberry Pi  
**Architecture:** Basic neural network (TFLite optimized)  
**Dataset:** Same as V2  
**Description:**  
- Converted from drowsiness_model_V2
- Optimized for edge deployment using TensorFlow Lite
- Designed for real-time inference on Raspberry Pi

**Results:**
- Inference time: [Add if available]
- Memory footprint: [Add if available]

---

### drowsiness_model_V3
**Date:** [Add date]  
**Status:** Experimental  
**Architecture:** Basic neural network  
**Dataset:** BINA dataset (new source)  
**Description:**  
- Attempted to use different dataset (BINA) instead of video-sourced images
- Goal: Improve generalization with external dataset

**Results:**
- [Add results/notes on why continued or discontinued]

---

### xception_drowsiness_model
**Date:** [Add date]  
**Status:** ⚠️ Deprecated - unreliable  
**Architecture:** Xception (transfer learning)  
**Dataset:** [Specify]  
**Description:**  
- Initial Xception model implementation
- Used foreign language labels
- Mismatched reasoning between labels and training

**Issues:**
- Limited reliability due to label inconsistencies
- Superseded by V2

---

### xception_drowsiness_model_v2
**Date:** [Add date]  
**Status:** Improved baseline  
**Architecture:** Xception (transfer learning)  
**Dataset:** [Specify]  
**Description:**  
- English-checked labels
- Code-based label assignment for consistency
- Fixed labeling issues from V1

**Improvements:**
- Corrected label mapping
- More reliable than V1

---

### xception_drowsiness_model_v3_imageChanges
**Date:** [Add date]  
**Status:** Experimental  
**Architecture:** Xception (transfer learning)  
**Dataset:** Same as V2 with image preprocessing  
**Description:**  
- Based on xception_drowsiness_model_v2
- Added image preprocessing/augmentation changes

**Results:**
- [Add comparison to V2]

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
- **Accuracy on me: 63.27%**
//TODO - mention the distriibution of these %'s, hwo good is it compaired to the training dataset. 100% for Alertness, what % of the training data was alertness -> compair to the drowsiness figures

---

## Performance Comparison

| Model | Architecture | Classes | Accuracy | Status |
|-------|-------------|---------|----------|--------|
| drowsiness_model_V2 | Basic NN | [tired, awake] | [?] | Overwritten |
| drowsiness_model_V2_pi | Basic NN (TFLite) | [tired, awake] | [?] | Deployed |
| drowsiness_model_V3 | Basic NN | [eyes open, eyes closed, yawning, not yawning] | [?] | Experimental |
| xception_drowsiness_model | Xception | [eyes open, eyes closed, yawning, not yawning] | Low | Deprecated |
| xception_drowsiness_model_v2 | Xception | [eyes open, eyes closed, yawning, not yawning] | [?] | Baseline |
| xception_drowsiness_model_v3 | Xception | [eyes open, eyes closed, yawning, not yawning] | [?] | Experimental |
| xception_drowsiness_model_v4 | Xception | [tired, alert]| 65.91% | Medium |
| xception_drowsiness_model_v5 | Xception | [tired, alert] | 63.50% | Lower |
| xception_drowsiness_model_v6 | Xception | [tired, alert] | **73.72%** | ✅ Best |


---

## Key Insights

1. **Xception outperforms basic NN**: Transfer learning approach shows promise
2. **Binary classification works better**: 2-class (tired/alert) achieves 73.72% vs unclear multi-class performance
3. **Label quality matters**: V2 English labels significantly improved reliability
4. **Training variance**: V4 retrain shows ~2.4% accuracy variance, suggesting training stability issues or near-optimal convergence