# Random Forest Version 2 — Development and Evaluation Log

## 1. Purpose

This document records the development and evaluation activities carried out during the improvement of the respiratory disease prediction system.

The purpose of Version 2 is to improve the handling of incomplete patient information and reduce unsupported, overconfident predictions while preserving Version 1 for comparison.

## 2. Version 1 Sparse-Input Evaluation

The original Random Forest model was tested using short symptom combinations to examine its behaviour when users provided limited information.

The tests were performed using the saved Version 1 model and its original feature structure. Unreported evidence features were encoded as zero, following the existing Version 1 input representation.

The patient age was set to 21 years. Sex was encoded as male unless otherwise specified.

### 2.1 Evaluation Results

| Input Symptoms | Predicted Condition | Model Probability |
|---|---|---|
| Fever only | Croup | 37.25% |
| Cough only | Tuberculosis | 62.75% |
| Fever and cough (male) | Tuberculosis | 99.75% |
| Fever and cough (female) | Tuberculosis | 96.75% |
| Fever, cough and muscle aches | Tuberculosis | 90.00% |
| Fever, cough and sore throat | Tuberculosis | 75.50% |
| Runny nose and sore throat | Allergic sinusitis | 69.25% |
| Cough, heavy sweating and weight loss | Tuberculosis | 74.25% |

### 2.2 Observations

The evaluation showed that Version 1 frequently selected tuberculosis when limited symptoms were provided.

For example, fever and cough produced a tuberculosis prediction with a model probability of 99.75% for a 21-year-old male.

The results also showed differences in model probabilities between male and female inputs.

These findings suggest that Version 1 may be unreliable when processing incomplete symptom information, despite its high accuracy on the original synthetic DDXPlus test dataset.

The exploratory scenarios did not contain independently verified diagnoses. Therefore, these results demonstrate potentially misleading confidence behaviour rather than a measured clinical misdiagnosis rate.

## 3. Version 2 Feature Representation

A new Python module, `nlp/symptom_state.py`, was developed to represent patient symptoms using three states:

- Present: The patient confirms having a symptom.
- Absent: The patient explicitly denies having a symptom.
- Unknown: The patient has not provided information about the symptom.

The module also supports storing minimum and maximum symptom duration in days.

For example, a patient reporting a cough lasting five days while denying headache can be represented with cough marked as present, headache marked as absent and other unreported symptoms marked as unknown.

This representation is intended to support future NLP improvements and Version 2 model development.

The new module does not yet replace Version 1's model input encoding.

## 4. Automated Testing

Eight automated tests were added in `tests/test_symptom_state.py`.

The tests covered:

- Present, absent and unknown symptom states.
- Coverage of supported canonical symptoms.
- Symptom duration storage.
- Unsupported symptom rejection.
- Invalid status rejection.
- Duration validation.

All eight new tests passed.

The complete test suite was also executed, including the existing NLP tests.

**Test results:**

- Existing NLP tests: 12 passed.
- New symptom-state tests: 8 passed.
- Total: 20 tests passed.
- Failed tests: 0.

These results confirm the tested symptom-state functionality but do not establish model prediction accuracy or resolve the known sentence-level NLP issues.

## 5. Identified Limitations

The following issues remain:

- Version 1 produces highly confident predictions from limited symptom information.
- Existing NLP processing does not correctly handle some negated symptoms.
- Symptom duration is not yet extracted automatically from user text.
- Certain respiratory expressions and urgent warning signs are not consistently recognised.
- The Version 2 feature representation has not yet been integrated into model training.
- The system has not been clinically validated using Kenyan patient data.

## 6. Planned Improvements

The following development activities will address these limitations:

- Improve English and Kiswahili symptom processing.
- Implement sentence-level negation and duration extraction.
- Improve red-flag identification.
- Refine training-data augmentation.
- Train Random Forest Version 2.
- Introduce evaluated uncertainty handling.
- Compare Version 2 with Version 1 using realistic symptom inputs.

## 7. Conclusion

The initial evaluation identified important limitations in the behaviour of Random Forest Version 1 when processing incomplete symptom information.

The new symptom-state module provides a foundation for representing patient information more accurately.

Further NLP improvements, model retraining and independent evaluation are required before Version 2 can be considered ready for integration into the health-awareness application.