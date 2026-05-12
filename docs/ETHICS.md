# Ethical Analysis — Campus Safety AI System

> BCS407 – Artificial Intelligence | Canadian University Dubai | 2026

---

## 1. Ethical Frameworks Applied

### 1.1 ACM Code of Ethics — Principle 1.2: Avoid Harm

> *"Avoid harm to others"* — ACM Code of Ethics, Section 1.2

**Application:**
- The system is designed exclusively for **safety enforcement**, not disciplinary action
- False negatives (missed violations) are prioritized for correction over false positives, because a missed safety hazard can cause physical injury
- **Mitigation measure:** Confidence threshold set at 0.25 (conservative) to minimize false negatives while maintaining manageable false positive rates
- System alerts trigger **human verification** before any enforcement action

**Risk Assessment:** Medium — The system could cause harm if alerts lead to punitive actions without human review. This is mitigated by the human-in-the-loop design.

### 1.2 ACM Code of Ethics — Principle 2.5: Respect Privacy

> *"Respect privacy"* — ACM Code of Ethics, Section 2.5

**Application:**
- The system detects **objects** (helmets, signs, alarms), **not people**
- No facial recognition capability is implemented or planned
- No personally identifiable information (PII) is captured or stored
- Images are processed in real-time and discarded immediately after analysis
- No recording or storage of surveillance footage by this system

**Risk Assessment:** Low — By design, the system cannot identify individuals.

### 1.3 IEEE Code of Ethics — Public Safety Obligation

> *"To accept responsibility in making decisions consistent with the safety, health, and welfare of the public"* — IEEE Code of Ethics, Principle 1

**Application:**
- The system is positioned as a **supplement** to existing safety measures, not a replacement
- Regular testing and validation ensure the model maintains accuracy standards
- Documentation clearly communicates the system's capabilities and limitations
- The model's failure modes are documented and communicated to stakeholders

**Risk Assessment:** High if over-relied upon — the system must not create a false sense of security that leads to reduced human safety oversight.

### 1.4 IST/CIPS Code of Ethics — Responsible Use of Technology

> *"Ensure that the products of their efforts are used in a socially responsible manner"* — IST Code of Ethics

**Application:**
- The system is transparent about its AI-driven nature
- Campus occupants are informed through signage that AI-based monitoring is in operation
- All system outputs are auditable — every detection includes timestamps, confidence scores, and bounding boxes
- Data governance policies ensure compliance with institutional IT policies

**Risk Assessment:** Low — The system enhances rather than replaces human judgment.

### 1.5 Canadian Privacy Framework (PIPEDA / Provincial OH&S)

| Regulation | Relevance | Compliance Measure |
|-----------|-----------|-------------------|
| **PIPEDA** (Personal Information Protection and Electronic Documents Act) | Governs collection, use, and disclosure of personal information | No PII collected; system processes objects only |
| **Provincial OH&S** (Occupational Health & Safety) | Requires employers to provide safe workplaces | System supports employer obligation by monitoring compliance |
| **FIPPA** (Freedom of Information and Protection of Privacy Act) | Applies to public institutions in some provinces | Data retention policies ensure no unauthorized storage |
| **CSA Standards** (Canadian Standards Association) | Workplace safety standards | System aligns with CSA Z1000 (Risk Management) principles |

---

## 2. Privacy Analysis

### 2.1 Data Flow Diagram

```
Camera Feed → [Real-time Processing] → Detection Results
                    ↓
            [Immediate Discard]
                    ↓
            No Storage / No Recording
```

### 2.2 Privacy Safeguards

| Safeguard | Implementation |
|-----------|---------------|
| No facial recognition | Model does not detect faces; trained only on safety objects |
| No PII storage | Detection results contain only object class, confidence, and bounding box coordinates |
| Real-time processing | Images are processed in memory and never written to disk |
| No behavioral analysis | System does not track, identify, or profile individuals |
| Data minimization | Only the minimal data needed (class, confidence, location) is output |
| Access control | System outputs are accessible only to authorized safety personnel |

---

## 3. Bias Analysis

### 3.1 Potential Bias Sources

| Source | Risk Level | Mitigation |
|--------|-----------|------------|
| **Training data imbalance** | Medium | Addressed through class equalization (2,500 images per class) |
| **Lighting conditions** | Medium | Augmentation includes brightness, contrast, and gamma variations |
| **Camera angles** | Low-Medium | Source datasets use multiple perspectives; flip augmentation adds diversity |
| **Object appearance** | Medium | Multiple visual variants in training data (different helmet colors, sign sizes) |
| **Cultural context** | Low | Safety signs are standardized internationally (ISO 7010); colors are universal |

### 3.2 Fairness Considerations

- The system does not make decisions about specific individuals — it detects the presence/absence of safety equipment
- No demographic classification is performed or possible
- Detection performance is consistent across lighting conditions due to augmentation
- The system is **equally applicable** to all campus occupants regardless of identity

---

## 4. Transparency and Accountability

### 4.1 Transparency Measures

| Measure | Description |
|---------|-------------|
| **Open documentation** | All methodology, metrics, and limitations are publicly documented |
| **Interpretable outputs** | Every detection includes bounding box visualization showing exactly what was detected |
| **Confidence scores** | Each prediction includes a confidence score (0–1) for informed decision-making |
| **Model cards** | Model architecture, training data, and performance metrics are documented |

### 4.2 Accountability Framework

| Question | Answer |
|----------|--------|
| **Who is responsible if the system misses a hazard?** | The institution retains ultimate responsibility; the system is a tool, not a decision-maker |
| **Who is responsible if the system generates false alarms?** | System operators should tune confidence thresholds based on deployment context |
| **How are errors corrected?** | Misclassified images can be added to retraining data; model is periodically updated |
| **Who has access to outputs?** | Authorized safety personnel only; no public access to raw detections |

---

## 5. Ethics Impact Assessment

| Dimension | Risk Level | Specific Concern | Mitigation |
|-----------|-----------|-----------------|------------|
| **Privacy** | 🟢 Low | No PII processed or stored | Object-only detection; real-time discard |
| **Bias** | 🟡 Medium | Potential for unequal detection across conditions | Balanced dataset; diverse augmentations |
| **Safety** | 🟡 Medium | Over-reliance on system; missed hazards | Human-in-the-loop; system is advisory, not autonomous |
| **Accountability** | 🟡 Medium | Unclear responsibility for errors | Clear governance; system as decision-support tool |
| **Transparency** | 🟢 Low | Black-box AI concern | Full documentation; interpretable visual outputs |
| **Consent** | 🟢 Low | No individual identification | No biometric data collected |
| **Data Security** | 🟢 Low | Minimal data retained | No persistent storage of images or detections |

---

## 6. Recommendations for Ethical Deployment

1. **Inform all campus occupants** that AI-based safety monitoring is in operation via clear signage
2. **Never use the system for surveillance, disciplinary action, or personnel tracking**
3. **Maintain human oversight** — all detections should be reviewed by trained safety personnel before action
4. **Conduct regular audits** (quarterly) to verify detection accuracy and check for bias
5. **Provide opt-out mechanisms** for areas where individuals have heightened privacy expectations (e.g., restrooms, private offices)
6. **Establish clear escalation procedures** for when the system reports a safety violation
7. **Periodically retrain** the model with updated data to maintain accuracy and adapt to new campus environments
8. **Document and publish** system performance metrics and any incidents of false positives/negatives

---

## 7. Conclusion

This campus safety monitoring system is designed with ethical principles as a foundational requirement, not an afterthought. The system:

- ✅ Protects privacy by not identifying individuals
- ✅ Supports safety without enabling surveillance
- ✅ Maintains transparency through documentation and interpretable outputs
- ✅ Addresses bias through balanced datasets and augmentation
- ✅ Preserves human accountability through advisory (not autonomous) operation

By adhering to the ACM, IEEE, and IST codes of ethics, as well as Canadian privacy regulations, this system demonstrates that AI-based safety monitoring can be implemented responsibly and ethically.

---

### References

1. ACM. (2018). *ACM Code of Ethics and Professional Conduct.* https://www.acm.org/code-of-ethics
2. IEEE. (2020). *IEEE Code of Ethics.* https://www.ieee.org/about/corporate/governance/p7-8.html
3. CPSR. (n.d.). *ICT Code of Ethics.* https://ethics.acm.org/code-of-ethics
4. Office of the Privacy Commissioner of Canada. (2024). *PIPEDA Fair Information Principles.* https://www.priv.gc.ca
5. International Organization for Standardization. (2011). *ISO 26000: Guidance on Social Responsibility.*