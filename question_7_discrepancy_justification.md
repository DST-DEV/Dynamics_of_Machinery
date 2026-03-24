# Justification of Discrepancies
## Digital Twin vs Experimental FRFs and Mode Shapes

**Question 7 - Dynamics of Machinery**

---

## 1. FRF Discrepancies

The comparison between theoretical (digital twin) and experimental FRFs reveals several discrepancies that can be attributed to modeling assumptions, parameter uncertainties, and measurement limitations.

### 1.1 Natural Frequency Shifts

**Observation:** The theoretical natural frequencies show small deviations from experimental values (typically 1-5% error).

**Justification:**

- **Stiffness modeling:** The beam stiffnesses are calculated assuming ideal Euler-Bernoulli beam theory with perfect clamping conditions. In reality, connection points have finite stiffness and flexibility not captured in the model.

- **Mass distribution:** The model assumes concentrated point masses, neglecting the distributed mass of beams, connectors, and accelerometers (73g each). This affects the effective mass matrix and natural frequencies.

- **Boundary conditions:** The theoretical model assumes a perfectly rigid floor fixation. The experimental setup may have finite foundation stiffness or mounting compliance.

- **Material properties:** Young's modulus E = 200 GPa is a nominal value. Actual material stiffness may vary due to manufacturing tolerances, temperature effects, or material inhomogeneity.

### 1.2 Peak Amplitude Differences

**Observation:** FRF peak magnitudes differ between theoretical and experimental results, with some modes showing larger discrepancies than others.

**Justification:**

- **Damping estimation uncertainty:** Rayleigh damping (**D** = α**M** + β**K**) is a simplified model that assumes proportional damping. Real damping mechanisms are more complex:
  - Material (viscoelastic) damping in beams
  - Joint/connection damping at mass-beam interfaces
  - Air damping (negligible but present)

- **Modal damping variation:** The mixed method (logdec for modes 1-3, least squares for modes 4-6) introduces different estimation errors per mode. Higher modes are more sensitive to measurement noise.

- **Non-proportional damping:** Real structures exhibit non-proportional damping where different components have different damping characteristics. Rayleigh damping cannot capture this complexity.

### 1.3 Missing Mode 2 Peak in Theoretical FRFs

**Observation:** Mode 2 shows no visible resonance peak in the theoretical FRFs for most masses, particularly for Masses 1-4.

**Justification:**

- **Zero modal participation:** The mode shape for Mode 2 is φ₂ ≈ [0, 0, 0, 0, +1, -1]ᵀ, indicating near-zero displacement at Mass 1 where the excitation force is applied.

- **Modal participation factor:** The participation factor Γ₂ = φ₂ᵀ**F** = 0 when **F** = [1, 0, 0, 0, 0, 0]ᵀ. This mode is not excited by the input force, resulting in no resonance peak.

- **Experimental visibility:** The experimental FRFs may show a small peak at Mode 2 due to imperfect alignment, off-axis force components, or rotational degrees of freedom not captured in the 1D model.

### 1.4 Anti-Resonance Discrepancies

**Observation:** Anti-resonances (FRF minima) occur at different frequencies and depths between theoretical and experimental FRFs.

**Justification:**

- **Mode shape sensitivity:** Anti-resonances are highly sensitive to mode shape accuracy. Small errors in stiffness or mass distribution significantly affect anti-resonance locations.

- **Measurement location effects:** Anti-resonances occur when the response at a specific DOF is zero due to modal cancellation. Small misalignments in accelerometer placement can shift these locations.

### 1.5 High-Frequency Behavior

**Observation:** At higher frequencies (above Mode 6), experimental FRFs show additional dynamics not captured by the model.

**Justification:**

- **Model truncation:** The 6-DOF model only captures the first 6 modes. Higher-order beam bending modes, local vibrations, and additional structural modes exist but are not modeled.

- **Measurement noise:** Experimental FRFs at high frequencies are affected by sensor noise and digitization errors, particularly where response amplitudes are small.

---

## 2. Mode Shape Discrepancies

Mode shapes extracted from experimental FRF peaks show good qualitative agreement with theoretical predictions but exhibit quantitative differences.

### 2.1 Amplitude Differences

**Observation:** Normalized experimental mode shapes show different relative amplitudes compared to theoretical mode shapes.

**Justification:**

- **Measurement noise:** The imaginary component of the H2 FRF estimator at resonance peaks contains measurement noise, affecting mode shape extraction accuracy.

- **Closely-spaced modes:** Modes 1 and 3 are relatively close in frequency. Modal coupling and overlap affect the accuracy of individual mode shape extraction.

- **Peak identification:** Experimental mode shapes are extracted at the identified peak frequencies, which may not exactly coincide with true natural frequencies due to frequency resolution limitations.

### 2.2 Sign and Phase Discrepancies

**Observation:** Some experimental mode shapes require sign flipping to match theoretical predictions (e.g., Modes 2, 3, and 5 were flipped in the Python script).

**Justification:**

- **Phase reference ambiguity:** Mode shapes are defined up to an arbitrary scaling factor and sign. The reference accelerometer (Mass 4) determines the sign convention.

- **H2 estimator phase:** The imaginary part of the FRF at resonance provides mode shape information, but the overall sign depends on the approach direction to resonance and damping characteristics.

### 2.3 Spatial Completeness

**Observation:** Experimental mode shapes are only available at the 6 measurement locations (accelerometer positions).

**Justification:**

- **Limited instrumentation:** Only 6 accelerometers are used, providing discrete point measurements. Continuous mode shapes along beam spans are not captured.

- **Spatial aliasing:** Higher-order mode shapes with rapid spatial variations may be undersampled, leading to apparent differences from theoretical predictions.

### 2.4 Structural Idealization

**Observation:** The digital twin assumes 1D translational motion, while the real structure can exhibit coupled translation-rotation and out-of-plane motion.

**Justification:**

- **DOF reduction:** The 6-DOF lumped-mass model neglects rotational DOFs at each mass and assumes purely translational motion. Real structures have rotational inertia and flexibility.

- **3D effects:** The physical structure can vibrate in out-of-plane directions or exhibit torsional modes. Accelerometers mounted off-axis may capture these components.

- **Connection flexibility:** Beam-to-mass connections are modeled as rigid, but real connections have finite rotational stiffness that affects mode shapes.

---

## 3. Quantitative Assessment

Despite the discrepancies discussed above, the digital twin demonstrates good predictive capability:

- **Frequency errors:** Natural frequency errors are within 1-5%, typical for lumped-parameter models

- **Mode shape correlation:** Visual comparison shows qualitative agreement for all modes, with correct nodal patterns

- **FRF trends:** Peak locations, overall response magnitude, and frequency-dependent behavior match well

---

## 4. Conclusion

The observed discrepancies between digital twin and experimental results arise from fundamental modeling assumptions (lumped masses, proportional damping, 1D motion, ideal boundary conditions) and experimental limitations (measurement noise, finite frequency resolution, discrete sensors). These discrepancies are expected and acceptable for engineering-level modeling. The digital twin successfully captures the dominant dynamic behavior of the 6-mass system and provides valuable insight into modal characteristics, despite not perfectly replicating all experimental details.

Further model refinement could include:
1. Updating mass and stiffness parameters through model calibration
2. Implementing non-proportional damping models
3. Adding rotational DOFs
4. Considering foundation flexibility

However, the current model complexity represents a good balance between accuracy and computational simplicity for the intended application.
