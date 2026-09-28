
### Main findings

- **Data used:** 135 valid experiments.
- **Particle size:** median **59.63 nm**, mean **78.93 nm**, range **31.16–488.90 nm**.
- **Strongest simple linear correlation:** **FRR** with Pearson r = **-0.288**.
- **Best tested model:** **Gradient Boosting Regressor with Huber loss**.
- **5-fold cross-validation:** **R² = 0.440 ± 0.328** and **MAE = 22.37 ± 8.05 nm**.
- **Strongest predictive parameter:** **FRR**. Shuffling it raises held-out MAE by about **19.1 nm** on average.
- **Feature ranking:** **FRR** is dominant, followed by **F127 concentration**; **TFR** and **AIE concentration** have substantially smaller predictive contributions in this four-variable model.

### Interpretation

The nonlinear model captures useful structure, but the moderate cross-validated R² and substantial fold-to-fold variability show that the four process variables do not explain all particle-size variation. The workbook contains several different AIE molecules, yet molecule identity is intentionally excluded to follow the four-input assignment specification. Therefore, this model should be interpreted as an interpolation model for the represented experiments, not as proof that it will generalize to a completely new AIE chemistry.
