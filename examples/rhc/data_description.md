The dataset contains admission records for critically ill 
patients from the SUPPORT study (Connors et al., 1996) with the following structure:

**Treatment (T)**: Binary indicator
* T = 1: RHC administered on day 1 of ICU admission
* T = 0: No RHC administered
The observed treatment rate is {treatment_rate}.

**Covariates (X)**: Demographics (age, sex, race, income, education), primary disease 
category, cancer status, comorbidities (cardiac, pulmonary, renal, liver, malignancy, 
immunosuppression, AMI history), physiological measurements at admission (APACHE score, 
mean blood pressure, heart rate, respiratory rate, temperature, PaO2/FiO2 ratio, 
albumin, creatinine), and organ system involvement indicators (cardiac, respiratory, 
neurological, gastrointestinal, renal, metabolic, haematological, sepsis).

**Outcome (Y)**: Binary indicator
* Y = 1: Death within 30 days of admission
* Y = 0: Survival at 30 days
The baseline 30-day mortality rate is {outcome_mean}.

**Important framing note for policy recommendations**: Treatment (RHC) is an 
individually actionable clinical decision made by the treating physician. Policy 
recommendations should target patient selection criteria for RHC administration, 
identifying subgroups for whom RHC is associated with net benefit versus net harm, 
with the goal of refining clinical protocols for this procedure.
