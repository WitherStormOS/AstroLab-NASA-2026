# AstroLab-NASA-2026
Lightweight Python simulator for NASA Flame in Freefall: integrates PSI-25/PSI-99 microgravity combustion CSV data, custom EM3 Fluid Velocity Formula, SciPy DOP853 orbital freefall, and spherical flame diagnostics.

## 🚀 Quick Start for NASA Judges (Download & Install)

1. Go to the **[Releases Section](../../releases)** on the right panel of this repository and download **`Astrolab.rar`** (`211 MB`).
2. Extract the `.rar` archive to any folder on your Windows PC.
3. Navigate to **`Installer` → `Output`** and double-click **`AstroLab-Setup`** to install the application.
4. For detailed usage and NASA CSV data instructions, open **`docs/PETUNJUK_DATA_FLAME_FREEFALL`**.
---

## ⚖️ Scientific Transparency & Usage Boundaries (Disclaimer)

To ensure strict scientific integrity and transparency for the **NASA International Space Apps Challenge 2026** evaluation:

* **Custom Mathematical Formulation:** The three-sector **EM3 Fluid Velocity Formula** implemented in this simulator is a custom theoretical and computational modeling equation conceived by the project developer. Computed force outputs (`Scenario F` in Newtons) and scaled integer values (`F_int`) represent deterministic software simulation outputs rather than direct physical load-cell measurements from a spacecraft.
* **Sample Data Provenance:** The bundled sensor dataset (`data/flame_freefall_timeseries.csv`) models time-series structures referencing official microgravity combustion investigations from the **NASA Physical Sciences Informatics (PSI)** repository—specifically **BASS-II (`PSI-25`)** and **SAFFIRE-II (`PSI-99`)**. The software does not independently verify physical hardware calibration certificates.
* **Diagnostic Heuristics:** Fire safety status indicators (`IDLE`, `STABLE`, `WATCH`, `CRITICAL`) operate as software demonstration rules for exploratory visualization and do not replace flight-certified spacecraft fire detection hardware or laboratory safety protocols.

### 🤖 AI / Super Intelligence (SI) Acknowledgment
* **Human-Led Core Design:** The project concept, three-sector *EM3 Fluid Velocity Formula*, system architecture, and orbital/combustion simulation workflows were designed and directed by the human project lead (**Chief Engineer — EM3 Development Group**).
* **SI-Assisted Engineering:** Large Language Model (LLM) assistants were utilized strictly as an engineering aid for Python code refactoring, automated unit-test structuring, English technical documentation translation, and CSV schema formatting. All visual screenshots included in the documentation are authentic captures of the running application.
