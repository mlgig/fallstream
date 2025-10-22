# Watch Your Step: A Cost-Sensitive Framework for Accelerometer-Based Fall Detection in Real-World Streaming Scenarios

## Abstract
Real-time fall detection is crucial for enabling timely interventions and mitigating the severe health consequences of falls, particularly in older adults. However, existing methods often rely on simulated data or assumptions such as prior knowledge of fall events, limiting their real-world applicability. Practical deployment also requires efficient computation and robust evaluation metrics tailored to continuous monitoring.

This paper presents a real-time fall detection framework for continuous monitoring without prior knowledge of fall events. Using over 60 hours of inertial measurement unit (IMU) data from the FARSEEING real-world falls dataset, we employ recent efficient classifiers to compute fall probabilities in streaming mode. To enhance robustness, we introduce a cost-sensitive learning strategy that tunes the decision threshold using a cost function reflecting the higher risk of missed falls compared to false alarms.

Unlike many methods that achieve high recall only at the cost of precision, our framework achieved **Recall of 1.00**, **Precision of 0.84**, and an **F<sub>1</sub> score of 0.91** on FARSEEING, detecting all falls while keeping false alarms low, with average inference time below 5 ms per sample. These results demonstrate that cost-sensitive threshold tuning enhances the robustness of accelerometer-based fall detection. They also highlight the potential of our computationally efficient framework for deployment in real-time wearable sensor systems for continuous monitoring.

---

## Using the Code
The project can be run directly through the Jupyter notebook [`main.ipynb`](./main.ipynb) in the root folder.  

It contains the full pipeline for:
- Loading and preprocessing data  
- Training and evaluating models  
- Running cost-sensitive threshold tuning  
- Visualising detection traces and confusion matrices  

### Requirements
- **Python 3.10.18**  
- Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Citation

If you use this code or results in your research, please cite the following work:

**Timilehin B. Aderinola, Luca Palmerini, Ilaria D’Ascanio, Lorenzo Chiari, Jochen Klenk, Clemens Becker, Brian Caulfield, and Georgiana Ifrim.**
[_Watch Your Step: A Cost-Sensitive Framework for Accelerometer-Based Fall Detection in Real-World Streaming Scenarios_](https://www.arxiv.org/abs/2509.11789). Under review, 2025.

A BibTeX entry will be provided once the paper is accepted. In the meantime, you may use the following placeholder:

```
@article{aderinola2025watchyourstep,
  title   = {Watch Your Step: A Cost-Sensitive Framework for Accelerometer-Based Fall Detection in Real-World Streaming Scenarios},
  author  = {Aderinola, Timilehin B. and Palmerini, Luca and D’Ascanio, Ilaria and Chiari, Lorenzo and Klenk, Jochen and Becker, Clemens and Caulfield, Brian and Ifrim, Georgiana},
  journal = {Under review},
  year    = {2025},
  note    = {Preprint; citation details will be updated upon publication}
}
```
