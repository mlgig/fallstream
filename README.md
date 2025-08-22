# Watch Your Step: Realistic Fall Detection in Real-world Streaming Scenarios

## Abstract
Real-time fall detection is crucial for enabling timely interventions and mitigating the severe health consequences of falls, particularly in older adults. However, existing methods often rely on simulated data or unrealistic assumptions, limiting their real-world applicability. Practical deployment also requires efficient computation and robust evaluation metrics tailored to continuous monitoring. 

This paper presents a real-time fall detection framework for continuous monitoring without prior knowledge of fall events. Using over 60 hours of inertial measurement unit (IMU) data from the FARSEEING real-world fall dataset, we employ recent efficient classifiers to compute fall probabilities in streaming mode. To enhance robustness, we introduce a cost-sensitive learning strategy that tunes the decision threshold based on a clinically relevant cost function, prioritising timely fall detection while minimising false alarms. 

Unlike many methods that achieve high recall only at the expense of low precision, our framework achieves **perfect recall (1.00)**, **precision of 0.84**, and an **F1 score of 0.91** on FARSEEING, eliminating missed falls while keeping false alarms low. These results demonstrate the effectiveness of cost-sensitive threshold tuning and highlight the potential of our computationally efficient framework for practical, real-time fall monitoring and improved fall management strategies.

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
_Watch Your Step: Realistic Fall Detection in Real-world Streaming Scenarios_. Under review, 2025.

A BibTeX entry will be provided once the paper is accepted. In the meantime, you may use the following placeholder:

```
@article{aderinola2025watchyourstep,
  title   = {Watch Your Step: Realistic Fall Detection in Real-world Streaming Scenarios},
  author  = {Aderinola, Timilehin B. and Palmerini, Luca and D’Ascanio, Ilaria and Chiari, Lorenzo and Klenk, Jochen and Becker, Clemens and Caulfield, Brian and Ifrim, Georgiana},
  journal = {Under review},
  year    = {2025},
  note    = {Preprint; citation details will be updated upon publication}
}
```