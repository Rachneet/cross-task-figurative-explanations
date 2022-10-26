# Cross-transfer learning for figurative language explanations

## Project description
In this project, we compare sequential fine-tuning with a model for multi-task learning in the context
where we are interested in boosting performance on two tasks, one of which depends on the other. We test 
these models on the *FigLang2022* shared task which requires participants to predict language inference labels on
figurative language along with corresponding textual explanations of the inference predictions.


## Installation

1. Create a virtual environment and activate it
   ```bash
   # create a virtual environment using conda
    conda create -n cross-transfer python=3.7
    conda activate cross-transfer
   ```
2. Install the requirements:

    ```bash
    
    # Libraries for explanation evaluation
        
      git clone https://github.com/google-research/bleurt.git
      cd bleurt
      pip install .
      cd /content/
    
      wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
      unzip BLEURT-20.zip
    
    # Rest of the requirements
    
      pip install -r requirements.txt
    ```

## Contact
For any questions, please contact:

- [Rachneet Sachdeva](mailto:sachdeva@ukp.informatik.tu-darmstadt.de?subject=[GitHub]%20Figurative%20Explanations)
- [Irina Bigoulaeva](mailto:bigoulaeva@ukp.informatik.tu-darmstadt.de?subject=[GitHub]%20Figurative%20Explanations)
- [Harish Tayyar Madabushi](mailto:htm43@bath.ac.uk?subject=[GitHub]%20Figurative%20Explanations)

## Links
- [UKP lab](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp)
- [TU Darmstadt](https://www.tu-darmstadt.de/)

## Disclaimer
This repository contains experimental software and is published for the sole purpose of giving additional
background details on the respective publication. 

## Citation

If you use this code, please cite the following paper:

```text
COMING SOON
```