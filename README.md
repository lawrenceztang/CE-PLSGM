# Overview
Reproduction code for "CE-PSGM"

# Usage
1. 
```
   cd source_code
   conda create --name ce-plsgm --file requirements-conda.txt
   conda activate ce-plsgm
   pip3 install -r requirements-pip.txt
   ```
2. To run the Python file:
```
python src/train.py --optimizer_name ce_plsgm
```
3. To replicate paper results:
```
./scripts/reproduction.sh
```
4. Run the Jupyter notebook `notebook_for_evaluation.ipynb`.
