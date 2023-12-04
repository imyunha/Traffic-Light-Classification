## Installation
    conda create -n tensorflow_gpu pip python=3.5
    source activate tensorflow_gpu
    pip install --upgrade tensorflow-gpu==1.3.0
    pip3 install -r lane-detection-model/requirements.txt 


## Test
   ./run.sh
## NVProf
   ./run_nvprof_time.sh
