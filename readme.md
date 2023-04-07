## install
- first you should have a miniconda or anaconda
- then type `source install.sh`

## run
- (if need, type `conda activate colink-protocol-unifed-tff`)
- type `python test/test_all_config.py`

## result 
- see `0.log`, it's the result for present config file.
- And because now here's not real distributed version of tff, so all matters are implemented in server edge (in `protocol.py` and `workload_sim.py`)
