#!/bin/bash

# Activate the Anaconda environment
source /opt/anaconda3/bin/activate /opt/anaconda3/envs/Biomass_Pipeline

cd /opt/Biomass

# Run each Python script sequentially
/opt/anaconda3/envs/Biomass_Pipeline/bin/python /opt/Biomass/data_extraction.py
/opt/anaconda3/envs/Biomass_Pipeline/bin/python /opt/Biomass/gwr_model.py
/opt/anaconda3/envs/Biomass_Pipeline/bin/python /opt/Biomass/rasterize.py
/opt/anaconda3/envs/Biomass_Pipeline/bin/python /opt/Biomass/import_biomass.py



# Deactivate the Anaconda environment
conda deactivate
