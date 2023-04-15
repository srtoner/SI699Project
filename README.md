# SI699Project
Group Project for SI 699: Project Gutenberg Analysis

Directory Structure:

To recreate the dataset used in this analysis:

1. Install pgcorpus and run get_data.py as well as process_data.py

2. Using the metadata.csv, run ProcessMetadata.py

3. Create the seed set by running SampleText.py; beware that if using the gutenbergpy dataset. some of the E-text ids are inaccurate. It is better to simply download the data itself

4. Run Cleaning.ipynb to join with the metadata

5. Run Prune.ipynb to cut down data to the appropriate size
