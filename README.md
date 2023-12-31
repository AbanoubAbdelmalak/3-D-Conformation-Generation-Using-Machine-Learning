# Masters_project

This repo contains all the work of my masters thesis project **3-D Molecular Conformation Generation Using Machine Learning** The structure as as the following:
<br>
- **Codes**: This directory contains all the modeules I use and also developped.
    - Codes is split into different irectories:
    - [My_Codes](/Codes/My_Codes/): Contains codes developed by me.
        - [dataset_class](/Codes/My_Codes/Project/data_process/qm9_dataset.py): this is the dataset class mentioned in the methodology.
        - [testing spherenet and the dataset](/Codes/My_Codes/Project/data_process/geom_qm9.ipynb.py)
        - [SphereNet as embeddnings layer](/Codes/My_Codes/Project/embeddings/Embed_SphereNet.py): This is a work i progress of a class that extends SphereNet and could be used as embeddings layer in a context of conformation generation.
        - [visualization_notebook](/Codes/My_Codes/Project/visualization_code.ipynb): This jupyter notebook is built to visualize the generated conformations and the refernce conformations.
    - [others_approaches](/Codes/others_approaches/): Contains other modules to be used or edited according to the project.
        - [GeoMol_Legacy](/Codes/others_approaches/conformation_generation/GeoMol_Legacy/): This directory contains the GeoMol code, that is minimally edited to run on the original environment described in the thesis as legacy environment. It also contained the generated data from the reproducability test in folder [test_0](/Codes/others_approaches/conformation_generation/GeoMol_Legacy/test_0/). And the data generated by the original approach in trained_models dirctory. 
        -  [GeoMol-Updated]: This contains the code after iterations of editing trying to extend it to work on the newer environment. The edits are described in the report and can be seen in the faturization.py file and utils file.
        - [DIG](/Codes/others_approaches/embeddings_nets) is where the installation of [DIG](https://github.com/divelab/DIG) module should be installed in order to use run [my code](/Codes/My_Codes) successfully.
    - The datasets could not be included here and added to the .gitignore file as they are huge and it makes more sense to download them when you need to test the codes from [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).
    - The conda environments are [GeoMol](/Codes/GeoMol.yml) which is the legacy environment that runs the original GeoMol code successfully. and [DIG](/Codes/DIG.yml) is the file containing the environment we used to test geoMol extensibility.
- **Updates**: Contains a readme file which contains all the updates and thought process throughout the project work.
- Any mention of GeoMol or SphereNet in the code is refering to the to papers mentioned in the references list below: 
- **References**
    -[Ganea et al., 2021] Octavian Ganea, Lagnajit Pattanaik, Connor Coley, Regina Barzilay, Klavs Jensen, William Green, and Tommi Jaakkola. "GeoMol: Torsional Geometric Generation of Molecular 3D Conformer Ensembles." Advances in Neural Information Processing Systems, 34 (2021). Cited by 16. [Code Repository](https://github.com/PattanaikL/GeoMol)
    - [Liu et al., 2021] Yi Liu, Limei Wang, Meng Liu, Xuan Zhang, Bora Oztekin, and Shuiwang Ji. "Spherical Message Passing for 3D Graph Networks." arXiv preprint arXiv:2102.05013 (2021). Available at: [https://github.com/divelab/DIG](https://github.com/divelab/DIG)


<br>

