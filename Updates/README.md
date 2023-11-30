# Updates:
- **6.3.2023**
    - Have updated the repo directories and included the works of [GeoMol](/Codes/others_approaches/conformation_generation/GeoMol) and [spherical 3D molecular embeddings](/Codes/others_approaches/embeddings_nets/DIG). The current task is to make the code of [GeoMol](/Codes/others_approaches/conformation_generation/GeoMol) up to date with the newest modules to be able to run the codes together as one module for 3-D conformation generation. 
    - Need to update GeoMol to have pytorch version of 1.10.0 and torch geometric of 2.0.0 so that it is compatable with the spherical 3-D embeddings. 
    - Now creaet an environment with variabls from spherical neural nets and try to run the GeoMol code using the new environment from [yml file](/Codes/others_approaches/embeddings_nets/DIG/docs/environment.yaml). 
- **7.3.2023**
    - Starting to debug the GeoMol in order to make it compatible with the DIG library so I can later test them together.
    - Currently stuck at a step where the DataLoader object stacks the molecules together as it raises "No matching tensors's sizes error" which is similar to the error found [here](https://forums.fast.ai/t/runtimeerror-in-dataloader-worker-process-0/56019). It seems that changing how the DataLoader gets initialized is the key.
- **8.3.2023**
    - Found out the reason of the runtime error which is due to the differnet sizes of different molecules. It was the same case for previous versions of Torch_Geometric however it did not cause any errors. 
    - It must be an issue related to how the data is processed or maybe the new versions have ne data types for graphs with different sizes.
    - It was found out that by using old version of torch_geometric that the data still have different shapes. However, it works fine.
- **11.3.2023**
    - Solved one error which was caused by the inconcsestency in the list of molecular positions passed to the data variables. The solustion was to put the max number of conformations as the last dimenssion in the pos tensor.
    - New issue rises now from the dictionary of neighbours in the data variables which is also related to the collate function.
    - Understanding how the collate function works help  in solving these issues. 
- **15.3.2023**
    - I have reached the conclusion that the code from GeoMol is very problematic and difficult to run on newer versions of pytorch Geometric. 
    - The code keeps throwing errors related to how the data is being handeled (pickelingerror, dictionary invalid keys) and this means that the code might have run succefully on the previous library versions due to bugs within the libraries that made it difficult for them to detect those mistakes. However, it generates good molecules and the error values are low which means that even with those errors and code mistakes, the resulting models could work properly (in terms of application and results). 
    - The final decision now is to use the older versions on python modules to recreat the resuolts of GeoMol and then adapt their code into the code of DIG which will make more sense as DIG code already handles the data using the more up to date modules.
- **21.3.2023**
    - Have used the code developed by GeoMol to recreate their results and this is what I got after generating molecules using their models  that I have trained on the cluster:

            -Recall Coverage: Mean = 99.66, Median = 100.00

            -Recall AMR: Mean = 0.2268, Median = 0.1934

            -Precision Coverage: Mean = 99.46, Median = 100.00
            -Precision AMR: Mean = 0.2703, Median = 0.2370
    - And this is what I got after running it using their models' variables:
            
            - Recall Coverage: Mean = 99.66, Median = 100.00
            - Recall AMR: Mean = 0.2268, Median = 0.1934
            - Precision Coverage: Mean = 99.46, Median = 100.00
            - Precision AMR: Mean = 0.2703, Median = 0.2370
    - They are different thean the values provided in their paper which need to be investigated.

    - I have also run the SphereNet using the data from Pytorch_Geometric's QM9 Dataset and it threw no errors. However I need to change how the data is represented to get similar results(Incorporate similar features to the ones they are using but also similar to the structure used in GeoMol code). 
    - I think that GeoMol did not have to use one-hot vector for each of the features but could have used numeric values(to investigate). 
- **28.3.2023**
    - Today I have started building the Dataset clas for qm9 that can connect the world of GeoMol and the Spherical messsage passing. 
        - The class was created to extend the Dataset class from Pytorch Geometric and was tested with the get function successfully.
- **19.4.2023**
    - I have made some changes in the code of GeoMol to use the dataset class created in the Notebook. 
    - The code showing now different types of errors so instead of having a problem with the data type, it is giving this error message: 
        - Traceback (most recent call last):
  File "train.py", line 73, in <module>
    train_loss = train(model, train_loader, optimizer, device, scheduler, logger if args.verbose else None, epoch, writer)
  File "/home/aabdel2s/Documents/masters_work/repo/masters_project/Codes/others_approaches/conformation_generation/GeoMol/model/training.py", line 19, in train
    for i, data in tqdm(enumerate(loader), total=len(loader)):
  File "/home/aabdel2s/anaconda3/envs/DIG-Stable/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 368, in __iter__
    return self._get_iterator()
  File "/home/aabdel2s/anaconda3/envs/DIG-Stable/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 314, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
  File "/home/aabdel2s/anaconda3/envs/DIG-Stable/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 927, in __init__
    w.start()
  File "/home/aabdel2s/anaconda3/envs/DIG-Stable/lib/python3.8/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
  File "/home/aabdel2s/anaconda3/envs/DIG-Stable/lib/python3.8/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
  File "/home/aabdel2s/anaconda3/envs/DIG-Stable/lib/python3.8/multiprocessing/context.py", line 284, in _Popen
    return Popen(process_obj)
  File "/home/aabdel2s/anaconda3/envs/DIG-Stable/lib/python3.8/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/home/aabdel2s/anaconda3/envs/DIG-Stable/lib/python3.8/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/home/aabdel2s/anaconda3/envs/DIG-Stable/lib/python3.8/multiprocessing/popen_spawn_posix.py", line 58, in _launch
    self.pid = util.spawnv_passfds(spawn.get_executable(),
  File "/home/aabdel2s/anaconda3/envs/DIG-Stable/lib/python3.8/multiprocessing/util.py", line 452, in spawnv_passfds
    return _posixsubprocess.fork_exec(
ValueError: bad value(s) in fds_to_keep
- **26.4.2023**
  - The observations from 21.3.2023 were all different than the values reported in the GeoMol paper because the seed value was set to 0. After changeing that value to 5 as mentioned in the paper, similar values were reported for AMR and COV (but only Mean and Median in Å but still different in percentages)
- **28.4.2023**
  - The problem with the percentage is caused by the "threshold" array in the file compare_confs.py. After changeing it, the percentages have changed too. Now is time to figure out which values it should be set to. Which are found to be nearly "arange(0.0, 1.0, 0.05)"

- **15.5.2023**
  - The data processing done in the code from GeoMol has many deficits:
    - It is manually collated, which means that the dataset is not correctly built using Pytorch Geometric's guidlines. Therefore, some of the data objects parts are not possible to include 'in order to correctly implement the approach'. Those parts are (neighbors_dictionary: which can be calculated as part of the models variables instead of as part of the dataset). 
    - The approach is hardcoded to handle a limited set of molecule which is okay for the selected dataset (qm9) but can not be generalized. 
    - They batched the data manually instead of using the batches created by Pytorch Geometric. By batching here we mean gathering different graphs together in one graph in order to use them for training (which is Pytorch Geometric's way of training on data) 
- **20.5.2023**
  - To solve the above mentioned challenges, I tried to follow the running code step by step and fix each error that happens due to the inconcestencies in the dataset. 
  - First part was to remove "neighbors_dictionary" because it caused errors during the batching step. 
  - Then to calculate that missing feature as part of data processing so that it could be used within the model.
  - After that is to include count each stable conformer as a data element and not each set of conformers as a data element.
  - The manual batching of the data and treating each group of conformers as one data element resulted in huge complexity. This complexity can be seen in the dimesnions of the matrices used to calculate different elements needed to build the model.
- **28.5.2023**
  - After a long and deep look at the code developed by GeoMol, it is not clear if modifying it can be a feasible approach.
  - I have worked on it separately from this repo in order to have a clear and isolated look at it. 
  - The approach is so complicated each resulting matrix leads to a new error in the next step as the whole base of the dataset is not following the guidlines by Pytorch Geometric. 
  - I decidec it might be more feasible to reimplement there approach, however this might not result in a working approach and will result in us not being able to try new evaluation crtierea. 
  - The current progress now is that we have a version of the GeoMol's qm9 dataset that can be fed to Sphenet without raising runtime errors. And that dataset version can be the starting point to re-implement the GeoMol correctly using modules from the SpherNet Library.
- **30.5.2023**
  - I have fed the GeoMol's version of qm9 dataset to SphereNet without erros. 
  - However, the results are not readable as we need to specify the dataset's features to be used. 
  - Current goal would be to have a prototype of Spherenet that uses the Geom-QM9 dataset in order to predcit molecular global features.