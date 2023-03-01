# bash_scripting_helps!-

Bash Scripting can help a great deal with your data science, machine learning projects! 

Suppose that I have a dataset X that I wish to implement (structurally similar) algorithms Alg_1, Alg_2, ..., Alg_m on it ( by structurally similar, I mean there exists substantial similarity between these algorithms' hyperparameters i.e., their inputs). As an example for structurally similar algorithms, we may consider tree based algorithms for tabular datasets. Algorithms such as XGBoost, Random Forest and CatBoost all share similar hyperparameters, namely num_estimators, max_depth, etc.  Outputs often include figures, spreadsheets, logs (where you could save errors that happened during the run), f1 scores, etc.   

Now, imagine that, for example, my goal is to use the tree based algorithms mentioned above and run them over a range of hyperparameters for dataset X for the purpose of a classification task and I would like to have all the resulting outputs saved in some output directory. Here there are some common pain points:

a) How should I create these directories so that I know the corresponding hyperparameters for the outputs in it? What if I need to run multiple experiments and wish to have my results saved for comparison in future? 

b) How to not accidentally overwrite the previous results (and waste lots of computation already done)?

c) What if I just need to augment  my algorithms a bit and re-run the entire experiments again? I cannot go through the hassle of creating output directories each time. For instance, if I wish to apply Radial Basis Function to perform some dimensionality reduction, I would like to have an output dir named (say) rbf created inside the output dir where all the results will be saved. 

d) What if, all of a sudden, I realize that the specifics I chose for my figures (e.g., titles, size, etc) sound off and I wish to change them across all my outputs. Should I open up each python file separately and change the title font size? This does seem a lot of work and often leads to inconsistencies.  

These pain points all sound familiar for anyone who has been coding for at least like 5 years! Lots of time wasted in opening up File Explorer, creating the output dir and then heading to the python file to change the path_to_csv variable. A bit more experienced will take care of this inside the python file itself; However, they will still tire themselves out when there are many files scattered inside their working dir. 

Now lets see how bash scripting will save you some good time as I will explain below. 

Suppose that you are inside your working dir. Name it myproj. First, we need two subfolders: data and src. myproj/data contains the dataset X and src contains all the python files that contain my algorithms.  Inside myproj, I would only insert the bash scripts that are used for running the python files. The bash files contain:

a) All the hyperparameters needed inside python files are to be included inside the bash scripts. This could be for example, the number of estimators used inside your random forest algorithm. If you wish to consider a range for your hyperparameter simply define start_num_estimators, end_num_estimators, step_num_estimators. 

b) All the paths for directories we wish to use for outputs. We could simply command it to create the directory if it already does not exist. No need to head over to File Explorer.  This is particularly convenient as for example in item c) above, if we wish to run the same experiments on dimensionally reduced dataset, we could simply alter the output file via a simple if statement. If RBF=0, then OUPUTDIR=outputs/tree_based/model_name. Else  OUPUTDIR=outputs/tree_based/model_name/rbf. 

Note that you could pass the model name you wish to run as an argument to your bash file. For example, the following command can be used to run the driver_treebased.sh file: bash driver_treebased.sh xgboost. This means that I have written inside driver_treebased.sh that xgboost will be considered as my model's name and this will be incorporated inside the OUTPUTDIR and anywhere else that is needed. 

Now heading to the python files, they need to contain an argument parser.  All the arguments defined inside your bash scripts will appear in the name space. You need to 'parse' them. Voila! You are done! Simply run the bash scripts and have all the outputs saved.  As you could see, this way, you could only change one single bash file to experiment with more hyperparameters.



