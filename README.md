# bash_scripting_helps!-

Bash scripting can significantly streamline your data science and machine learning projects.

Consider a dataset \( X \) on which you want to apply a series of structurally similar algorithms, $\text{Alg}_1, \text{Alg}_2, \ldots, \text{Alg}_m$. By "structurally similar," I mean that these algorithms share substantial similarities in their hyperparameters, such as their inputs. For example, tree-based algorithms for tabular datasets—like XGBoost, Random Forest, and CatBoost—have similar hyperparameters, including `num_estimators` and `max_depth`. The outputs typically include figures, spreadsheets, logs (which can record errors during execution), F1 scores, and more.

Consider the scenario where you want to use the aforementioned tree-based algorithms to explore a range of hyperparameters on dataset $X$ for a classification task, and you need to save all resulting outputs in a designated directory. Common issues arise in this process:

1. How should you structure directories to clearly associate outputs with their corresponding hyperparameters? What if you need to run multiple experiments and want to store results for future comparison?
2. How can you avoid accidentally overwriting previous results and wasting computational resources?
3. What if you need to make minor adjustments to your algorithms and rerun the experiments? Creating output directories manually each time is cumbersome. For instance, if you want to apply Radial Basis Function for dimensionality reduction, you’d like to have an output directory named `rbf` created within the main output directory.
4. What if you realize that the formatting of figures (e.g., titles, sizes) is off and needs adjustment across all outputs? Should you manually edit each Python file to change the title font size? This approach is inefficient and can lead to inconsistencies.

These issues are familiar to anyone with at least five years of coding experience. Time is often wasted opening File Explorer, creating output directories, and manually updating paths in Python files. Even more experienced developers may address this within the Python code, but managing numerous files scattered across a working directory can still be exhausting.

Let’s explore how bash scripting can save you time and streamline this process.

Assume you’re working in a directory named `myproj`. Create two subfolders: `data` and `src`. Place your dataset $X$ in `myproj/data` and all Python files with your algorithms in `myproj/src`. In `myproj`, only include the bash scripts used to execute the Python files. These bash scripts should:

1. Define all the necessary hyperparameters for the Python files. For example, specify parameters like the number of estimators for your random forest algorithm. To handle a range of values, define variables such as `start_num_estimators`, `end_num_estimators`, and `step_num_estimators`.
2. Set paths for the output directories. Use commands to create these directories if they do not already exist, eliminating the need to manually create them in File Explorer. For instance, to run experiments on a dimensionally reduced dataset, modify the output directory using a simple conditional statement. If `RBF=0`, set `OUTPUTDIR=outputs/tree_based/model_name`; otherwise, set `OUTPUTDIR=outputs/tree_based/model_name/rbf`.

You can pass the model name as an argument to your bash script. For example, use the command `bash driver_treebased.sh xgboost` to execute the script with `xgboost` as the model name. This model name will be incorporated into `OUTPUTDIR` and other necessary places.

In the Python files, include an argument parser to handle the arguments from the bash scripts. Simply parse these arguments, and you’re set! By running the bash scripts, you can save all outputs efficiently. This approach allows you to modify just one bash file to experiment with different hyperparameters.
