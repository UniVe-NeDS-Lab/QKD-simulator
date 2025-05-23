The following code reproduces the results in the paper. These are the steps needed to preproduce them:

0) install the dependencies. Essentially, the distfit and networkx Python packages. If you need specific versions you can use the provided requirements.txt file

1) download the visibility graphs from https://zenodo.org/records/4905536/files/visibility_graphs.zip?download=1. Unzip the data, this will create the visibility_graphs folder

2) run the fit-link-length.py script. You can use the --runs and --graph_size to modify the number of random graphs to fit the distribution and the size of the sampled graphs. Defaulst (10000, 30) are the values used in the paper. The random seeds can be controlled to create exactly the same graphs, use --seed 0 to replicate the results of the paper. On a PC with 64G of RAM and 8 cores this takes a few minutes. 

This will generate a './results/' folder containing:
 -  9 files, for each of the scenario you will have one .csv file that reports the results of the distfit Python package, a .png with the graphical representation of the pdf, and a .gnuplot file with data ready to be plotted.
 - all the generated (10000!) graphs in the .graphml format

3) move to the scripts folder and launch gnuplot plot_pdf.gnuplot. This will recreate the graph of the paper. Note however that Networkx seems to save graphs in .graphml format in a non-deterministic way, so the graphs are isomorphic but the actual file may differ.


If you just want to just reuse the probability distrbutions, scripts/replot.py file shows how to regenerate the PDF functions.
