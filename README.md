# audiosearch

Interactive query-by-example audio search system. 

Demo video: https://youtu.be/q7nVXnR_DP0

Writeup: https://www.sharelatex.com/project/5697cbbb3e30dc3f56e2544e

package dependencies: 
- numpy, scipy
- soundfile https://github.com/bastibe/PySoundFile

data package installation:
1. Download UrbanSound8K.tar.gz from https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html;
2. Extract the gz file under the directory 'data'.

How to run: 
1. cd to top level directory
2. run: python gui.py username 
(username can be any string)

Note: Due to copyright issue, the original audio of UrbanSound8K dataset is not uploaded. The code can still be run, except that audio cannot be played. 
