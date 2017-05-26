# sbm_ina
Stochastic blockmodels for data clustering

## Running
Running graph tool is a huge pain in the ass. The easiest way to get up and running is to use our custom container that pulls in all dependencies and running it with `docker run -it -v /home/pavlin/fri/ina_sbm:/sbm -w /sbm --rm pavlin/sbm bash`. Replace `bash` with whatever your heart so desires e.g. `python test.py`.

Note that this will create a large amount of useless containers, which can be removed with `docker ps -a | awk '/pavlin\/sbm/ { print $1 }' | xargs docker rm`.

## Resources
- https://www.cs.umd.edu/class/spring2008/cmsc828g/Slides/block-models.pdf
- https://graph-tool.skewed.de/static/doc/inference.html#graph_tool.community.minimize_blockmodel_dl
