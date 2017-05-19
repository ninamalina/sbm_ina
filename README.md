# sbm_ina
Stochastic blockmodels for data clustering

## Running
Running graph tool is a huge pain in the ass. The easiest way to get up and running is by downloading the docker container with `docker pull tiagopeixoto/graph-tool` and then running it with `docker run -it -v /home/pavlin/fri/ina_sbm:/sbm -w /sbm tiagopeixoto/graph-tool`. Replace `bash` with whatever your heart so desires.

### TODO
Be aware that you will need to install the dependencies inside each container (using `pip install -r requirements.txt`). This could be  achieved with a custom docker image.

## Resources
- https://www.cs.umd.edu/class/spring2008/cmsc828g/Slides/block-models.pdf
- https://graph-tool.skewed.de/static/doc/inference.html#graph_tool.community.minimize_blockmodel_dl
