#### Set Base image
##thudergbm恰好cuda9.0
FROM   registry.cn-shanghai.aliyuncs.com/tcc-public/python:3
# 使用阿里云提供的镜像
# Basic Imgage list from aliyun.
# 链接[https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586973.0.0.5c315cfdPLegap&postId=67720]

# Image Creator Information
MAINTAINER HongGuang

# Image Information
LABEL "version"="1.0"
LABEL "description"="First image to tianchi."
RUN uname -a
COPY sources.list /
### 如果用自建基础镜像就不用下面这两个 
#RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
#mv /sources.list /etc/apt/ && apt-get update -y
# Install Basic Env
#RUN apt-get install  -y  --no-install-recommends git cmake build-essential libboost-dev libboost-system-dev libboost-filesystem-dev
 
# Install python libs.
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip&& pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy  pandas  sklearn  catboost && \
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  lightgbm==2.2.3 && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cleanlab
# Set work dir
WORKDIR /

# Copy files to image.
COPY run.sh /
COPY common.py / 
COPY main.py /
COPY predict.py /
COPY train.py  /
CMD /run.sh
