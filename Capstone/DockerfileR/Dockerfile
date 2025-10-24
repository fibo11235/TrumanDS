FROM rocker/tidyverse:latest

RUN apt-get update && apt-get install -y python3-pip && pip3 install virtualenv

#RUN pip3 install virtualenv

RUN apt-get update && \
  install2.r --deps TRUE keras \
  tidymodels \
  sf \
  terra \
  spData \
  rpostgis \
  leaflet \
  plotly \
  leaflet.extras \
  quanteda \
  reshape2 \
  tidytext \
  sparklyr \
  rnaturalearth \
  rnaturalearthdata \
  multidplyr


RUN apt-get clean all && \
	apt-get update && \
	apt-get upgrade -y && \
	apt-get install -y \
		libproj-dev \
		libgeos-dev \
		libgdal-dev \
		libudunits2-dev \
		texlive-latex-base \
		texlive-latex-extra \
		texlive-latex-recommended \
		texlive-fonts-recommended \
		libgsl-dev \
		openjdk-11-jdk \	
	&& apt-get clean all && \
	apt-get purge && \
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

