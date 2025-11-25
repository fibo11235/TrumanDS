FROM rocker/tidyverse:latest



#RUN pip3 install virtualenv




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

