FROM python:3.9
RUN apt update
RUN pip3 install flask
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install tensorflow-cpu
RUN pip3 install tqdm
COPY . /opt/
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["/opt/app.py"]
