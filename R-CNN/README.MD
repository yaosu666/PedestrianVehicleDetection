# Pedestrian & Vehicle Detection
This is a repository for a computer vision project targeting for Pedestrian and Vehicle Detection.
## R-CNN
This is a fork of maskrcnn-benchmark repository that is focused on detecting objects from drone view images. The model checkpoint is trained on the VisDrone dataset.

A sample output video: [https://youtu.be/ga5NOm-yL7A](https://youtu.be/ga5NOm-yL7A)

### Setup:

1. pip3 install -r requirements.txt
2. pip install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/torch_stable.html
3. python setup.py build develop

### Demo:

go to drone_demo folder and download pre-trained [weights](https://drive.google.com/file/d/1SCJf2JJmyCbxpDuy4njFaDw7xPqurpaQ/view) , run demo.py file

then will see a demo_result.jpg file.

To run prediction on video, modify the input path in run_on_vid.py

