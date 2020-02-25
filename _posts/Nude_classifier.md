<a href="https://colab.research.google.com/github/lahorekid/cnn/blob/master/Nude_classifier.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
pip install git+https://github.com/bedapudi6788/NudeNet
```

    Collecting git+https://github.com/bedapudi6788/NudeNet
      Cloning https://github.com/bedapudi6788/NudeNet to /tmp/pip-req-build-pj306xbb
      Running command git clone -q https://github.com/bedapudi6788/NudeNet /tmp/pip-req-build-pj306xbb
    Requirement already satisfied: keras==2.2.4 in /usr/local/lib/python3.6/dist-packages (from NudeNet==1.0.4) (2.2.4)
    Collecting opencv-python==4.0.0.21 (from NudeNet==1.0.4)
    [?25l  Downloading https://files.pythonhosted.org/packages/37/49/874d119948a5a084a7ebe98308214098ef3471d76ab74200f9800efeef15/opencv_python-4.0.0.21-cp36-cp36m-manylinux1_x86_64.whl (25.4MB)
    [K     |████████████████████████████████| 25.4MB 1.3MB/s 
    [?25hCollecting keras-retinanet==0.5.0 (from NudeNet==1.0.4)
    [?25l  Downloading https://files.pythonhosted.org/packages/28/bc/1e926156e950073af90f9347cb74bf4a75e749942e75f398472bad7ef146/keras-retinanet-0.5.0.tar.gz (59kB)
    [K     |████████████████████████████████| 61kB 22.2MB/s 
    [?25hRequirement already satisfied: keras-preprocessing>=1.0.5 in /usr/local/lib/python3.6/dist-packages (from keras==2.2.4->NudeNet==1.0.4) (1.0.9)
    Requirement already satisfied: h5py in /usr/local/lib/python3.6/dist-packages (from keras==2.2.4->NudeNet==1.0.4) (2.8.0)
    Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.6/dist-packages (from keras==2.2.4->NudeNet==1.0.4) (1.12.0)
    Requirement already satisfied: keras-applications>=1.0.6 in /usr/local/lib/python3.6/dist-packages (from keras==2.2.4->NudeNet==1.0.4) (1.0.7)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.6/dist-packages (from keras==2.2.4->NudeNet==1.0.4) (3.13)
    Requirement already satisfied: numpy>=1.9.1 in /usr/local/lib/python3.6/dist-packages (from keras==2.2.4->NudeNet==1.0.4) (1.16.3)
    Requirement already satisfied: scipy>=0.14 in /usr/local/lib/python3.6/dist-packages (from keras==2.2.4->NudeNet==1.0.4) (1.2.1)
    Collecting keras-resnet (from keras-retinanet==0.5.0->NudeNet==1.0.4)
      Downloading https://files.pythonhosted.org/packages/76/d4/a35cbd07381139dda4db42c81b88c59254faac026109022727b45b31bcad/keras-resnet-0.2.0.tar.gz
    Requirement already satisfied: cython in /usr/local/lib/python3.6/dist-packages (from keras-retinanet==0.5.0->NudeNet==1.0.4) (0.29.7)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.6/dist-packages (from keras-retinanet==0.5.0->NudeNet==1.0.4) (4.3.0)
    Requirement already satisfied: progressbar2 in /usr/local/lib/python3.6/dist-packages (from keras-retinanet==0.5.0->NudeNet==1.0.4) (3.38.0)
    Requirement already satisfied: olefile in /usr/local/lib/python3.6/dist-packages (from Pillow->keras-retinanet==0.5.0->NudeNet==1.0.4) (0.46)
    Requirement already satisfied: python-utils>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from progressbar2->keras-retinanet==0.5.0->NudeNet==1.0.4) (2.3.0)
    Building wheels for collected packages: NudeNet, keras-retinanet, keras-resnet
      Building wheel for NudeNet (setup.py) ... [?25l[?25hdone
      Stored in directory: /tmp/pip-ephem-wheel-cache-6gfy35bg/wheels/34/0e/81/b084279dbf7ca80c2648dc54e9425930d68b909743c02f7556
      Building wheel for keras-retinanet (setup.py) ... [?25l[?25hdone
      Stored in directory: /root/.cache/pip/wheels/cf/f6/a0/c5b176d6bcfd610872135192fbfb28187daf3b852893ae6eb8
      Building wheel for keras-resnet (setup.py) ... [?25l[?25hdone
      Stored in directory: /root/.cache/pip/wheels/5f/09/a5/497a30fd9ad9964e98a1254d1e164bcd1b8a5eda36197ecb3c
    Successfully built NudeNet keras-retinanet keras-resnet
    [31mERROR: albumentations 0.1.12 has requirement imgaug<0.2.7,>=0.2.5, but you'll have imgaug 0.2.8 which is incompatible.[0m
    Installing collected packages: opencv-python, keras-resnet, keras-retinanet, NudeNet
      Found existing installation: opencv-python 3.4.5.20
        Uninstalling opencv-python-3.4.5.20:
          Successfully uninstalled opencv-python-3.4.5.20
    Successfully installed NudeNet-1.0.4 keras-resnet-0.2.0 keras-retinanet-0.5.0 opencv-python-4.0.0.21



```python
from google.colab import files
from IPython.display import Image
```


```python
uploaded = files.upload()
```



<input type="file" id="files-150d96ea-8a6d-4f70-beae-a2c96a8fe058" name="files[]" multiple disabled />
<output id="result-150d96ea-8a6d-4f70-beae-a2c96a8fe058">
 Upload widget is only available when the cell has been executed in the
 current browser session. Please rerun this cell to enable.
 </output>
 <script src="/nbextensions/google.colab/files.js"></script> 


    Saving nude.jpg to nude.jpg



```python
Image("nude.jpg", width=600)
```




![jpeg](output_4_0.jpg)




```python
# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scope=email%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&response_type=code
    
    Enter your authorization code:
    ··········
    Mounted at /content/drive



```python
# After executing the cell above, Drive
# files will be present in "/content/drive/My Drive".
!ls "/content/drive/My Drive"
```

     Abu
     Altair-updated.ipynb
     classifier_model
    'coding portal.gdoc'
    'Colab Notebooks'
    'Copy of Final Report.gdoc'
    'Copy of ISDS 551 project.gdoc'
    'Copy of ISDS 556 Data Warehousing Project Report 2.gdoc'
    'Healthcare IT and Analytics Seminar - Autumn 2015 (003).pdf'
    'In-n-Out ISDS 556.gdoc'
     ISDS551-GroupProject-AppendixC
    'ISDS 551 project.gdoc'
    'ISDS 553 work folder'
    'Marketing 351'
     Part2Task2_scope_statement.docx
    'Presentation 2.gslides'
    'Presentation 2.pptx'
    'Previous courses'
    'Project 2 - 551.gsheet'
    'Project 2 - ISDS 551.gdoc'
     Syllabus-final.doc
    'The beginning is always the most challenge thing for this project.docx'
    'Things to do.gdoc'
     Trump.gdoc
    'Untitled presentation.gslides'



```python
!cp "/content/drive/My Drive/classifier_model" "classifier_model"
```


```python
!ls
```

    classifier_model  drive  nude.jpg  sample_data



```python
# Using the classifier
from nudenet import NudeClassifier
classifier = NudeClassifier('classifier_model')
classifier.classify('nude.jpg')
```




    {'nude.jpg': {'nude': 0.9814771, 'safe': 0.018522946}}




```python
# This will prompt for authorization.
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
drive.mount('/content/drive', force_remount=True)
```

    Mounted at /content/drive



```python
!cp "/content/drive/My Drive/detector_model" "detector_model"
```


```python
!ls
```

    classifier_model  detector_model  drive  nude.jpg  sample_data



```python
from google.colab.patches import cv2_imshow
```


```python
from nudenet import NudeDetector
detector = NudeDetector('detector_model')

# Performing detection
detector.detect('nude.jpg')
# [{'box': [352, 688, 550, 858], 'score': 0.9603578, 'label': 'BELLY'}, {'box': [507, 896, 586, 1055], 'score': 0.94103414, 'label': 'F_GENITALIA'}, {'box': [221, 467, 552, 650], 'score': 0.8011624, 'label': 'F_BREAST'}, {'box': [359, 464, 543, 626], 'score': 0.6324697, 'label': 'F_BREAST'}]


```

    /usr/local/lib/python3.6/dist-packages/keras/engine/saving.py:292: UserWarning: No training configuration found in save file: the model was *not* compiled. Compile it manually.
      warnings.warn('No training configuration found in save file: '





    [{'box': [624, 334, 806, 452], 'label': 'BUTTOCKS', 'score': 0.6763165}]




```python
from nudenet import NudeDetector
detector = NudeDetector('detector_model')

# Censoring an image
detector.censor('nude.jpg',out_path='new.jpg', visualize=False)

```

    /usr/local/lib/python3.6/dist-packages/keras/engine/saving.py:292: UserWarning: No training configuration found in save file: the model was *not* compiled. Compile it manually.
      warnings.warn('No training configuration found in save file: '



```python
Image("new.jpg", width=600)
```




![jpeg](output_17_0.jpg)


