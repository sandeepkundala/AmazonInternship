# Text Recognition (OCR) with MXNet Gluon 

These notebooks (except 3a_handwriting_recognition.ipynb and ImgProc.ipynb) have been created by [Jonathan Chung](https://github.com/jonomon), as part of his internship as Applied Scientist @ Amazon AI, in collaboration with [Thomas Delteil](https://github.com/ThomasDelteil) who built the original prototype.

I have used [this](https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet.git) repository to train model for my use case which is expense report application. I have trained the model using custom dataset in AWS Amazon DL AMI P2-16 instance with close to 24000 images. Around 6000 images were used for testing.

I have modified few programs for my own requirement of training model for OCR. The dataset was generated using modified version of receipt-scanner by [Yang Zhuohan](https://github.com/billstark) which is available at https://github.com/sandeepkundala/receipt-scanner (modified version). This particular GitHub repo helps to generate random texts including date. Also, this repository helps to create bill receipts which would be used for inference of the OCR Model.

## Setup

`git clone https://github.com/sandeepkundala/Text-recognition-for-apache-mxnet.git --recursive`

`git clone https://github.com/sandeepkundala/receipt-scanner.git --recursive`

## To generate your own dataset

In receipt-scanner run draw_receipt.py which is present in ReceiptGenerator folder.

The command is `python3 draw_receipt.py 100` 

to generate 100 sample images for each type like 'word', 'word_column', 'word_bracket', 'int', 'float', 'price_left', 'price_right', 'percentage','line', 'date'. 

The images are saved in results_test folder. The bill receipt images are stored in results folder.

## Overview 

![](https://cdn-images-1.medium.com/max/1000/1*nJ-ePgwhOjOhFH3lJuSuFA.png)

The pipeline is composed of 2 steps:
- Detecting lines of texts [[blog post](https://medium.com/apache-mxnet/handwriting-ocr-line-segmentation-with-gluon-7af419f3a3d8)], [[jupyter notebook](https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet/blob/master/2_line_word_segmentation.ipynb)], [[python script](https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet/blob/master/word_and_line_segmentation.py)]
- Recognising characters and applying a language model to correct errors. [[blog post](https://medium.com/apache-mxnet/handwriting-ocr-handwriting-recognition-and-language-modeling-with-mxnet-gluon-4c7165788c67)], [[jupyter notebook](https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet/blob/master/3_handwriting_recognition.ipynb)], [[python script](https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet/blob/master/ocr/scripts/handwriting_line_recognition.py)]

The entire inference pipeline can be found in this [notebook](https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet/blob/master/0_handwriting_ocr.ipynb). See the *pretrained models* section for the pretrained models.

A recorded talk detailing the approach is available on youtube. [[video](https://www.youtube.com/watch?v=xDcOdif4lj0)]

The corresponding slides are available on slideshare. [[slides](https://www.slideshare.net/apachemxnet/ocr-with-mxnet-gluon)]

## Appendix

### 1) Line Detection


### 2) Text recognition

##### Model architecture

![](https://cdn-images-1.medium.com/max/800/1*JTbCUnKgAySN--zJqzqy0Q.png)



