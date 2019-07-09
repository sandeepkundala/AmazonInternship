# ***PYTHON PROGRAM FOR FEE RECEIPT RECOGNITION***

## **SKILLS GAINED:**

* Dataset generation
* Training model in AWS Environment using MxNet
* Performing inference
* Image Processing

## **PROBLEM STATEMENT:**

Train a model using MxNet and perform OCR for receipts so that the total amount can be accurately recognized and be used in expense report.

## **OBJECTIVE:**
* Generate dataset for training the model with random characters, date etc.
* Train the model using MxNet and AWS Amazon DL AMI
* Perform inference with receipts

## **INPUT:**
Receipts (e-receipts only)

## **OUTPUT:**
Printing the total amount

## **CONSTRAINTS:**
* The receipts should have perfect horizontal allignment.\
* The model wouldn't give results if there are too many words placed closely in a line.

## **FUTURE WORK:**
* Train a model for line segmentation so that the application can support even physical bills.

## **PROCESS:**

### *Data Generation:*
The dataset was generated using modified version of receipt-scanner by [Yang Zhuohan](https://github.com/billstark) which is available at https://github.com/sandeepkundala/receipt-scanner (modified version). This particular GitHub repo helps to generate random texts including date.

To setup: `git clone https://github.com/sandeepkundala/receipt-scanner.git --recursive`

To generate data: `python3 draw_receipt.py 100` -> this generates 100 images of each category. The images are saved at result_test4 directory. Compress and zip the folder to generate result_test4.zip

### *Model Training:*
For model training, I have used [this](https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet.git) repository available [here](https://github.com/sandeepkundala/Text-recognition-for-apache-mxnet/blob/master/3b_text_recon.ipynb) to train model for my use case which is expense report application. I have trained the model using custom dataset in AWS Amazon DL AMI P2-16 instance with close to 12000 images. Around 3000 images were used for testing.

In the AWS Instance, git clone [this](https://github.com/sandeepkundala/Text-recognition-for-apache-mxnet.git) repository.

To move the file from your local computer to AWS EC2 instance use `scp -i "key.pem" receipt-scanner/ReceiptGenerator/results_test4.zip ec2-user@<...>:/home/ec2-user/handwritten-text-recognition-for-apache-mxnet`

Unzip the file using `unzip results_test4.zip`.

To train the model, first activate the mxnet in EC2 instance using `source activate mxnet_p36`.

Now to train the model, use `python3 modeltraining.py`.

The model would be saved in .params and .json format.

To run inference on local station or anywhere else, copy both .params and .json files to local PC by running the following in the local terminal where .pem file is present.
`scp -i "AmazonDL.pem" ec2-user@ec2-<...>:/home/ec2-user/handwritten-text-recognition-for-apache-mxnet/<filename> ./Downloads`

### *Inference with bill receipts:*

![](https://github.com/sandeepkundala/AmazonInternship/blob/master/OCR/Python/amazon_internship_fig_1.png)

