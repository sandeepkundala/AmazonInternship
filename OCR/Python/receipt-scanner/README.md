# CRNN Receipt Scanner
A machine learning implementation of OCR  
[Project report of original authors](project_report.pdf)

> ### Development Environment
> Python 3+

> ### Installation:
> Clone Repository  
> original: `git clone https://github.com/billstark/receipt-scanner`
>
> modified: `git clone https://github.com/sandeepkundala/receipt-scanner`
>  
> In repository root directory:  
> `pip install -r requirements.txt`

> ### Note:
> The image files are not shared since they take up lot of space. To generate images, run draw_receipts.py program with number X which would produce X images for each category  = ['line', 'date' , 'word', 'word_column', 'word_bracket', 'tax', 'priceL','priceR','totL','totR','price_left','price_right','percentage','float','int']
