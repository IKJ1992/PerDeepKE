[![PyPI - Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://pypi.org/project/keybert/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MaartenGr/keybert/blob/master/LICENSE)
[![PyPI - PyPi](https://img.shields.io/pypi/v/PerDeepKE)](https://pypi.org/project/PerDeepKE/)

<img src="images/logo.png" width="35%" height="35%" align="right" />

# PerDeepKE

PerDeepKE is a minimal, easy-to-use, and self-supervised Persian keyword extractor library with deep learning techniques such as transformer-based embeddings to retrieve keywords most similar to your input document.

## Installation
Installation can be done using [pypi](https://pypi.org/project/PerDeepKE/):

```
pip install perdeepke
```


## Usage
Here is an example of how PerDeepKE can be used:
```python
from PerDeepKE import Keyword_Extraction

text = "بر اساس تحلیل نقشه‌های همدیدی و آینده‌نگری سازمان هواشناسی امروز در استان‌های ساحلی دریای خزر، اردبیل، شمال آذربایجان شرقی و ارتفاعات البرز مرکزی بارش باران، همراه با وزش باد شدید موقتی و کاهش نسبی دما پیش‌بینی شده است. فردا از میزان بارش‌های این مناطق کاسته شده و فقط در سواحل شمالی بارش پراکنده روی می‌دهد."

ke = Keyword_Extraction(text, segment_num=2)

for word, score in ke.top_words(num=10):
        print(word, score)
```

## TODO
> coming soon
