# GroupAdapter


##Dataset
1.The datasets folder contains four datasets: Weibo2017,WeChat,Twitter,CHECKED.
Weibo2017 come from Title[EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection](https://dl.acm.org/citation.cfm?id=3219819.3219903)
WeChat come from Title[Weak Supervision for Fake News Detection via Reinforcement Learning](https://arxiv.org/abs/1912.12520)
Twitter come from [MediaEval2015](https://github.com/MKLab-ITI/image-verification-corpus)
CHECKED come from Title[CHECKED: Chinese COVID‑19 fake news dataset](https://arxiv.org/pdf/2010.09029v2.pdf)

2.dataformat:|text|img_vec|label|event_label|,weibo2017 and Twitter have event_label,WeChat and CHECKED do not have event_label.

3.we process datasets to ensure every text has one image tensor. process results can be listed:

| Dataset | real_image | fake_image | train_text | validation_text |
|---|---|---|---|---|
| Weibo2017 | 1489 | 1861 | 6528 | 1465 |
| WeChat | 7040 | 1814 | 8470 | 2117 |
| Twitter | 360 | 50 | 11468 | 2867 |
| CHECKED | 1144 | 53 | 958 | 239 |



##  Project structure


│  train_EANN.py
│  train_MCAN.py
│  
├─datasets
│  │  data_statistic.txt
│  │  
│  ├─CHECKED
│  │      train.pkl
│  │      validation.pkl
│  │      
│  ├─mediaeval2015
│  │      data_statistics.txt
│  │      train.pkl
│  │      validation.pkl
│  │      
│  ├─WeChat
│  │      data_count.txt
│  │      train.pkl
│  │      validation.pkl
│  │      
│  └─weibo2017
│          train.pkl
│          validation.pkl
│          
├─EANN_checkpoint
│  └─seed_2021
│      └─epochs_80
│              11.pth
│              
├─EANN_result
│      seed_2021_epochs_80_weibo2017.txt
│      seed_2022_epochs_80_weibo2017.txt
│      seed_2023_epochs_80_weibo2017.txt
│      
├─Fig
│      GroupAdapter_vs_Adapter_Twitter.png
│      GroupAdapter_vs_Adapter_WeChat.png
│      MCAN_CHECKED_Epoch_Loss.png    
│      
├─models
│  │  Adapter.py
│  │  basebert.py
│  │  DCT_module.py
│  │  EANN.py
│  │  HMCAN.py
│  │  image_backbone.py
│  │  MCAN.py
│  │  res50_backbone.py
│  │  vgg19.pth
│          
├─utils
│  │  data_loader.py
│  │  plot_curve.py
│  │  stop_words.txt
│  │  train_utils.py
│             
└─vocab
    ├─bert
    │  ├─bert-base-uncased
    │  │      config.json
    │  │      pytorch_model.bin
    │  │      vocab.txt
    │  │      
    │  └─chinese_roberta_wwm_base_ext_pytorch
    │          config.json
    │          pytorch_model.bin
    │          vocab.txt
    │          
    └─w2v
            tencent_embedding_zh_d100_words100k.pkl
            tencent_embedding_zh_d200_words10k.pkl
            w2v.pickle